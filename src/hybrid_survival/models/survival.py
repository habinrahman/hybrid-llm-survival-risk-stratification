"""Cox PH (lifelines) and DeepSurv (PyTorch) with Breslow baseline survival."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")


class CoxModel:
    def __init__(self, penalizer: float = 0.01, l1_ratio: float = 0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model: CoxPHFitter | None = None
        self.feature_names: List[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        y_event: np.ndarray,
        y_time: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> CoxModel:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names = list(feature_names)
        df = pd.DataFrame(X, columns=self.feature_names)
        df["event"] = y_event
        df["time"] = y_time
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model.fit(df, duration_col="time", event_col="event")
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.feature_names is not None
        df = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_partial_hazard(df).values.ravel()

    def predict_survival_function(self, X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.model is not None and self.feature_names is not None
        df = pd.DataFrame(X, columns=self.feature_names)
        surv = self.model.predict_survival_function(df, times=times)
        return surv.T.values


class DeepSurvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU() if activation == "relu" else nn.ELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepSurvModel:
    """
    Deep proportional hazards network.
    Risk output is treated as log-relative-risk eta; survival uses Breslow baseline on training set.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        device: Optional[str] = None,
        early_stopping_patience: int = 0,
        early_stopping_eval_every: int = 1,
        early_stopping_min_delta: float = 0.0,
    ):
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_eval_every = max(1, early_stopping_eval_every)
        self.early_stopping_min_delta = early_stopping_min_delta
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = DeepSurvNet(input_dim, self.hidden_layers, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.feature_names: List[str] | None = None
        self.baseline_times_: np.ndarray | None = None
        self.baseline_cumulative_hazard_: np.ndarray | None = None

    @staticmethod
    def negative_log_likelihood(
        risk_scores: torch.Tensor, y_event: torch.Tensor, y_time: torch.Tensor
    ) -> torch.Tensor:
        sorted_indices = torch.argsort(y_time, descending=True)
        risk_scores = risk_scores[sorted_indices]
        y_event = y_event[sorted_indices]
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8)
        uncensored = risk_scores - log_risk
        return -torch.sum(uncensored * y_event)

    def _concordance_index_numpy(
        self, y_time: np.ndarray, y_event: np.ndarray, risk: np.ndarray
    ) -> float:
        return float(concordance_index(y_time, -risk, y_event))

    def _fit_breslow_baseline(
        self, X_train: np.ndarray, y_event_train: np.ndarray, y_time_train: np.ndarray
    ) -> None:
        eta = self.predict_risk(X_train)
        event_times = y_time_train[y_event_train == 1]
        if len(event_times) == 0:
            self.baseline_times_ = np.array([0.0])
            self.baseline_cumulative_hazard_ = np.array([0.0])
            return
        unique_times = np.sort(np.unique(event_times))
        H = 0.0
        times_out: list[float] = []
        chf_out: list[float] = []
        for tau in unique_times:
            at_risk = y_time_train >= tau
            risk_sum = float(np.sum(np.exp(eta[at_risk])))
            d = int(np.sum((y_time_train == tau) & (y_event_train == 1)))
            if d > 0 and risk_sum > 1e-12:
                H += d / risk_sum
                times_out.append(float(tau))
                chf_out.append(float(H))
        if not times_out:
            self.baseline_times_ = np.array([0.0])
            self.baseline_cumulative_hazard_ = np.array([0.0])
        else:
            self.baseline_times_ = np.asarray(times_out)
            self.baseline_cumulative_hazard_ = np.asarray(chf_out)

    def _baseline_chf_at(self, t: float) -> float:
        assert self.baseline_times_ is not None and self.baseline_cumulative_hazard_ is not None
        if len(self.baseline_times_) == 0:
            return 0.0
        idx = int(np.searchsorted(self.baseline_times_, t, side="right") - 1)
        if idx < 0:
            return 0.0
        return float(self.baseline_cumulative_hazard_[idx])

    def predict_survival_function(self, X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray:
        if self.baseline_times_ is None:
            raise RuntimeError("Call fit() before predict_survival_function().")
        eta = self.predict_risk(X)
        if times is None:
            times = self.baseline_times_
        times = np.asarray(times, dtype=np.float64).ravel()
        h0 = np.array([self._baseline_chf_at(float(t)) for t in times])
        # S(t|x) = exp(-H0(t) * exp(eta))
        return np.exp(-np.exp(eta)[:, None] * h0[None, :])

    def fit(
        self,
        X_train: np.ndarray,
        y_event_train: np.ndarray,
        y_time_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_event_val: Optional[np.ndarray] = None,
        y_time_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> DeepSurvModel:
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_event_train_t = torch.FloatTensor(y_event_train).to(self.device)
        y_time_train_t = torch.FloatTensor(y_time_train).to(self.device)
        has_val = X_val is not None and y_event_val is not None and y_time_val is not None
        if has_val:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_event_val_t = torch.FloatTensor(y_event_val).to(self.device)
            y_time_val_t = torch.FloatTensor(y_time_val).to(self.device)

        best_state: Dict[str, torch.Tensor] | None = None
        best_c = -1.0
        patience_left = self.early_stopping_patience

        self.model.train()
        n_batches = max(1, (len(X_train) + self.batch_size - 1) // self.batch_size)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            indices = torch.randperm(len(X_train), device=self.device)
            for i in range(0, len(X_train), self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                xb = X_train_t[batch_idx]
                eb = y_event_train_t[batch_idx]
                tb = y_time_train_t[batch_idx]
                self.optimizer.zero_grad()
                risk = self.model(xb).squeeze()
                loss = self.negative_log_likelihood(risk, eb, tb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_risk = self.model(X_val_t).squeeze()
                    vloss = self.negative_log_likelihood(val_risk, y_event_val_t, y_time_val_t)
                    self.val_losses.append(float(vloss.item()))
                self.model.train()

            if has_val and self.early_stopping_patience > 0 and (epoch + 1) % self.early_stopping_eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_risk_np = self.model(X_val_t).squeeze().cpu().numpy()
                c_val = self._concordance_index_numpy(y_time_val, y_event_val, val_risk_np)
                self.model.train()
                if c_val > best_c + self.early_stopping_min_delta:
                    best_c = c_val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.early_stopping_patience
                else:
                    patience_left -= 1
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} | train_loss={avg_loss:.4f} | "
                        f"val_C={c_val:.4f} | patience={patience_left}"
                    )
                if patience_left <= 0:
                    if verbose:
                        print("Early stopping on validation concordance.")
                    break
            elif verbose and (epoch + 1) % max(10, self.early_stopping_eval_every) == 0:
                extra = ""
                if has_val:
                    extra = f" | val_loss={self.val_losses[-1]:.4f}"
                print(f"Epoch {epoch + 1}/{self.epochs} | train_loss={avg_loss:.4f}{extra}")

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self._fit_breslow_baseline(X_train, y_event_train, y_time_train)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            return self.model(xt).squeeze().cpu().numpy()

    def get_training_history(self) -> Dict[str, list]:
        return {"train_loss": self.train_losses, "val_loss": self.val_losses if self.val_losses else None}
