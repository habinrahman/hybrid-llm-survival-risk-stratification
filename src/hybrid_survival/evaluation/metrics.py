"""Survival metrics, plots, and formatted reporting."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

try:
    from numpy import trapezoid as trap_int  # NumPy 2.0+
except ImportError:
    trap_int = np.trapz  # type: ignore[misc, assignment]


def print_model_comparison_table(
    results_list: List[Dict[str, Any]],
    time_horizons: Optional[List[float]] = None,
) -> None:
    """
    Print a fixed-width research-style comparison block.
    Brier rows use union of horizons from `time_horizons` and keys in results.
    """
    width = 60
    print("-" * width)
    print("Model Comparison")
    print("-" * width)
    print(f"{'Model':<10}{'C-index':>10}{'N Samples':>12}{'N Events':>12}{'Event Rate':>12}")
    for r in results_list:
        name = str(r.get("model_name", ""))[:10]
        cidx = float(r.get("c_index", 0.0))
        n = int(r.get("n_samples", 0))
        nev = int(r.get("n_events", 0))
        rate = float(r.get("event_rate", 0.0))
        print(f"{name:<10}{cidx:>10.4f}{n:>12}{nev:>12}{rate:>11.2%}")
    print()
    print("Brier Scores:")
    horizons = list(time_horizons or [])
    for r in results_list:
        bs = r.get("brier_scores") or {}
        for k in bs:
            if k.startswith("t="):
                try:
                    tval = float(k.split("=", 1)[1])
                    if tval not in horizons:
                        horizons.append(tval)
                except ValueError:
                    pass
    horizons = sorted(set(horizons))
    if not horizons:
        print("(no Brier scores — models may lack survival function support)")
    else:
        for t in horizons:
            key = f"t={t:.1f}"
            parts: list[str] = []
            for r in results_list:
                bs = r.get("brier_scores") or {}
                if key in bs:
                    parts.append(f"{r.get('model_name', '?')}={float(bs[key]):.4f}")
            if parts:
                print(f"{key}: " + " | ".join(parts))
            else:
                print(f"{key}: " + " | ".join(f"{r.get('model_name', '?')}=n/a" for r in results_list))
    print("-" * width)


class SurvivalMetrics:
    @staticmethod
    def concordance_index(y_time: np.ndarray, risk_scores: np.ndarray, y_event: np.ndarray) -> float:
        return float(concordance_index(y_time, -risk_scores, y_event))

    @staticmethod
    def brier_score(
        y_time: np.ndarray,
        y_event: np.ndarray,
        survival_probs: np.ndarray,
        eval_time: float,
    ) -> float:
        observed = (y_time <= eval_time) & (y_event == 1)
        mask = (y_time > eval_time) | ((y_time <= eval_time) & (y_event == 1))
        y_time_m = y_time[mask]
        obs_m = observed[mask]
        s_m = survival_probs[mask]
        return float(np.mean((obs_m.astype(float) - (1.0 - s_m)) ** 2))

    @staticmethod
    def integrated_brier_score(
        y_time: np.ndarray,
        y_event: np.ndarray,
        survival_functions: np.ndarray,
        time_points: np.ndarray,
    ) -> float:
        brier_scores = []
        for i, t in enumerate(time_points):
            bs = SurvivalMetrics.brier_score(y_time, y_event, survival_functions[:, i], float(t))
            brier_scores.append(bs)
        if len(time_points) < 2:
            return float(brier_scores[0]) if brier_scores else 0.0
        return float(trap_int(brier_scores, time_points) / (time_points[-1] - time_points[0]))


class ModelEvaluator:
    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_time_test: np.ndarray,
        y_event_test: np.ndarray,
        model_name: str,
        eval_times: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        risk_scores = model.predict_risk(X_test)
        c_index = SurvivalMetrics.concordance_index(y_time_test, risk_scores, y_event_test)
        out: Dict[str, Any] = {
            "model_name": model_name,
            "c_index": c_index,
            "n_samples": len(X_test),
            "n_events": int(y_event_test.sum()),
            "event_rate": float(y_event_test.mean()),
        }
        if hasattr(model, "predict_survival_function") and eval_times is not None and len(eval_times) > 0:
            eval_times = np.asarray(eval_times, dtype=np.float64).ravel()
            surv = model.predict_survival_function(X_test, times=eval_times)
            brier_scores: Dict[str, float] = {}
            for i, t in enumerate(eval_times):
                bs = SurvivalMetrics.brier_score(y_time_test, y_event_test, surv[:, i], float(t))
                brier_scores[f"t={t:.1f}"] = float(bs)
            out["brier_scores"] = brier_scores
        self.results[model_name] = out
        return out

    def compare_models(self, results_list: List[Dict]) -> pd.DataFrame:
        rows = []
        for result in results_list:
            row = {
                "Model": result["model_name"],
                "C-index": result["c_index"],
                "N Samples": result["n_samples"],
                "N Events": result["n_events"],
                "Event Rate": f"{result['event_rate']:.2%}",
            }
            if "brier_scores" in result:
                for time_label, score in result["brier_scores"].items():
                    row[f"Brier {time_label}"] = score
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_calibration(
        self,
        model: Any,
        X_test: np.ndarray,
        y_time_test: np.ndarray,
        y_event_test: np.ndarray,
        eval_time: float,
        model_name: str,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        if not hasattr(model, "predict_survival_function"):
            print(f"{model_name}: no survival function; skip calibration.")
            return
        survival_probs = model.predict_survival_function(X_test, times=np.array([eval_time]))[:, 0]
        survived = (y_time_test > eval_time) | ((y_time_test <= eval_time) & (y_event_test == 0))
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.clip(np.digitize(survival_probs, bins) - 1, 0, n_bins - 1)
        pred_probs, obs_probs = [], []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                pred_probs.append(float(survival_probs[mask].mean()))
                obs_probs.append(float(survived[mask].mean()))
        plt.figure(figsize=(8, 6))
        plt.plot(pred_probs, obs_probs, "o-", label=model_name, linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.xlabel("Predicted survival probability")
        plt.ylabel("Observed survival probability")
        plt.title(f"Calibration at t={eval_time:.1f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_survival_curves(
        self,
        model: Any,
        X_test: np.ndarray,
        time_range: np.ndarray,
        n_samples: int = 10,
        model_name: str = "",
        save_path: Optional[str] = None,
    ) -> None:
        if not hasattr(model, "predict_survival_function"):
            print(f"{model_name}: no survival function; skip survival curves.")
            return
        idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        Xs = X_test[idx]
        surv = model.predict_survival_function(Xs, times=time_range)
        plt.figure(figsize=(10, 6))
        for i in range(len(idx)):
            plt.plot(time_range, surv[i, :], alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.title(f"Predicted survival curves {model_name}".strip())
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_training_history(
        self, model: Any, model_name: str = "", save_path: Optional[str] = None
    ) -> None:
        if not hasattr(model, "get_training_history"):
            return
        history = model.get_training_history()
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train", linewidth=2)
        if history.get("val_loss"):
            plt.plot(history["val_loss"], label="Val loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training history {model_name}".strip())
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()


def cross_validate_survival(
    model_class: Any,
    X: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
    n_folds: int = 5,
    **model_params: Any,
) -> Dict[str, Any]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores: list[float] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        model = model_class(**model_params)
        model.fit(X[train_idx], y_event[train_idx], y_time[train_idx])
        rs = model.predict_risk(X[test_idx])
        c = SurvivalMetrics.concordance_index(y_time[test_idx], rs, y_event[test_idx])
        scores.append(c)
        print(f"Fold {fold + 1}/{n_folds} C-index: {c:.4f}")
    return {
        "cv_scores": scores,
        "mean_c_index": float(np.mean(scores)),
        "std_c_index": float(np.std(scores)),
    }
