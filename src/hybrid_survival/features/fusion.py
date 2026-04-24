"""Multimodal fusion: structured tabular + text embeddings."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class FeatureFusion:
    def __init__(
        self,
        fusion_method: str = "concatenation",
        text_dim_reduction: Optional[int] = None,
        normalize_before_fusion: bool = True,
    ):
        self.fusion_method = fusion_method
        self.text_dim_reduction = text_dim_reduction
        self.normalize_before_fusion = normalize_before_fusion
        self.text_pca: PCA | None = None
        self.struct_scaler: StandardScaler | None = None
        self.text_scaler: StandardScaler | None = None
        self.struct_weight = 0.5
        self.text_weight = 0.5

    def fit(self, X_struct: np.ndarray, X_text: np.ndarray) -> FeatureFusion:
        if self.text_dim_reduction is not None:
            n_components = int(min(self.text_dim_reduction, X_text.shape[1], X_text.shape[0]))
            self.text_pca = PCA(n_components=n_components, random_state=42)
            self.text_pca.fit(X_text)
        if self.normalize_before_fusion:
            self.struct_scaler = StandardScaler()
            self.struct_scaler.fit(X_struct)
            if self.text_pca is not None:
                xt = self.text_pca.transform(X_text)
                self.text_scaler = StandardScaler()
                self.text_scaler.fit(xt)
            else:
                self.text_scaler = StandardScaler()
                self.text_scaler.fit(X_text)
        return self

    def transform(self, X_struct: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        if self.text_pca is not None:
            X_text = self.text_pca.transform(X_text)
        if self.normalize_before_fusion:
            assert self.struct_scaler is not None and self.text_scaler is not None
            X_struct = self.struct_scaler.transform(X_struct)
            X_text = self.text_scaler.transform(X_text)
        if self.fusion_method == "concatenation":
            return np.hstack([X_struct, X_text])
        if self.fusion_method == "weighted":
            if X_struct.shape[1] != X_text.shape[1]:
                raise ValueError("Weighted fusion requires equal dimensions.")
            return self.struct_weight * X_struct + self.text_weight * X_text
        raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def fit_transform(self, X_struct: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        self.fit(X_struct, X_text)
        return self.transform(X_struct, X_text)


class MultimodalDataset:
    def __init__(
        self,
        X_struct: np.ndarray,
        X_text: np.ndarray,
        y_event: np.ndarray,
        y_time: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
    ):
        self.X_struct = X_struct
        self.X_text = X_text
        self.y_event = np.asarray(y_event)
        self.y_time = np.asarray(y_time, dtype=np.float64)
        self.patient_ids = patient_ids if patient_ids is not None else np.arange(len(y_event))
        assert len(X_struct) == len(X_text) == len(self.y_event) == len(self.y_time)
        self.n_samples = len(self.y_event)

    def get_fused_features(self, fusion_module: FeatureFusion, fit: bool = False) -> np.ndarray:
        if fit:
            return fusion_module.fit_transform(self.X_struct, self.X_text)
        return fusion_module.transform(self.X_struct, self.X_text)

    def split(self, indices: np.ndarray) -> MultimodalDataset:
        return MultimodalDataset(
            X_struct=self.X_struct[indices],
            X_text=self.X_text[indices],
            y_event=self.y_event[indices],
            y_time=self.y_time[indices],
            patient_ids=self.patient_ids[indices],
        )

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"MultimodalDataset(n={self.n_samples}, struct={self.X_struct.shape[1]}, "
            f"text={self.X_text.shape[1]}, event_rate={self.y_event.mean():.2%})"
        )


def create_multimodal_dataset(
    df: pd.DataFrame,
    X_struct: np.ndarray,
    X_text: np.ndarray,
    event_col: str = "event",
    time_col: str = "time",
    patient_id_col: str = "subject_id",
) -> MultimodalDataset:
    pids = df[patient_id_col].values if patient_id_col in df.columns else None
    return MultimodalDataset(
        X_struct=X_struct,
        X_text=X_text,
        y_event=df[event_col].values,
        y_time=df[time_col].values,
        patient_ids=pids,
    )
