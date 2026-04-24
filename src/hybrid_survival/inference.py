"""Load saved artifacts and score a single patient (structured + note)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from hybrid_survival.features.fusion import FeatureFusion
from hybrid_survival.features.text_embeddings import extract_text_features

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_inference_bundle(
    models_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Tuple[dict, Any, Any, Any, Any]:
    """Return (config, struct_preprocessor, fusion, cox_model, deepsurv_model)."""
    root = project_root()
    cfg_path = Path(config_path or root / "configs" / "config.yaml")
    md = Path(models_dir or load_yaml(cfg_path)["paths"]["models_dir"])
    cfg = load_yaml(cfg_path)
    if joblib is not None:
        pre = joblib.load(md / "struct_preprocessor.pkl")
        fusion = joblib.load(md / "fusion_module.pkl")
    else:
        with open(md / "struct_preprocessor.pkl", "rb") as f:
            pre = pickle.load(f)
        with open(md / "fusion_module.pkl", "rb") as f:
            fusion = pickle.load(f)
    with open(md / "cox_model.pkl", "rb") as f:
        cox = pickle.load(f)
    with open(md / "deepsurv_model.pkl", "rb") as f:
        ds = pickle.load(f)
    return cfg, pre, fusion, cox, ds


def build_patient_frame(row: Dict[str, Any], numeric_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    data = {**row}
    for c in numeric_cols + cat_cols:
        if c not in data:
            data[c] = np.nan
    return pd.DataFrame([data])


def predict_patient(
    row: Dict[str, Any],
    clinical_note: str,
    cfg: dict,
    struct_preprocessor: Any,
    fusion: FeatureFusion,
    model: Any,
    numeric_cols: List[str],
    cat_cols: List[str],
) -> Tuple[float, np.ndarray]:
    """Return (risk_score, survival_probs_at_horizons)."""
    df = build_patient_frame(row, numeric_cols, cat_cols)
    Xs = struct_preprocessor.transform(df)
    m = cfg["model"]
    Xt = extract_text_features(
        [clinical_note],
        model_name=m["llm_model"],
        max_length=int(m["max_sequence_length"]),
        batch_size=int(m["embedding_batch_size"]),
        preprocess=True,
    )
    X = fusion.transform(Xs, Xt)
    risk = float(model.predict_risk(X)[0])
    times = np.asarray(cfg["evaluation"]["time_horizons"], dtype=np.float64)
    if hasattr(model, "predict_survival_function"):
        surv = model.predict_survival_function(X, times=times)[0]
    else:
        surv = np.full(len(times), np.nan)
    return risk, surv
