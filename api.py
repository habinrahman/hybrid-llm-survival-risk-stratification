"""
FastAPI service for hybrid multimodal survival inference.

Run from project root::

    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_survival.inference import load_inference_bundle, predict_patient  # noqa: E402
from hybrid_survival.pipelines.hybrid_pipeline import feature_lists_from_config  # noqa: E402
from hybrid_survival.utils.advanced_features import categorize_risk, confidence_interval  # noqa: E402

try:
    from fastapi import FastAPI
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install fastapi and uvicorn: pip install fastapi uvicorn") from e


@lru_cache(maxsize=1)
def _bundle():
    return load_inference_bundle(models_dir=ROOT / "models", config_path=ROOT / "configs" / "config.yaml")


class PatientData(BaseModel):
    """Structured fields for the default demo schema; unknown keys are allowed for custom ``data.*`` configs."""

    model_config = ConfigDict(extra="allow")

    age: float = Field(default=65.0)
    heart_rate: float = Field(default=80.0)
    systolic_bp: float = Field(default=120.0)
    diastolic_bp: float = Field(default=80.0)
    temperature: float = Field(default=37.0)
    respiratory_rate: float = Field(default=18.0)
    spo2: float = Field(default=97.0)
    glucose: float = Field(default=110.0)
    creatinine: float = Field(default=1.0)
    hemoglobin: float = Field(default=13.0)
    wbc: float = Field(default=8.0)
    gender: str = Field(default="M")
    ethnicity: str = Field(default="WHITE")
    insurance: str = Field(default="Private")
    admission_type: str = Field(default="EMERGENCY")
    clinical_notes: str = Field(default="")


def _row_from_payload(data: PatientData, numeric_cols: list[str], cat_cols: list[str]) -> Dict[str, Any]:
    d = data.model_dump()
    row: Dict[str, Any] = {}
    for c in numeric_cols:
        v = d.get(c)
        if v is None or v == "":
            row[c] = float("nan")
        else:
            row[c] = float(v)
    for c in cat_cols:
        v = d.get(c)
        row[c] = str(v) if v is not None and str(v) != "" else "UNKNOWN"
    return row


app = FastAPI(title="Hybrid LLM–Survival Model API", version="0.2.0")


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Hybrid LLM–Survival Model API is running"}


@app.post("/predict")
def predict(data: PatientData) -> dict[str, Any]:
    cfg, pre, fusion, cox, ds = _bundle()
    num_cols, cat_cols = feature_lists_from_config(cfg)
    row = _row_from_payload(data, num_cols, cat_cols)
    note = data.clinical_notes or "No clinical note provided."
    risk, surv = predict_patient(row, note, cfg, pre, fusion, ds, num_cols, cat_cols)
    risk_cox, _ = predict_patient(row, note, cfg, pre, fusion, cox, num_cols, cat_cols)

    stats_path = ROOT / "models" / "train_risk_stats.pkl"
    try:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        r_lo, r_hi = float(stats["deepsurv_min"]), float(stats["deepsurv_max"])
    except (OSError, KeyError, ValueError):
        r_lo, r_hi = float(risk * 0.5), float(max(risk * 1.5, risk + 1e-6))
    rn = float(np.clip((risk - r_lo) / (r_hi - r_lo + 1e-12), 0.0, 1.0))
    category = categorize_risk(rn)
    lo, hi = confidence_interval([risk])

    horizons = list(cfg.get("evaluation", {}).get("time_horizons", []))
    surv_list = [float(s) if np.isfinite(s) else None for s in surv]

    return {
        "risk_score": float(risk),
        "risk_category": category,
        "normalized_risk_0_1": rn,
        "cox_risk_score": float(risk_cox),
        "survival_probabilities": {str(t): s for t, s in zip(horizons, surv_list)},
        "confidence_interval_single_score_formula": {"lower": float(lo), "upper": float(hi)},
    }
