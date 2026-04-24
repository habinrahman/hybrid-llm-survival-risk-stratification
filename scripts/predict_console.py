#!/usr/bin/env python3
"""Interactive console inference for one patient."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_survival.inference import load_inference_bundle, predict_patient  # noqa: E402
from hybrid_survival.pipelines.hybrid_pipeline import feature_lists_from_config  # noqa: E402


def _prompt_float(label: str) -> float:
    return float(input(f"{label}: ").strip())


def _prompt_str(label: str) -> str:
    return input(f"{label}: ").strip()


def main() -> int:
    cfg, pre, fusion, cox, ds = load_inference_bundle()
    numeric_cols, cat_cols = feature_lists_from_config(cfg)
    horizons = cfg["evaluation"]["time_horizons"]
    print("Enter patient fields (leave numeric blank for missing).")
    row: dict = {}
    for c in numeric_cols:
        s = input(f"{c}: ").strip()
        row[c] = float(s) if s else float("nan")
    for c in cat_cols:
        row[c] = input(f"{c}: ").strip() or "UNKNOWN"
    note = input("clinical_notes: ").strip()

    w = 60
    risk, surv = predict_patient(row, note, cfg, pre, fusion, ds, numeric_cols, cat_cols)
    print("-" * w)
    print("Patient Risk Prediction")
    print("-" * w)
    print(f"Risk Score: {risk:.4f}")
    for t, s in zip(horizons, surv):
        if np.isfinite(s):
            print(f"Survival Probability (t={t:.1f}): {s:.4f}")
        else:
            print(f"Survival Probability (t={t:.1f}): n/a")
    print("-" * w)
    risk_c, _ = predict_patient(row, note, cfg, pre, fusion, cox, numeric_cols, cat_cols)
    print(f"(Cox reference risk score: {risk_c:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
