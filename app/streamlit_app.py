"""Streamlit UI for hybrid multimodal survival inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_survival.features.text_embeddings import extract_text_features  # noqa: E402
from hybrid_survival.inference import (  # noqa: E402
    build_patient_frame,
    load_inference_bundle,
    predict_patient,
)
from hybrid_survival.pipelines.hybrid_pipeline import feature_lists_from_config  # noqa: E402


@st.cache_resource
def _bundle():
    return load_inference_bundle(models_dir=ROOT / "models", config_path=ROOT / "configs" / "config.yaml")


def main() -> None:
    st.set_page_config(page_title="Hybrid LLM–Survival", layout="wide")
    st.title("Hybrid LLM–Survival risk stratification")
    st.caption("Structured EHR + clinical note embeddings + survival models")

    try:
        cfg, pre, fusion, cox, ds = _bundle()
    except Exception as e:
        st.error(f"Could not load models. Train first (`python scripts/train.py`).\n\n{e}")
        return

    numeric_cols, cat_cols = feature_lists_from_config(cfg)
    horizons = cfg["evaluation"]["time_horizons"]
    c1, c2 = st.columns(2)
    row: dict = {}
    with c1:
        st.subheader("Structured features")
        for c in numeric_cols:
            row[c] = st.number_input(c, value=70.0 if c == "age" else 0.0, format="%.4f")
        for c in cat_cols:
            row[c] = st.text_input(c, value="UNKNOWN")
    with c2:
        st.subheader("Clinical note")
        note = st.text_area("clinical_notes", height=220, placeholder="Admission note...")

    if st.button("Predict", type="primary"):
        if not note.strip():
            st.warning("Please enter clinical notes.")
            return
        risk, surv = predict_patient(row, note, cfg, pre, fusion, ds, numeric_cols, cat_cols)
        st.markdown("---")
        st.metric("DeepSurv risk score (higher → higher hazard)", f"{risk:.4f}")
        cols = st.columns(len(horizons))
        for i, (t, s) in enumerate(zip(horizons, surv)):
            with cols[i]:
                st.metric(f"S(t={t:.1f})", f"{s:.4f}" if np.isfinite(s) else "n/a")
        tmax = float(max(horizons)) * 1.25 if horizons else 10.0
        tgrid = np.linspace(0.01, max(tmax, 1.0), 80)
        df1 = build_patient_frame(row, numeric_cols, cat_cols)
        Xs1 = pre.transform(df1)
        Xt1 = extract_text_features(
            [note],
            model_name=cfg["model"]["llm_model"],
            max_length=int(cfg["model"]["max_sequence_length"]),
            batch_size=int(cfg["model"]["embedding_batch_size"]),
            preprocess=True,
        )
        Xf = fusion.transform(Xs1, Xt1)
        s_curve = ds.predict_survival_function(Xf, times=tgrid)[0]
        st.subheader("DeepSurv survival curve")
        st.line_chart(pd.DataFrame({"S(t)": s_curve}, index=tgrid))
        risk_c, _ = predict_patient(row, note, cfg, pre, fusion, cox, numeric_cols, cat_cols)
        with st.expander("Cox model (reference)"):
            st.metric("Cox risk (partial hazard scale)", f"{risk_c:.4f}")


if __name__ == "__main__":
    main()
