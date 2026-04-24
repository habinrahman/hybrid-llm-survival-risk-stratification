"""End-to-end multimodal survival pipeline (no train-set leakage)."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from hybrid_survival.data.preprocessing import MIMICDataLoader, StructuredDataPreprocessor
from hybrid_survival.evaluation.metrics import ModelEvaluator, print_model_comparison_table
from hybrid_survival.features.fusion import FeatureFusion, create_multimodal_dataset
from hybrid_survival.features.text_embeddings import ClinicalBERTEmbedder, extract_text_features
from hybrid_survival.models.survival import CoxModel, DeepSurvModel
from hybrid_survival.utils.advanced_features import (
    categorize_risk,
    confidence_interval,
    confidence_interval_mean,
    generate_pdf_report,
    generate_shap_explanation,
    normalize_risk_for_stratification,
    plot_kaplan_meier,
    plot_model_comparison,
    setup_logging,
)
from hybrid_survival.utils.repro import set_global_seed

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore


NUMERIC_FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "spo2",
    "glucose",
    "creatinine",
    "hemoglobin",
    "wbc",
]
CATEGORICAL_FEATURES = ["gender", "ethnicity", "insurance", "admission_type"]


def feature_lists_from_config(config: Dict[str, Any]) -> tuple[list[str], list[str]]:
    """Tabular columns for preprocessing; defaults match the synthetic MIMIC-style demo."""
    data_cfg = config.get("data") or {}
    num = data_cfg.get("numeric_features")
    cat = data_cfg.get("categorical_features")
    return (
        list(num) if num is not None else list(NUMERIC_FEATURES),
        list(cat) if cat is not None else list(CATEGORICAL_FEATURES),
    )


def _default_config_path() -> str:
    root = Path(__file__).resolve().parents[3]
    p = root / "configs" / "config.yaml"
    if p.exists():
        return str(p)
    return "configs/config.yaml"


class HybridSurvivalPipeline:
    def __init__(self, config_path: Optional[str] = None):
        cfg_path = config_path or _default_config_path()
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        self.config_path = cfg_path
        self.numeric_features, self.categorical_features = feature_lists_from_config(self.config)
        self._create_directories()
        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.struct_preprocessor: Optional[StructuredDataPreprocessor] = None
        self.fusion_module: Optional[FeatureFusion] = None
        self.embedder: Optional[ClinicalBERTEmbedder] = None
        self.models: Dict[str, Any] = {}
        self.evaluator = ModelEvaluator()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.X_train = self.X_val = self.X_test = None
        self._last_eval_results: list[dict[str, Any]] = []
        self.logger: Optional[Any] = None

    def _create_directories(self) -> None:
        for path in self.config.get("paths", {}).values():
            os.makedirs(path, exist_ok=True)

    def _advanced_config(self) -> dict[str, Any]:
        return self.config.get("advanced") or {}

    def _init_logging(self) -> None:
        adv = self._advanced_config()
        if not adv.get("enabled", True):
            self.logger = logging.getLogger("hybrid_survival.noop")
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
            return
        logs_dir = self.config.get("paths", {}).get("logs_dir", "logs")
        log_name = adv.get("log_file", "project.log")
        self.logger = setup_logging(logs_dir=logs_dir, filename=log_name)
        self.logger.info("Hybrid survival pipeline logger initialized.")

    def load_data(self, outcome: Optional[str] = None) -> None:
        print("\n[1] Loading data")
        print("-" * 60)
        if outcome is None:
            outcome = self.config["outcomes"]["target"]
        rs = int(self.config["training"]["random_state"])
        set_global_seed(rs)
        data_cfg = self.config.get("data") or {}
        loader = MIMICDataLoader(self.config["paths"]["data_dir"])
        cohort_csv = data_cfg.get("cohort_csv")
        self.df = loader.load_cohort(
            outcome=outcome,
            random_state=rs,
            cohort_csv=cohort_csv,
            text_column=str(data_cfg.get("text_column", "clinical_notes")),
            time_column=str(data_cfg.get("time_column", "time")),
            event_column=str(data_cfg.get("event_column", "event")),
            patient_id_column=data_cfg.get("patient_id_column", "subject_id"),
            csv_kwargs=data_cfg.get("csv_kwargs"),
        )
        for col in self.numeric_features + self.categorical_features:
            if col not in self.df.columns:
                raise ValueError(
                    f"Cohort is missing column {col!r}. Present columns: {sorted(self.df.columns.tolist())}. "
                    "Set data.numeric_features / data.categorical_features in config to match your file."
                )
        print(f"Shape: {self.df.shape} | event rate: {self.df['event'].mean():.2%}")

    def split_raw_data(self) -> None:
        print("\n[2] Train / validation / test split (before preprocessing)")
        print("-" * 60)
        assert self.df is not None
        rs = int(self.config["training"]["random_state"])
        idx = np.arange(len(self.df))
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=float(self.config["training"]["test_size"]),
            random_state=rs,
            stratify=self.df["event"].values,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=float(self.config["training"]["val_size"]),
            random_state=rs,
            stratify=self.df["event"].values[train_val_idx],
        )
        self.train_df = self.df.iloc[train_idx].reset_index(drop=True)
        self.val_df = self.df.iloc[val_idx].reset_index(drop=True)
        self.test_df = self.df.iloc[test_idx].reset_index(drop=True)
        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")

    def preprocess_structured_data(self) -> None:
        print("\n[3] Structured preprocessing (fit on train only)")
        print("-" * 60)
        assert self.train_df is not None and self.val_df is not None and self.test_df is not None
        self.struct_preprocessor = StructuredDataPreprocessor(
            imputation_strategy=self.config["training"]["imputation_strategy"],
            normalization=self.config["training"]["normalization"],
        )
        self.struct_preprocessor.fit(self.train_df, self.numeric_features, self.categorical_features)
        X_tr = self.struct_preprocessor.transform(self.train_df)
        X_va = self.struct_preprocessor.transform(self.val_df)
        X_te = self.struct_preprocessor.transform(self.test_df)
        self._X_struct_train = X_tr
        self._X_struct_val = X_va
        self._X_struct_test = X_te
        print(f"Structured features: dim={X_tr.shape[1]}")

    def generate_text_embeddings(self) -> None:
        print("\n[4] Text embeddings (encoder not fine-tuned; no leakage from val/test labels)")
        print("-" * 60)
        assert self.train_df is not None
        mcfg = self.config["model"]
        self.embedder = ClinicalBERTEmbedder(
            model_name=mcfg["llm_model"],
            max_length=int(mcfg["max_sequence_length"]),
            batch_size=int(mcfg["embedding_batch_size"]),
        )
        notes_tr = self.train_df["clinical_notes"].tolist()
        notes_va = self.val_df["clinical_notes"].tolist()
        notes_te = self.test_df["clinical_notes"].tolist()
        self._X_text_train = extract_text_features(
            notes_tr,
            model_name=mcfg["llm_model"],
            max_length=int(mcfg["max_sequence_length"]),
            batch_size=int(mcfg["embedding_batch_size"]),
            preprocess=True,
            embedder=self.embedder,
        )
        self._X_text_val = extract_text_features(
            notes_va,
            model_name=mcfg["llm_model"],
            max_length=int(mcfg["max_sequence_length"]),
            batch_size=int(mcfg["embedding_batch_size"]),
            preprocess=True,
            embedder=self.embedder,
        )
        self._X_text_test = extract_text_features(
            notes_te,
            model_name=mcfg["llm_model"],
            max_length=int(mcfg["max_sequence_length"]),
            batch_size=int(mcfg["embedding_batch_size"]),
            preprocess=True,
            embedder=self.embedder,
        )
        print(f"Text embeddings: dim={self._X_text_train.shape[1]}")

    def fuse_features(self) -> None:
        print("\n[5] Fusion (fit on train only)")
        print("-" * 60)
        fcfg = self.config["fusion"]
        self.fusion_module = FeatureFusion(
            fusion_method=fcfg.get("fusion_method", "concatenation"),
            text_dim_reduction=int(fcfg["text_dim_reduction"]),
            normalize_before_fusion=bool(fcfg.get("normalize_before_fusion", True)),
        )
        self.train_dataset = create_multimodal_dataset(
            self.train_df, self._X_struct_train, self._X_text_train
        )
        self.val_dataset = create_multimodal_dataset(self.val_df, self._X_struct_val, self._X_text_val)
        self.test_dataset = create_multimodal_dataset(self.test_df, self._X_struct_test, self._X_text_test)
        self.X_train = self.train_dataset.get_fused_features(self.fusion_module, fit=True)
        self.X_val = self.val_dataset.get_fused_features(self.fusion_module, fit=False)
        self.X_test = self.test_dataset.get_fused_features(self.fusion_module, fit=False)
        print(f"Fused dim: {self.X_train.shape[1]} | {self.train_dataset}")

    def train_models(self) -> None:
        print("\n[6] Train survival models")
        print("-" * 60)
        assert self.X_train is not None
        ccfg = self.config["survival"]["cox"]
        self.models["cox"] = CoxModel(penalizer=float(ccfg["penalizer"]), l1_ratio=float(ccfg["l1_ratio"]))
        self.models["cox"].fit(self.X_train, self.train_dataset.y_event, self.train_dataset.y_time)

        dcfg = self.config["survival"]["deepsurv"]
        self.models["deepsurv"] = DeepSurvModel(
            input_dim=self.X_train.shape[1],
            hidden_layers=list(dcfg["hidden_layers"]),
            dropout=float(dcfg["dropout"]),
            learning_rate=float(dcfg["learning_rate"]),
            batch_size=int(dcfg["batch_size"]),
            epochs=int(dcfg["epochs"]),
            early_stopping_patience=int(dcfg.get("early_stopping_patience", 0)),
            early_stopping_eval_every=int(dcfg.get("early_stopping_eval_every", 2)),
            early_stopping_min_delta=float(dcfg.get("early_stopping_min_delta", 0.001)),
        )
        self.models["deepsurv"].fit(
            self.X_train,
            self.train_dataset.y_event,
            self.train_dataset.y_time,
            X_val=self.X_val,
            y_event_val=self.val_dataset.y_event,
            y_time_val=self.val_dataset.y_time,
            verbose=True,
        )

    def evaluate_models(self) -> pd.DataFrame:
        print("\n[7] Evaluation")
        print("-" * 60)
        horizons = np.asarray(self.config["evaluation"]["time_horizons"], dtype=np.float64)
        rows = []
        labels = {"cox": "Cox", "deepsurv": "DeepSurv"}
        for name, model in self.models.items():
            res = self.evaluator.evaluate_model(
                model,
                self.X_test,
                self.test_dataset.y_time,
                self.test_dataset.y_event,
                labels.get(name, name),
                eval_times=horizons,
            )
            rows.append(res)
            print(f"{name} C-index (test): {res['c_index']:.4f}")
        print_model_comparison_table(rows, time_horizons=list(horizons))
        comparison_df = self.evaluator.compare_models(rows)
        comparison_df.to_csv(
            os.path.join(self.config["paths"]["results_dir"], "model_comparison.csv"),
            index=False,
        )
        self._last_eval_results = rows
        return comparison_df

    def _run_advanced_outputs(self, comparison_df: pd.DataFrame) -> None:
        """KM curve, C-index bar chart, SHAP (optional), PDF example, console risk summary."""
        adv = self._advanced_config()
        if not adv.get("enabled", True):
            return
        rd = self.config["paths"]["results_dir"]
        assert self.test_dataset is not None and self.X_test is not None

        # Kaplan–Meier on observed test outcomes
        plot_kaplan_meier(
            self.test_dataset.y_time,
            self.test_dataset.y_event,
            os.path.join(rd, "kaplan_meier_curve.png"),
        )
        if self.logger:
            self.logger.info("Saved Kaplan–Meier curve to results/kaplan_meier_curve.png")

        # Bar chart from latest evaluation table
        names = comparison_df["Model"].astype(str).tolist()
        cidx = [float(x) for x in comparison_df["C-index"].tolist()]
        plot_model_comparison(names, cidx, os.path.join(rd, "model_comparison.png"))
        if self.logger:
            self.logger.info("Saved model comparison chart to results/model_comparison.png")

        # SHAP on DeepSurv (primary nonlinear model)
        if adv.get("run_shap", True) and "deepsurv" in self.models:
            n_bg = int(adv.get("shap_max_background", 300))
            n_ev = int(adv.get("shap_max_eval", 150))
            fnames = [f"f{i}" for i in range(self.X_train.shape[1])]
            path = generate_shap_explanation(
                self.models["deepsurv"],
                self.X_train,
                self.X_test,
                os.path.join(rd, "shap_summary_deepsurv.png"),
                max_background=n_bg,
                max_eval=n_ev,
                feature_names=fnames,
            )
            if path and self.logger:
                self.logger.info("Saved SHAP summary to %s", path)

        # Example: first test patient — risk tier + intervals + PDF
        ds = self.models.get("deepsurv")
        if ds is not None:
            risk = float(ds.predict_risk(self.X_test[:1])[0])
            r_train = ds.predict_risk(self.X_train)
            rn = float(np.clip(normalize_risk_for_stratification(risk, r_train), 0.0, 1.0))
            category = categorize_risk(rn)
            r_test = ds.predict_risk(self.X_test)
            lo_spread, hi_spread = confidence_interval(r_test)
            lo_mean, hi_mean = confidence_interval_mean(r_test)

            print("\n" + "-" * 60)
            print("Risk stratification (example: first test patient, DeepSurv)")
            print("-" * 60)
            print(f"Risk Score: {risk:.4f}")
            print(f"Risk Category (0–1 normalized vs train min/max): {category}")
            print(f"95% interval (mean ± 1.96·std of test predicted risks): ({lo_spread:.4f}, {hi_spread:.4f})")
            print(f"95% CI for mean test risk (mean ± 1.96·SE): ({lo_mean:.4f}, {hi_mean:.4f})")
            lo_pt, hi_pt = confidence_interval([risk])
            print(f"95% interval (single-score formula): ({lo_pt:.4f}, {hi_pt:.4f})")
            print("-" * 60)

            extra = [
                f"Normalized risk (train scale): {rn:.4f}",
                f"Mean test risk CI: ({lo_mean:.4f}, {hi_mean:.4f})",
            ]
            try:
                generate_pdf_report(
                    risk_score=risk,
                    risk_category=category,
                    output_path=os.path.join(rd, "prediction_report.pdf"),
                    extra_lines=extra,
                )
                if self.logger:
                    self.logger.info("Wrote results/prediction_report.pdf")
            except Exception as e:
                if self.logger:
                    self.logger.warning("PDF report skipped: %s", e)

    def visualize_results(self) -> None:
        print("\n[8] Visualizations")
        print("-" * 60)
        rd = self.config["paths"]["results_dir"]
        if "deepsurv" in self.models:
            self.evaluator.plot_training_history(
                self.models["deepsurv"],
                model_name="DeepSurv",
                save_path=os.path.join(rd, "deepsurv_training.png"),
            )
        med_t = float(np.median(self.test_dataset.y_time[self.test_dataset.y_event == 1]))
        tr = np.linspace(0, float(self.test_dataset.y_time.max()), 100)
        for model_name, model in self.models.items():
            self.evaluator.plot_calibration(
                model,
                self.X_test,
                self.test_dataset.y_time,
                self.test_dataset.y_event,
                med_t,
                model_name.upper(),
                save_path=os.path.join(rd, f"{model_name}_calibration.png"),
            )
            self.evaluator.plot_survival_curves(
                model,
                self.X_test,
                tr,
                n_samples=10,
                model_name=model_name.upper(),
                save_path=os.path.join(rd, f"{model_name}_survival_curves.png"),
            )

    def save_models(self) -> None:
        print("\n[9] Save artifacts")
        print("-" * 60)
        md = self.config["paths"]["models_dir"]
        for model_name, model in self.models.items():
            path = os.path.join(md, f"{model_name}_model.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved {path}")
        if joblib is not None:
            joblib.dump(self.struct_preprocessor, os.path.join(md, "struct_preprocessor.pkl"))
            joblib.dump(self.fusion_module, os.path.join(md, "fusion_module.pkl"))
            if self.fusion_module is not None and self.fusion_module.text_pca is not None:
                joblib.dump(self.fusion_module.text_pca, os.path.join(md, "text_pca.pkl"))
        else:
            with open(os.path.join(md, "struct_preprocessor.pkl"), "wb") as f:
                pickle.dump(self.struct_preprocessor, f)
            with open(os.path.join(md, "fusion_module.pkl"), "wb") as f:
                pickle.dump(self.fusion_module, f)
            if self.fusion_module is not None and self.fusion_module.text_pca is not None:
                with open(os.path.join(md, "text_pca.pkl"), "wb") as f:
                    pickle.dump(self.fusion_module.text_pca, f)
        with open(os.path.join(md, "embedder_config.pkl"), "wb") as f:
            pickle.dump(
                {
                    "llm_model": self.config["model"]["llm_model"],
                    "max_sequence_length": self.config["model"]["max_sequence_length"],
                    "embedding_batch_size": self.config["model"]["embedding_batch_size"],
                },
                f,
            )
        if "deepsurv" in self.models and self.X_train is not None:
            rtr = self.models["deepsurv"].predict_risk(self.X_train)
            with open(os.path.join(md, "train_risk_stats.pkl"), "wb") as f:
                pickle.dump(
                    {"deepsurv_min": float(np.min(rtr)), "deepsurv_max": float(np.max(rtr))},
                    f,
                )
        print("Saved preprocessors, fusion, optional text PCA, embedder config.")

    def run_full_pipeline(self) -> pd.DataFrame:
        set_global_seed(int(self.config["training"]["random_state"]))
        self._init_logging()
        if self.logger:
            self.logger.info("Pipeline started.")
        print("=" * 60)
        print("Hybrid LLM–Survival pipeline")
        print("=" * 60)
        self.load_data()
        self.split_raw_data()
        self.preprocess_structured_data()
        self.generate_text_embeddings()
        self.fuse_features()
        self.train_models()
        df = self.evaluate_models()
        self._run_advanced_outputs(df)
        self.visualize_results()
        self.save_models()
        if self.logger:
            self.logger.info("Pipeline completed successfully.")
        print("\n" + "=" * 60)
        print("Done.")
        print("=" * 60)
        return df
