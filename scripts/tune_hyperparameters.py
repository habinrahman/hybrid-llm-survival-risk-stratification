#!/usr/bin/env python3
"""Optuna hyperparameter search (validation concordance on fused multimodal features)."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_survival.data.preprocessing import MIMICDataLoader, StructuredDataPreprocessor  # noqa: E402
from hybrid_survival.evaluation.metrics import SurvivalMetrics  # noqa: E402
from hybrid_survival.features.fusion import FeatureFusion, create_multimodal_dataset  # noqa: E402
from hybrid_survival.features.text_embeddings import ClinicalBERTEmbedder, extract_text_features  # noqa: E402
from hybrid_survival.models.survival import CoxModel, DeepSurvModel  # noqa: E402
from hybrid_survival.pipelines.hybrid_pipeline import feature_lists_from_config  # noqa: E402
from hybrid_survival.utils.repro import set_global_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--n-patients", type=int, default=4000, help="Synthetic cohort size for tuning speed")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_global_seed(int(cfg["training"]["random_state"]))

    data_cfg = cfg.get("data") or {}
    num_feats, cat_feats = feature_lists_from_config(cfg)
    loader = MIMICDataLoader(cfg["paths"]["data_dir"])
    df = loader.load_cohort(
        outcome=cfg["outcomes"]["target"],
        n_patients=args.n_patients,
        random_state=int(cfg["training"]["random_state"]),
        cohort_csv=data_cfg.get("cohort_csv"),
        text_column=str(data_cfg.get("text_column", "clinical_notes")),
        time_column=str(data_cfg.get("time_column", "time")),
        event_column=str(data_cfg.get("event_column", "event")),
        patient_id_column=data_cfg.get("patient_id_column", "subject_id"),
        csv_kwargs=data_cfg.get("csv_kwargs"),
    )
    idx = np.arange(len(df))
    train_val_idx, _test_idx = train_test_split(
        idx,
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["training"]["random_state"]),
        stratify=df["event"].values,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=float(cfg["training"]["val_size"]),
        random_state=int(cfg["training"]["random_state"]),
        stratify=df["event"].values[train_val_idx],
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    pre = StructuredDataPreprocessor(
        imputation_strategy=cfg["training"]["imputation_strategy"],
        normalization=cfg["training"]["normalization"],
    )
    pre.fit(train_df, num_feats, cat_feats)
    Xs_tr = pre.transform(train_df)
    Xs_va = pre.transform(val_df)

    mcfg = cfg["model"]
    embedder = ClinicalBERTEmbedder(
        model_name=mcfg["llm_model"],
        max_length=int(mcfg["max_sequence_length"]),
        batch_size=int(mcfg["embedding_batch_size"]),
    )
    Xt_tr = extract_text_features(
        train_df["clinical_notes"].tolist(),
        embedder=embedder,
        model_name=mcfg["llm_model"],
        max_length=int(mcfg["max_sequence_length"]),
        batch_size=int(mcfg["embedding_batch_size"]),
    )
    Xt_va = extract_text_features(
        val_df["clinical_notes"].tolist(),
        embedder=embedder,
        model_name=mcfg["llm_model"],
        max_length=int(mcfg["max_sequence_length"]),
        batch_size=int(mcfg["embedding_batch_size"]),
    )

    ds_tr = create_multimodal_dataset(train_df, Xs_tr, Xt_tr)
    ds_va = create_multimodal_dataset(val_df, Xs_va, Xt_va)

    def objective(trial: optuna.Trial) -> float:
        text_pca = trial.suggest_int("text_dim_reduction", 32, 128, step=16)
        fusion = FeatureFusion(
            fusion_method="concatenation",
            text_dim_reduction=text_pca,
            normalize_before_fusion=True,
        )
        X_tr = ds_tr.get_fused_features(fusion, fit=True)
        X_va = ds_va.get_fused_features(fusion, fit=False)

        penalizer = trial.suggest_float("cox_penalizer", 1e-4, 0.2, log=True)
        l1 = trial.suggest_float("cox_l1_ratio", 0.0, 1.0)
        cox = CoxModel(penalizer=penalizer, l1_ratio=l1)
        cox.fit(X_tr, ds_tr.y_event, ds_tr.y_time)
        c_cox = SurvivalMetrics.concordance_index(ds_va.y_time, cox.predict_risk(X_va), ds_va.y_event)

        hidden = trial.suggest_categorical(
            "deepsurv_hidden",
            [
                (128, 64, 32),
                (256, 128, 64),
                (256, 128, 64, 32),
                (128, 128, 64, 32),
            ],
        )
        hidden = list(hidden)
        dropout = trial.suggest_float("deepsurv_dropout", 0.1, 0.5)
        lr = trial.suggest_float("deepsurv_lr", 1e-4, 3e-3, log=True)
        bs = trial.suggest_categorical("deepsurv_batch", [32, 64, 128])
        epochs = trial.suggest_int("deepsurv_epochs", 40, 150)

        ds_model = DeepSurvModel(
            input_dim=X_tr.shape[1],
            hidden_layers=list(hidden),
            dropout=dropout,
            learning_rate=lr,
            batch_size=bs,
            epochs=epochs,
            early_stopping_patience=12,
            early_stopping_eval_every=2,
            early_stopping_min_delta=0.001,
        )
        ds_model.fit(
            X_tr,
            ds_tr.y_event,
            ds_tr.y_time,
            X_val=X_va,
            y_event_val=ds_va.y_event,
            y_time_val=ds_va.y_time,
            verbose=False,
        )
        c_ds = SurvivalMetrics.concordance_index(ds_va.y_time, ds_model.predict_risk(X_va), ds_va.y_event)
        return float((c_cox + c_ds) / 2.0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    print("Best trial:", study.best_trial.params)
    print(f"Best validation mean C-index (Cox+DeepSurv): {study.best_value:.4f}")

    best_cfg = copy.deepcopy(cfg)
    bp = study.best_trial.params
    best_cfg["fusion"]["text_dim_reduction"] = int(bp["text_dim_reduction"])
    best_cfg["survival"]["cox"]["penalizer"] = float(bp["cox_penalizer"])
    best_cfg["survival"]["cox"]["l1_ratio"] = float(bp["cox_l1_ratio"])
    best_cfg["survival"]["deepsurv"]["hidden_layers"] = list(bp["deepsurv_hidden"])
    best_cfg["survival"]["deepsurv"]["dropout"] = float(bp["deepsurv_dropout"])
    best_cfg["survival"]["deepsurv"]["learning_rate"] = float(bp["deepsurv_lr"])
    best_cfg["survival"]["deepsurv"]["batch_size"] = int(bp["deepsurv_batch"])
    best_cfg["survival"]["deepsurv"]["epochs"] = int(bp["deepsurv_epochs"])

    out_path = ROOT / "configs" / "config.best.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote tuned config to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
