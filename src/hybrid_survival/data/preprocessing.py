"""Structured EHR preprocessing and cohort loading."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


class StructuredDataPreprocessor:
    """Imputation, optional standardization, and label encoding for tabular EHR."""

    def __init__(self, imputation_strategy: str = "median", normalization: str = "standardize"):
        self.imputation_strategy = imputation_strategy
        self.normalization = normalization
        self.numeric_imputer = SimpleImputer(strategy=imputation_strategy)
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler: StandardScaler | None = StandardScaler() if normalization == "standardize" else None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> StructuredDataPreprocessor:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        if numeric_features:
            X_numeric = X[numeric_features].values
            self.numeric_imputer.fit(X_numeric)
            X_numeric_imputed = self.numeric_imputer.transform(X_numeric)
            if self.scaler is not None:
                self.scaler.fit(X_numeric_imputed)
        if categorical_features:
            X_categorical = X[categorical_features].values
            self.categorical_imputer.fit(X_categorical)
            X_categorical_imputed = self.categorical_imputer.transform(X_categorical)
            for i, feature in enumerate(categorical_features):
                encoder = LabelEncoder()
                encoder.fit(X_categorical_imputed[:, i])
                self.label_encoders[feature] = encoder
        self.feature_names = numeric_features + categorical_features
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        processed_parts: list[np.ndarray] = []
        if self.numeric_features:
            X_numeric = X[self.numeric_features].values
            X_numeric_imputed = self.numeric_imputer.transform(X_numeric)
            if self.scaler is not None:
                X_numeric_processed = self.scaler.transform(X_numeric_imputed)
            else:
                X_numeric_processed = X_numeric_imputed
            processed_parts.append(X_numeric_processed)
        if self.categorical_features:
            X_categorical = X[self.categorical_features].values
            X_categorical_imputed = self.categorical_imputer.transform(X_categorical)
            encoded_features = []
            for i, feature in enumerate(self.categorical_features):
                encoder = self.label_encoders[feature]
                encoded = []
                for value in X_categorical_imputed[:, i]:
                    try:
                        encoded.append(int(encoder.transform([value])[0]))
                    except ValueError:
                        encoded.append(0)
                encoded_features.append(np.asarray(encoded, dtype=np.float64))
            processed_parts.append(np.column_stack(encoded_features))
        return np.hstack(processed_parts)

    def fit_transform(
        self, X: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]
    ) -> np.ndarray:
        self.fit(X, numeric_features, categorical_features)
        return self.transform(X)


class MIMICDataLoader:
    """Load cohort from CSV/Parquet under ``data_path``, or build a synthetic demo cohort."""

    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path

    def _load_table_file(self, path: Path, csv_kwargs: Optional[dict[str, Any]]) -> pd.DataFrame:
        suf = path.suffix.lower()
        if suf in (".parquet", ".pq"):
            return pd.read_parquet(path)
        kwargs = dict(csv_kwargs or {})
        return pd.read_csv(path, **kwargs)

    def _load_external_cohort(
        self,
        cohort_file: str,
        text_column: str,
        time_column: str,
        event_column: str,
        patient_id_column: Optional[str],
        csv_kwargs: Optional[dict[str, Any]],
    ) -> pd.DataFrame:
        raw = Path(cohort_file).expanduser()
        path = raw if raw.is_file() else Path(self.data_path) / cohort_file
        if not path.is_file():
            raise FileNotFoundError(f"Cohort file not found: {cohort_file!r} (resolved to {path})")

        df = self._load_table_file(path, csv_kwargs)
        need = [text_column, time_column, event_column]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Cohort file missing columns {missing}. Available: {list(df.columns)}")

        rename: dict[str, str] = {}
        if text_column != "clinical_notes":
            rename[text_column] = "clinical_notes"
        if time_column != "time":
            rename[time_column] = "time"
        if event_column != "event":
            rename[event_column] = "event"
        if rename:
            df = df.rename(columns=rename)

        if patient_id_column and patient_id_column in df.columns and patient_id_column != "subject_id":
            df = df.rename(columns={patient_id_column: "subject_id"})
        if "subject_id" not in df.columns:
            df.insert(0, "subject_id", np.arange(len(df), dtype=np.int64))

        df["clinical_notes"] = df["clinical_notes"].fillna("").astype(str)
        ev = pd.to_numeric(df["event"], errors="coerce")
        if ev.isna().any():
            raise ValueError("event column contains non-numeric values after coercion.")
        df["event"] = ev.astype(int).clip(0, 1)

        tt = pd.to_numeric(df["time"], errors="coerce")
        if tt.isna().any():
            raise ValueError("time column contains NaN; survival requires observed times.")
        df["time"] = tt.astype(np.float64).clip(lower=1e-6)

        print(f"Loaded external cohort from {path} | n={len(df)} | event rate: {df['event'].mean():.2%}")
        return df

    def load_cohort(
        self,
        outcome: str = "mortality",
        min_age: int = 18,
        max_los: int = 365,
        n_patients: int = 5000,
        random_state: int = 42,
        cohort_csv: Optional[str] = None,
        text_column: str = "clinical_notes",
        time_column: str = "time",
        event_column: str = "event",
        patient_id_column: Optional[str] = "subject_id",
        csv_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if cohort_csv:
            return self._load_external_cohort(
                cohort_csv,
                text_column=text_column,
                time_column=time_column,
                event_column=event_column,
                patient_id_column=patient_id_column,
                csv_kwargs=csv_kwargs,
            )

        print("Loading cohort (synthetic demo cohort if no external CSV is wired)...")
        rng = np.random.default_rng(random_state)
        n = n_patients

        data = {
            "subject_id": np.arange(n),
            "hadm_id": np.arange(n),
            "age": rng.integers(min_age, 90, size=n),
            "gender": rng.choice(["M", "F"], size=n),
            "ethnicity": rng.choice(["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"], size=n),
            "insurance": rng.choice(["Medicare", "Medicaid", "Private", "Government"], size=n),
            "admission_type": rng.choice(["EMERGENCY", "ELECTIVE", "URGENT"], size=n),
            "heart_rate": rng.normal(80, 15, n) * (rng.random(n) > 0.1),
            "systolic_bp": rng.normal(120, 20, n) * (rng.random(n) > 0.1),
            "diastolic_bp": rng.normal(80, 15, n) * (rng.random(n) > 0.1),
            "temperature": rng.normal(37, 0.8, n) * (rng.random(n) > 0.1),
            "respiratory_rate": rng.normal(18, 4, n) * (rng.random(n) > 0.1),
            "spo2": rng.normal(97, 3, n) * (rng.random(n) > 0.1),
            "glucose": rng.normal(110, 30, n) * (rng.random(n) > 0.2),
            "creatinine": rng.lognormal(0, 0.5, n) * (rng.random(n) > 0.2),
            "hemoglobin": rng.normal(13, 2, n) * (rng.random(n) > 0.2),
            "wbc": rng.normal(8, 3, n) * (rng.random(n) > 0.2),
            "clinical_notes": [self._generate_synthetic_note(i, rng) for i in range(n)],
            "los_days": np.abs(rng.lognormal(1.5, 1.0, n)),
        }
        df = pd.DataFrame(data)
        numeric_cols = [
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
        for col in numeric_cols:
            df.loc[df[col] == 0, col] = np.nan

        if outcome == "mortality":
            risk_score = (
                (df["age"] - 50) / 40
                + np.abs(df["heart_rate"].fillna(80) - 80) / 40
                + np.abs(df["systolic_bp"].fillna(120) - 120) / 60
                + rng.normal(0, 0.35, n)
            )
            df["mortality"] = (risk_score > np.percentile(risk_score, 65)).astype(int)
            df["event"] = df["mortality"]
        elif outcome == "readmission_30d":
            risk_score = (df["age"] > 65).astype(int) + (df["admission_type"] == "EMERGENCY").astype(int) + rng.normal(
                0, 0.35, n
            )
            df["readmission_30d"] = (risk_score > np.percentile(risk_score, 65)).astype(int)
            df["event"] = df["readmission_30d"]
        else:
            raise ValueError(f"Unknown outcome: {outcome}")

        df["los_days"] = df["los_days"].clip(0.1, max_los)
        df["time"] = df["los_days"]

        print(f"Loaded cohort: {len(df)} patients | event rate: {df['event'].mean():.2%}")
        return df

    def _generate_synthetic_note(self, patient_id: int, rng: np.random.Generator) -> str:
        templates = [
            f"Patient {patient_id} admitted with acute respiratory distress. Vital signs stable. Supplemental oxygen.",
            f"Patient {patient_id} presents with chest pain. ECG sinus rhythm. Troponin negative. Musculoskeletal pain suspected.",
            f"Patient {patient_id} transferred with sepsis. Cultures pending. Broad-spectrum antibiotics. Stable on fluids.",
            f"Patient {patient_id} post-operative day 1. Pain controlled. Ambulating. No infection signs.",
            f"Patient {patient_id} CHF exacerbation. Crackles bilaterally. Diuretics. Strict I/O and weights.",
        ]
        return str(rng.choice(templates))
