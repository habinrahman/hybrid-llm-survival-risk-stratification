"""
Advanced reporting: risk tiers, intervals, SHAP, KM curves, PDF, logging, charts.

Designed for use with ``HybridSurvivalPipeline`` and inference scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None  # type: ignore[misc, assignment]

try:
    import shap
except ImportError:
    shap = None  # type: ignore[misc, assignment]

try:
    from lifelines import KaplanMeierFitter
except ImportError:
    KaplanMeierFitter = None  # type: ignore[misc, assignment]


def categorize_risk(score: float) -> str:
    """
    Map a scalar risk score to a clinical-style tier.

    Thresholds assume ``score`` is on a **0–1** scale (e.g. min–max normalized
    risk vs the training cohort). Values outside [0, 1] are clipped for labeling.
    """
    s = float(np.clip(score, 0.0, 1.0))
    if s < 0.3:
        return "Low Risk"
    if s < 0.7:
        return "Medium Risk"
    return "High Risk"


def normalize_risk_for_stratification(
    risk: Union[float, np.ndarray],
    train_risks: np.ndarray,
) -> np.ndarray:
    """Min–max normalize risk using training-set min/max (leakage-safe if train_risks is train-only)."""
    lo = float(np.min(train_risks))
    hi = float(np.max(train_risks))
    denom = hi - lo + 1e-12
    return (np.asarray(risk, dtype=np.float64) - lo) / denom


def confidence_interval(predictions: Sequence[float]) -> Tuple[float, float]:
    """
    Gaussian-style interval: mean ± 1.96 · std of the provided sample.

    For a single value, std is 0 and the interval collapses to that value.
    For multiple predicted risks (e.g. test set), this summarizes the
    **spread** of scores (not a formal CI for the mean unless interpreted carefully).
    """
    arr = np.asarray(predictions, dtype=np.float64).ravel()
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    return lower, upper


def confidence_interval_mean(risks: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    """Classic normal approximation for the mean: mean ± z · (s / sqrt(n))."""
    arr = np.asarray(risks, dtype=np.float64).ravel()
    n = max(len(arr), 1)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n)
    return m - z * se, m + z * se


def setup_logging(
    logs_dir: Union[str, Path] = "logs",
    filename: str = "project.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure file logging under ``logs_dir`` and return the application logger.

    Also attaches a stream handler so pipeline progress appears on the console.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / filename

    logger = logging.getLogger("hybrid_survival")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def plot_model_comparison(
    model_names: List[str],
    c_index: List[float],
    save_path: Union[str, Path],
    title: str = "Model Comparison (C-Index)",
) -> Path:
    """Bar chart of concordance indices; saves PNG and returns path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    plt.bar(model_names, c_index, color=colors)
    plt.ylim(0.0, 1.0)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    plt.title(title)
    plt.ylabel("C-Index")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_kaplan_meier(
    durations: np.ndarray,
    events: np.ndarray,
    save_path: Union[str, Path],
    title: str = "Kaplan–Meier Survival Curve (observed)",
) -> Path:
    """Non-parametric KM fit on observed times and event indicators."""
    if KaplanMeierFitter is None:
        raise ImportError("lifelines is required for Kaplan–Meier plots.")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(durations, dtype=float), event_observed=np.asarray(events, dtype=int))
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax, label="KM estimate")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def generate_pdf_report(
    risk_score: float,
    risk_category: str,
    output_path: Union[str, Path],
    extra_lines: Optional[List[str]] = None,
) -> Path:
    """Minimal one-page PDF report (requires ``fpdf2`` / ``fpdf``)."""
    if FPDF is None:
        raise ImportError("Install fpdf2: pip install fpdf2")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt="Patient Risk Prediction Report", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, txt=f"Risk Score: {risk_score:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, txt=f"Risk Category: {risk_category}", new_x="LMARGIN", new_y="NEXT")
    if extra_lines:
        for line in extra_lines:
            pdf.cell(0, 10, txt=line[:120], new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(output_path))
    return output_path


def generate_shap_explanation(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    save_path: Union[str, Path],
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_background: int = 300,
    max_eval: int = 200,
    feature_names: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    SHAP summary for a black-box ``predict_fn`` (defaults to ``model.predict_risk``).

    Uses a small background and evaluation subset for speed. Returns ``None`` if
    ``shap`` is unavailable or execution fails (logged, non-fatal).
    """
    if shap is None:
        _log.warning("SHAP not installed; skip explanation plot.")
        return None

    def _pred(X: Any) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float64)
        fn = predict_fn or (lambda z: model.predict_risk(z))
        out = fn(Xn)
        return np.asarray(out, dtype=np.float64).ravel()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bg = np.asarray(X_train, dtype=np.float64)
    if len(bg) > max_background:
        idx = np.random.choice(len(bg), max_background, replace=False)
        bg = bg[idx]
    Xe = np.asarray(X_test, dtype=np.float64)
    if len(Xe) > max_eval:
        idx = np.random.choice(len(Xe), max_eval, replace=False)
        Xe = Xe[idx]

    try:
        masker = shap.maskers.Independent(bg)
        explainer = shap.Explainer(_pred, masker)
        sv = explainer(Xe)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, Xe, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        return save_path
    except Exception as e:
        _log.warning("SHAP explanation skipped: %s", e)
        return None
