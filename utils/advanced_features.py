"""
Shim: re-exports advanced helpers from the canonical package
``hybrid_survival.utils.advanced_features``.

Use either::

    from hybrid_survival.utils.advanced_features import categorize_risk

or, with the project root on ``PYTHONPATH``::

    from utils.advanced_features import categorize_risk
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from hybrid_survival.utils.advanced_features import (  # noqa: E402,F401
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

__all__ = [
    "categorize_risk",
    "confidence_interval",
    "confidence_interval_mean",
    "normalize_risk_for_stratification",
    "setup_logging",
    "plot_model_comparison",
    "plot_kaplan_meier",
    "generate_pdf_report",
    "generate_shap_explanation",
]
