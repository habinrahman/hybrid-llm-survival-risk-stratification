"""
Backward-compatible entry point.
Delegates to `HybridSurvivalPipeline` in `src/hybrid_survival`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_survival.pipelines.hybrid_pipeline import HybridSurvivalPipeline  # noqa: E402


def main() -> None:
    pipeline = HybridSurvivalPipeline(config_path=str(ROOT / "configs" / "config.yaml"))
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
