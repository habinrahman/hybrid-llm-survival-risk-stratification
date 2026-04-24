#!/usr/bin/env python3
"""Run full training + evaluation (backward-compatible entry point)."""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    print(
        """
    Hybrid LLM-Survival Model — train / evaluate / save artifacts
    """
    )
    try:
        from hybrid_survival.pipelines.hybrid_pipeline import HybridSurvivalPipeline

        cfg = ROOT / "configs" / "config.yaml"
        pipeline = HybridSurvivalPipeline(config_path=str(cfg))
        results = pipeline.run_full_pipeline()
        print("\nFinal comparison (CSV columns):")
        print(results.to_string(index=False))
        return 0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
