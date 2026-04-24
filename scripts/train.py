#!/usr/bin/env python3
"""Train full hybrid pipeline (CLI)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_survival.pipelines.hybrid_pipeline import HybridSurvivalPipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = p.parse_args()
    pipe = HybridSurvivalPipeline(config_path=args.config)
    pipe.run_full_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
