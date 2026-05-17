from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.config import load_config
from fasb.training.sb3_trainer import SB3Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    final = SB3Trainer(cfg).train()
    print(f"Saved final checkpoint to {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
