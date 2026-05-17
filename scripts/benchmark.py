from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.config import load_config
from fasb.core.run_dir import create_run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    create_run_dir(cfg.experiment.output_dir, cfg, cfg.experiment.name)
    commands = [
        ["python", "scripts/evaluate.py", "--config", "configs/eval/heldout_random.yaml"],
        ["python", "scripts/explore_failures.py", "--config", "configs/explore/base_checkpoint.yaml", "eval.n_episodes=5"],
        ["python", "scripts/train.py", "--config", "configs/train/naive_ft.yaml", "training.total_timesteps=1000"],
        ["python", "scripts/train.py", "--config", "configs/train/fixed_budget_ft.yaml", "training.total_timesteps=1000"],
        ["python", "scripts/train.py", "--config", "configs/train/fasb_ppo.yaml", "training.total_timesteps=1000"],
        ["python", "scripts/analyze_failures.py", "--run", "runs/fasb_ppo"],
    ]
    for cmd in commands:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
