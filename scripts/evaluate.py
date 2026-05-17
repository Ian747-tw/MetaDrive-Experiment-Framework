from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.config import load_config
from fasb.core.run_dir import create_run_dir
from fasb.evaluation.evaluator import Evaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    run_dir = create_run_dir(cfg.experiment.output_dir, cfg, cfg.experiment.name)
    scenario_set = cfg.eval.get("scenario_set", "heldout_random")
    metrics = Evaluator(cfg, run_dir).evaluate_checkpoint(args.checkpoint or cfg.algorithm.get("checkpoint_path"), scenario_set, int(cfg.eval.n_episodes), bool(cfg.eval.get("deterministic", True)))
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
