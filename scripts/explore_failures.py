from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.buffers.failure_buffer import FailureBuffer
from fasb.core.config import load_config
from fasb.core.run_dir import create_run_dir
from fasb.evaluation.evaluator import Evaluator
from fasb.utils.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    run_dir = create_run_dir(cfg.experiment.output_dir, cfg, cfg.experiment.name)
    evaluator = Evaluator(cfg, run_dir)
    evaluator.evaluate_checkpoint(cfg.algorithm.get("checkpoint_path"), "train_random_debug", int(cfg.eval.n_episodes))
    records = read_jsonl(Path(run_dir) / "logs" / "episodes.jsonl")
    buffer = FailureBuffer(max_size=int(cfg.failure_buffer.get("max_size", 1000)))
    for record in records:
        if record.get("risk_score", 0.0) > 0.1 or not record.get("success", False):
            buffer.add(record)
    path = Path(run_dir) / "buffers" / "failure_buffer.jsonl"
    buffer.save(path)
    print(f"Wrote {len(buffer)} failures to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
