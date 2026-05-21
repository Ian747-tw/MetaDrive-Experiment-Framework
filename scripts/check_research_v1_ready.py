from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.imports import instantiate_from_config
from scripts.check_base_checkpoint_quality import check_quality, read_metrics
from scripts.check_failure_buffer_quality import load_records, summarize


DEFAULT_ROOT = Path("runs/research_v1")


def commands(root: Path) -> str:
    base_run = root / "base_pretrain_s42"
    eval_run = root / "eval_base_pretrain"
    explore_run = root / "base_explore"
    base_checkpoint = base_run / "checkpoints/final.zip"
    return f"""Commands to build missing local artifacts:
python scripts/train.py --config configs/research_v1/base_pretrain.yaml experiment.output_dir={base_run}
python scripts/evaluate.py --config configs/research_v1/base_eval.yaml --checkpoint {base_checkpoint} experiment.output_dir={eval_run} algorithm.checkpoint_path={base_checkpoint}
python scripts/explore_failures.py --config configs/research_v1/base_explore.yaml experiment.output_dir={explore_run} algorithm.checkpoint_path={base_checkpoint}
python scripts/check_research_v1_ready.py --root {root} --min-failures 30
"""


def check_configs() -> list[str]:
    failures: list[str] = []
    config_dir = Path("configs/research_v1")
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        return [f"no Research V1 configs found in {config_dir}"]
    for path in config_paths:
        try:
            OmegaConf.load(path)
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    return failures


def check_plugin_imports() -> list[str]:
    failures: list[str] = []
    targets = [
        "examples.custom_plugins.crash_only_cost.CrashOnlyCost",
        "examples.custom_plugins.near_miss_heavy_cost.NearMissHeavyCost",
        "examples.custom_plugins.near_failure_scorer.NearFailureScorer",
        "examples.custom_plugins.timeout_relaxed_budget.TimeoutRelaxedBudget",
        "examples.custom_plugins.fixed_penalty_scheduler.FixedPenaltyScheduler",
    ]
    for target in targets:
        try:
            instantiate_from_config({"_target_": target})
        except Exception as exc:
            failures.append(f"{target}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--min-failures", type=int, default=30)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-success-rate", type=float, default=0.10)
    parser.add_argument("--min-route-completion", type=float, default=0.35)
    parser.add_argument("--max-timeout-rate", type=float, default=0.95)
    args = parser.parse_args()
    root = Path(args.root)
    base_checkpoint = root / "base_pretrain_s42/checkpoints/final.zip"
    base_eval_csv = root / "eval_base_pretrain/eval/heldout_random.csv"
    failure_buffer = root / "base_explore/buffers/failure_buffer.jsonl"

    failures: list[str] = []
    failures.extend(check_configs())
    failures.extend(check_plugin_imports())

    if not base_checkpoint.exists():
        failures.append(f"base checkpoint missing: {base_checkpoint}")
    if not base_eval_csv.exists():
        failures.append(f"base eval CSV missing: {base_eval_csv}")
    else:
        try:
            quality_failures = check_quality(
                read_metrics(base_eval_csv),
                args.min_episodes,
                args.min_success_rate,
                args.min_route_completion,
                args.max_timeout_rate,
            )
            failures.extend(f"base checkpoint quality: {failure}" for failure in quality_failures)
        except Exception as exc:
            failures.append(f"base checkpoint quality check failed: {exc}")
    if not failure_buffer.exists():
        failures.append(f"failure buffer missing: {failure_buffer}")
    else:
        try:
            buffer_failures, warnings, summary = summarize(load_records(failure_buffer), args.min_failures)
            failures.extend(f"failure buffer quality: {failure}" for failure in buffer_failures)
            failures.extend(f"failure buffer quality warning: {warning}" for warning in warnings)
            print(f"Failure buffer summary: {summary}")
        except Exception as exc:
            failures.append(f"failure buffer quality check failed: {exc}")

    if failures:
        print("Research V1 readiness: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        print()
        print(commands(root))
        return 1
    print("Research V1 readiness: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
