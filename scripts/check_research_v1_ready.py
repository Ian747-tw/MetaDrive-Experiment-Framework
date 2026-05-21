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


def run_dir_for_artifact(path: Path, fallback: Path) -> Path:
    if path.parent.name in {"checkpoints", "eval", "buffers"} and len(path.parents) > 1 and path.parents[1] != Path("."):
        return path.parents[1]
    if path.parent != Path("."):
        return path.parent
    return fallback


def copy_command(source: Path, destination: Path) -> str:
    if source == destination:
        return ""
    lines: list[str] = []
    if destination.parent != Path("."):
        lines.append(f"mkdir -p {destination.parent}")
    lines.append(f"cp {source} {destination}")
    return "\n".join(lines)


def commands(root: Path, checkpoint: Path, eval_csv: Path, buffer: Path) -> str:
    base_run = run_dir_for_artifact(checkpoint, root / "base_pretrain_s42")
    eval_run = run_dir_for_artifact(eval_csv, root / "eval_base_pretrain")
    explore_run = run_dir_for_artifact(buffer, root / "base_explore_large")
    generated_checkpoint = base_run / "checkpoints/final.zip"
    generated_eval_csv = eval_run / "eval/heldout_random.csv"
    generated_buffer = explore_run / "buffers/failure_buffer.jsonl"
    command_lines = [
        "Commands to build missing local artifacts:",
        f"python scripts/train.py --config configs/research_v1/base_pretrain.yaml experiment.output_dir={base_run}",
    ]
    if checkpoint_copy := copy_command(generated_checkpoint, checkpoint):
        command_lines.extend(checkpoint_copy.splitlines())
    command_lines.append(
        f"python scripts/evaluate.py --config configs/research_v1/base_eval.yaml --checkpoint {checkpoint} experiment.output_dir={eval_run} algorithm.checkpoint_path={checkpoint}"
    )
    if eval_copy := copy_command(generated_eval_csv, eval_csv):
        command_lines.extend(eval_copy.splitlines())
    command_lines.append(
        f"python scripts/explore_failures.py --config configs/research_v1/base_explore_large.yaml experiment.output_dir={explore_run} algorithm.checkpoint_path={checkpoint}"
    )
    if buffer_copy := copy_command(generated_buffer, buffer):
        command_lines.extend(buffer_copy.splitlines())
    command_lines.append(
        f"python scripts/check_research_v1_ready.py --root {root} --checkpoint {checkpoint} --eval-csv {eval_csv} --buffer {buffer} --min-failures 1000"
    )
    return "\n".join(command_lines) + "\n"


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
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-csv")
    parser.add_argument("--buffer")
    parser.add_argument("--min-failures", type=int, default=1000)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-success-rate", type=float, default=0.10)
    parser.add_argument("--min-route-completion", type=float, default=0.35)
    parser.add_argument("--max-timeout-rate", type=float, default=0.95)
    parser.add_argument("--max-unknown-fraction", type=float, default=0.25)
    args = parser.parse_args()
    root = Path(args.root)
    base_checkpoint = Path(args.checkpoint) if args.checkpoint else root / "base_pretrain_s42/checkpoints/final.zip"
    base_eval_csv = Path(args.eval_csv) if args.eval_csv else root / "eval_base_pretrain/eval/heldout_random.csv"
    failure_buffer = Path(args.buffer) if args.buffer else root / "base_explore_large/buffers/failure_buffer.jsonl"

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
            buffer_failures, warnings, summary = summarize(
                load_records(failure_buffer),
                args.min_failures,
                max_unknown_fraction=args.max_unknown_fraction,
                require_multiple_modes=True,
            )
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
        print(commands(root, base_checkpoint, base_eval_csv, failure_buffer))
        return 1
    print("Research V1 readiness: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
