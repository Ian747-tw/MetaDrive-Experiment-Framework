from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "e2e_stress"
KNOWN_RUN_DIRS = [
    RUN_ROOT / "base_explore",
    RUN_ROOT / "fasb_ppo",
    RUN_ROOT / "heldout_random_eval",
]
EXPECTED_FILES = [
    RUN_ROOT / "base_explore" / "buffers" / "failure_buffer.jsonl",
    RUN_ROOT / "fasb_ppo" / "checkpoints" / "final.zip",
    RUN_ROOT / "fasb_ppo" / "logs" / "episodes.jsonl",
    RUN_ROOT / "heldout_random_eval" / "eval" / "heldout_random.csv",
    RUN_ROOT / "heldout_random_eval" / "analysis" / "failure_summary.csv",
    RUN_ROOT / "heldout_random_eval" / "analysis" / "failure_by_mode.csv",
    RUN_ROOT / "heldout_random_eval" / "analysis" / "paper_numbers.md",
]


def stage(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}", flush=True)


def run_command(args: list[str]) -> None:
    display = " ".join(args)
    print(f"$ {display}", flush=True)
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {display}")


def clean_runs() -> None:
    stage("Clean Known Stress Run Directories")
    for run_dir in KNOWN_RUN_DIRS:
        if run_dir.exists():
            print(f"Removing {run_dir.relative_to(ROOT)}")
            shutil.rmtree(run_dir)
        else:
            print(f"Skipping missing {run_dir.relative_to(ROOT)}")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_outputs() -> None:
    stage("Verify Expected Outputs")
    missing = [path for path in EXPECTED_FILES if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise SystemExit(f"Missing expected output files:\n{formatted}")

    nonempty_files = [
        RUN_ROOT / "base_explore" / "buffers" / "failure_buffer.jsonl",
        RUN_ROOT / "fasb_ppo" / "logs" / "episodes.jsonl",
        RUN_ROOT / "heldout_random_eval" / "analysis" / "paper_numbers.md",
    ]
    for path in nonempty_files:
        size = path.stat().st_size
        if size <= 0:
            raise SystemExit(f"Expected non-empty file: {path.relative_to(ROOT)}")

    line_checks = [
        RUN_ROOT / "base_explore" / "buffers" / "failure_buffer.jsonl",
        RUN_ROOT / "fasb_ppo" / "logs" / "episodes.jsonl",
    ]
    for path in line_checks:
        lines = count_lines(path)
        if lines < 1:
            raise SystemExit(f"Expected at least one line in {path.relative_to(ROOT)}")
        print(f"{path.relative_to(ROOT)}: {lines} line(s)")

    failure_records = read_jsonl(RUN_ROOT / "base_explore" / "buffers" / "failure_buffer.jsonl")
    if not any(record.get("seed") is not None for record in failure_records):
        raise SystemExit("Expected at least one failure-buffer record with a concrete seed")

    training_records = read_jsonl(RUN_ROOT / "fasb_ppo" / "logs" / "episodes.jsonl")
    if not any(record.get("scenario_source") == "failure_buffer" for record in training_records):
        raise SystemExit("Expected training to consume at least one failure-buffer scenario")
    print("Verified training consumed a failure-buffer scenario.")

    print("All expected stress outputs are present.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end FASB stress check.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip MetaDrive smoke test stage.")
    parser.add_argument(
        "--clean-runs",
        action="store_true",
        help="Remove only known stress run directories before running.",
    )
    args = parser.parse_args()

    if args.clean_runs:
        clean_runs()

    stage("Validate Configured Components")
    run_command(
        [
            sys.executable,
            "scripts/validate_components.py",
            "--config",
            "configs/train/fasb_ppo.yaml",
        ]
    )

    if not args.skip_smoke:
        stage("Smoke Test MetaDrive Env")
        run_command(
            [
                sys.executable,
                "scripts/smoke_test_env.py",
                "--config",
                "configs/env/metadrive_debug.yaml",
            ]
        )

    stage("Explore Tiny Failure Buffer")
    run_command(
        [
            sys.executable,
            "scripts/explore_failures.py",
            "--config",
            "configs/explore/base_checkpoint.yaml",
            "experiment.name=e2e_stress_base_explore",
            "experiment.output_dir=runs/e2e_stress/base_explore",
            "eval.n_episodes=2",
            "metadrive.config.start_seed=1000",
            "metadrive.config.num_scenarios=200",
            "metadrive.config.horizon=50",
        ]
    )

    stage("Train Tiny FASB-PPO")
    run_command(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/train/fasb_ppo.yaml",
            "experiment.name=e2e_stress_fasb_ppo",
            "experiment.output_dir=runs/e2e_stress/fasb_ppo",
            "failure_buffer.path=runs/e2e_stress/base_explore/buffers/failure_buffer.jsonl",
            "training.total_timesteps=32",
            "algorithm.params.n_steps=16",
            "algorithm.params.batch_size=16",
            "metadrive.config.horizon=30",
            "vec_env.type=dummy",
            "vec_env.n_envs=1",
            "algorithm.params.device=cpu",
            "sampler.failure_ratio=1.0",
        ]
    )

    stage("Evaluate Final Checkpoint")
    run_command(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            "configs/eval/heldout_random.yaml",
            "--checkpoint",
            "runs/e2e_stress/fasb_ppo/checkpoints/final.zip",
            "experiment.name=e2e_stress_heldout_random_eval",
            "experiment.output_dir=runs/e2e_stress/heldout_random_eval",
            "eval.n_episodes=2",
            "metadrive.config.horizon=30",
            "algorithm.params.device=cpu",
        ]
    )

    stage("Analyze Failure Outputs")
    run_command([sys.executable, "scripts/analyze_failures.py", "--run", "runs/e2e_stress/heldout_random_eval"])

    verify_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
