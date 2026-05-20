from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWN_RUN_DIRS = [
    ROOT / "runs" / "base_explore",
    ROOT / "runs" / "fasb_ppo",
    ROOT / "runs" / "heldout_random_eval",
]
EXPECTED_FILES = [
    ROOT / "runs" / "base_explore" / "buffers" / "failure_buffer.jsonl",
    ROOT / "runs" / "fasb_ppo" / "checkpoints" / "final.zip",
    ROOT / "runs" / "fasb_ppo" / "logs" / "episodes.jsonl",
    ROOT / "runs" / "heldout_random_eval" / "eval" / "heldout_random.csv",
    ROOT / "runs" / "heldout_random_eval" / "analysis" / "failure_summary.csv",
    ROOT / "runs" / "heldout_random_eval" / "analysis" / "failure_by_mode.csv",
    ROOT / "runs" / "heldout_random_eval" / "analysis" / "paper_numbers.md",
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


def verify_outputs() -> None:
    stage("Verify Expected Outputs")
    missing = [path for path in EXPECTED_FILES if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise SystemExit(f"Missing expected output files:\n{formatted}")

    nonempty_files = [
        ROOT / "runs" / "base_explore" / "buffers" / "failure_buffer.jsonl",
        ROOT / "runs" / "fasb_ppo" / "logs" / "episodes.jsonl",
        ROOT / "runs" / "heldout_random_eval" / "analysis" / "paper_numbers.md",
    ]
    for path in nonempty_files:
        size = path.stat().st_size
        if size <= 0:
            raise SystemExit(f"Expected non-empty file: {path.relative_to(ROOT)}")

    line_checks = [
        ROOT / "runs" / "base_explore" / "buffers" / "failure_buffer.jsonl",
        ROOT / "runs" / "fasb_ppo" / "logs" / "episodes.jsonl",
    ]
    for path in line_checks:
        lines = count_lines(path)
        if lines < 1:
            raise SystemExit(f"Expected at least one line in {path.relative_to(ROOT)}")
        print(f"{path.relative_to(ROOT)}: {lines} line(s)")

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
            "eval.n_episodes=2",
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
            "training.total_timesteps=32",
            "algorithm.params.n_steps=16",
            "algorithm.params.batch_size=16",
            "metadrive.config.horizon=30",
            "vec_env.type=dummy",
            "vec_env.n_envs=1",
            "algorithm.params.device=cpu",
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
            "runs/fasb_ppo/checkpoints/final.zip",
            "eval.n_episodes=2",
            "metadrive.config.horizon=30",
            "algorithm.params.device=cpu",
        ]
    )

    stage("Analyze Failure Outputs")
    run_command([sys.executable, "scripts/analyze_failures.py", "--run", "runs/heldout_random_eval"])

    verify_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
