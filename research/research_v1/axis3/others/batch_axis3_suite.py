from __future__ import annotations

import argparse
import csv
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    key: str
    config: str
    overrides: list[str]


@dataclass(frozen=True)
class JobResult:
    rep: int
    variant: str
    train_run: str
    eval_run: str
    heldout_csv: Path
    metrics: dict[str, float]


def _read_metrics(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    out: dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _train_and_eval_one(
    *,
    repo_root: Path,
    python: str,
    runs_root: str,
    seed: int,
    rep: int,
    train_timesteps: int,
    save_every: int,
    eval_episodes: int,
    eval_start_seed: int,
    eval_num_scenarios: int,
    horizon: int,
    variant: Variant,
) -> JobResult:
    train_run = f"axis3_{variant.key}_s{seed}_rep{rep:02d}"
    train_dir = Path(runs_root) / train_run
    train_cmd = [
        python,
        "scripts/train.py",
        "--config",
        variant.config,
        f"experiment.name={train_run}",
        f"experiment.output_dir={train_dir.as_posix()}",
        f"training.total_timesteps={int(train_timesteps)}",
        f"training.save_every_steps={int(save_every)}",
        f"experiment.seed={int(seed)}",
        *variant.overrides,
    ]
    _run(train_cmd, repo_root)

    ckpt = train_dir / "checkpoints" / "final.zip"
    if not ckpt.exists():
        raise RuntimeError(f"missing checkpoint: {ckpt}")

    eval_run = f"eval_{train_run}"
    eval_dir = Path(runs_root) / eval_run
    eval_cmd = [
        python,
        "scripts/evaluate.py",
        "--config",
        "configs/eval/heldout_random.yaml",
        "--checkpoint",
        ckpt.as_posix(),
        f"experiment.name={eval_run}",
        f"experiment.output_dir={eval_dir.as_posix()}",
        f"eval.n_episodes={int(eval_episodes)}",
        f"metadrive.config.start_seed={int(eval_start_seed)}",
        f"metadrive.config.num_scenarios={int(eval_num_scenarios)}",
        f"metadrive.config.horizon={int(horizon)}",
    ]
    _run(eval_cmd, repo_root)

    heldout_csv = eval_dir / "eval" / "heldout_random.csv"
    if not heldout_csv.exists():
        raise RuntimeError(f"missing heldout csv: {heldout_csv}")
    metrics = _read_metrics(heldout_csv)
    return JobResult(
        rep=rep,
        variant=variant.key,
        train_run=train_run,
        eval_run=eval_run,
        heldout_csv=heldout_csv,
        metrics=metrics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Axis3 suite in parallel, repeated N times:\n"
            "  fixed 0.03, fixed 0.05, fixed 0.10, adaptive default/strict/loose\n"
            "Each job does train then heldout eval; a summary CSV is written at the end."
        )
    )
    parser.add_argument("--reps", type=int, default=10, help="Number of repeats.")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed.")
    parser.add_argument("--train-timesteps", type=int, default=100000, help="Training timesteps per run (screening).")
    parser.add_argument("--save-every", type=int, default=50000, help="Checkpoint frequency.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Eval episodes.")
    parser.add_argument("--eval-start-seed", type=int, default=5000, help="Eval start seed.")
    parser.add_argument("--eval-num-scenarios", type=int, default=200, help="Eval num scenarios.")
    parser.add_argument("--horizon", type=int, default=500, help="MetaDrive horizon.")
    parser.add_argument(
        "--python",
        default="/tmp2/b14902068/.venv/bin/python",
        help="Python executable (default matches ws7 venv).",
    )
    parser.add_argument("--runs-root", default="runs/research_v1", help="Runs root.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers.")
    parser.add_argument("--metric", default="safety_efficiency_score", help="Metric for ranking.")

    # Config roots (you said screening/ is outdated, so we keep these in final/committed configs)
    parser.add_argument(
        "--fixed-config",
        default="configs/research_v1/axis3_budget_fixed_default_final.yaml",
        help="Base fixed-budget config (you created).",
    )
    parser.add_argument(
        "--adaptive-config",
        default="configs/research_v1/axis3_budget_adaptive_default_final.yaml",
        help="Base adaptive config.",
    )

    # Defaults for strict/loose (override only allowed axis variables)
    parser.add_argument("--strict-d-min", type=float, default=0.02)
    parser.add_argument("--strict-d-max", type=float, default=0.10)
    parser.add_argument("--strict-timeout-budget", type=float, default=0.07)
    parser.add_argument("--strict-lambda-max", type=float, default=0.50)

    parser.add_argument("--loose-d-min", type=float, default=0.15)
    parser.add_argument("--loose-d-max", type=float, default=0.40)
    parser.add_argument("--loose-timeout-budget", type=float, default=0.40)
    parser.add_argument("--loose-lambda-max", type=float, default=0.10)

    args = parser.parse_args()

    if args.reps <= 0:
        raise SystemExit("--reps must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = runs_root / f"axis3_suite_summary_s{args.seed}_{stamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    variants: list[Variant] = [
        Variant(
            key="fixed003",
            config=args.fixed_config,
            overrides=[
                "safety_budget.mode=fixed",
                f"safety_budget.budget={Decimal('0.03')}",
            ],
        ),
        Variant(
            key="fixed005",
            config=args.fixed_config,
            overrides=[
                "safety_budget.mode=fixed",
                f"safety_budget.budget={Decimal('0.05')}",
            ],
        ),
        Variant(
            key="fixed010",
            config=args.fixed_config,
            overrides=[
                "safety_budget.mode=fixed",
                f"safety_budget.budget={Decimal('0.10')}",
            ],
        ),
        Variant(
            key="adaptive_default",
            config=args.adaptive_config,
            overrides=[
                "penalty_scheduler.lambda_min=0.0",
            ],
        ),
        Variant(
            key="adaptive_strict",
            config=args.adaptive_config,
            overrides=[
                "penalty_scheduler.lambda_min=0.0",
                f"penalty_scheduler.lambda_max={Decimal(str(args.strict_lambda_max))}",
                f"safety_budget.d_min={float(args.strict_d_min)}",
                f"safety_budget.d_max={float(args.strict_d_max)}",
                f"safety_budget.timeout_budget={float(args.strict_timeout_budget)}",
            ],
        ),
        Variant(
            key="adaptive_loose",
            config=args.adaptive_config,
            overrides=[
                "penalty_scheduler.lambda_min=0.0",
                f"penalty_scheduler.lambda_max={Decimal(str(args.loose_lambda_max))}",
                f"safety_budget.d_min={float(args.loose_d_min)}",
                f"safety_budget.d_max={float(args.loose_d_max)}",
                f"safety_budget.timeout_budget={float(args.loose_timeout_budget)}",
            ],
        ),
    ]

    def metric_of(metrics: dict[str, float]) -> float:
        v = metrics.get(args.metric)
        if v is None or math.isnan(float(v)):
            return float("-inf")
        return float(v)

    all_results: list[JobResult] = []

    for rep in range(1, int(args.reps) + 1):
        print(f"=== Rep {rep}/{args.reps} ===", flush=True)
        futures = []
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            for v in variants:
                futures.append(
                    ex.submit(
                        _train_and_eval_one,
                        repo_root=repo_root,
                        python=str(args.python),
                        runs_root=str(args.runs_root),
                        seed=int(args.seed),
                        rep=rep,
                        train_timesteps=int(args.train_timesteps),
                        save_every=int(args.save_every),
                        eval_episodes=int(args.eval_episodes),
                        eval_start_seed=int(args.eval_start_seed),
                        eval_num_scenarios=int(args.eval_num_scenarios),
                        horizon=int(args.horizon),
                        variant=v,
                    )
                )
            for fut in as_completed(futures):
                res = fut.result()
                all_results.append(res)
                print(
                    f"[rep {res.rep:02d}] {res.variant} {args.metric}={metric_of(res.metrics)} success={res.metrics.get('success_rate')}",
                    flush=True,
                )

        rep_results = [r for r in all_results if r.rep == rep]
        best_rep = max(rep_results, key=lambda r: metric_of(r.metrics))
        print(f"Best rep {rep}: {best_rep.variant} {args.metric}={metric_of(best_rep.metrics)}", flush=True)

    results_csv = summary_dir / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rep",
            "variant",
            "train_run",
            "eval_run",
            "metric",
            "success_rate",
            "collision_rate",
            "offroad_rate",
            "timeout_rate",
            "route_completion_mean",
            "episode_cost_mean",
            "safety_efficiency_score",
            "heldout_csv",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(all_results, key=lambda x: (x.rep, x.variant)):
            w.writerow(
                {
                    "rep": r.rep,
                    "variant": r.variant,
                    "train_run": r.train_run,
                    "eval_run": r.eval_run,
                    "metric": r.metrics.get(args.metric, ""),
                    "success_rate": r.metrics.get("success_rate", ""),
                    "collision_rate": r.metrics.get("collision_rate", ""),
                    "offroad_rate": r.metrics.get("offroad_rate", ""),
                    "timeout_rate": r.metrics.get("timeout_rate", ""),
                    "route_completion_mean": r.metrics.get("route_completion_mean", ""),
                    "episode_cost_mean": r.metrics.get("episode_cost_mean", ""),
                    "safety_efficiency_score": r.metrics.get("safety_efficiency_score", ""),
                    "heldout_csv": r.heldout_csv.as_posix(),
                }
            )

    best_overall = max(all_results, key=lambda r: metric_of(r.metrics))
    best_txt = summary_dir / "best_overall.txt"
    with best_txt.open("w", encoding="utf-8") as f:
        f.write(f"metric={args.metric}\n")
        f.write(f"best_variant={best_overall.variant}\n")
        f.write(f"best_value={metric_of(best_overall.metrics)}\n")
        f.write(f"best_train_run={best_overall.train_run}\n")
        f.write(f"best_eval_run={best_overall.eval_run}\n")
        f.write(f"heldout_csv={best_overall.heldout_csv.as_posix()}\n")

    print(f"Done. Summary written to: {summary_dir.as_posix()}", flush=True)
    print(f"Best overall: {best_overall.variant} {args.metric}={metric_of(best_overall.metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

