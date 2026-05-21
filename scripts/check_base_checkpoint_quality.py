from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_COLUMNS = {"n_episodes", "success_rate", "route_completion_mean", "timeout_rate"}


def read_metrics(eval_csv: str | Path) -> dict[str, float]:
    path = Path(eval_csv)
    if not path.exists():
        raise FileNotFoundError(f"eval CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"eval CSV has no metric rows: {path}")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(f"eval CSV missing required metrics: {', '.join(missing)}")
    metrics: dict[str, float] = {}
    for key, value in rows[0].items():
        if value in (None, ""):
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def check_quality(
    metrics: dict[str, float],
    min_episodes: int,
    min_success_rate: float,
    min_route_completion: float,
    max_timeout_rate: float,
) -> list[str]:
    failures: list[str] = []
    if metrics["n_episodes"] < min_episodes:
        failures.append(f"n_episodes {metrics['n_episodes']:.0f} < {min_episodes}")
    if metrics["success_rate"] < min_success_rate:
        failures.append(f"success_rate {metrics['success_rate']:.4f} < {min_success_rate:.4f}")
    if metrics["route_completion_mean"] < min_route_completion:
        failures.append(
            f"route_completion_mean {metrics['route_completion_mean']:.4f} < {min_route_completion:.4f}"
        )
    if metrics["timeout_rate"] > max_timeout_rate:
        failures.append(f"timeout_rate {metrics['timeout_rate']:.4f} > {max_timeout_rate:.4f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-success-rate", type=float, default=0.10)
    parser.add_argument("--min-route-completion", type=float, default=0.35)
    parser.add_argument("--max-timeout-rate", type=float, default=0.95)
    args = parser.parse_args()

    try:
        metrics = read_metrics(args.eval_csv)
        failures = check_quality(
            metrics,
            args.min_episodes,
            args.min_success_rate,
            args.min_route_completion,
            args.max_timeout_rate,
        )
    except Exception as exc:
        print(f"FAIL base checkpoint quality: {exc}", file=sys.stderr)
        return 1

    print("Base checkpoint metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6f}")
    if failures:
        print("FAIL base checkpoint quality:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS base checkpoint quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
