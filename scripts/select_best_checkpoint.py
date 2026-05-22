from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.config import load_config
from fasb.evaluation.evaluator import Evaluator


METRIC_COLUMNS = [
    "n_episodes",
    "success_rate",
    "collision_rate",
    "offroad_rate",
    "timeout_rate",
    "route_completion_mean",
    "episode_reward_mean",
    "episode_modified_reward_mean",
    "episode_cost_mean",
    "cost_violation_rate",
    "avg_episode_length",
    "safety_efficiency_score",
]


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "final.zip":
        return (2, 0, path.name)
    if path.name == "best_mean_reward.zip":
        return (1, 0, path.name)
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (0, -1, path.name)


def discover_checkpoints(run_dir: str | Path) -> list[Path]:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    candidates: list[Path] = []
    candidates.extend(checkpoint_dir.glob("latest_*.zip"))
    for name in ("best_mean_reward.zip", "final.zip"):
        path = checkpoint_dir / name
        if path.exists():
            candidates.append(path)
    seen: set[Path] = set()
    unique = []
    for path in sorted(candidates, key=checkpoint_sort_key):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def hard_reject(metrics: dict[str, Any]) -> str:
    reasons = []
    if float(metrics.get("success_rate", 0.0)) < 0.20:
        reasons.append("success_rate < 0.20")
    if float(metrics.get("route_completion_mean", 0.0)) < 0.40:
        reasons.append("route_completion_mean < 0.40")
    if float(metrics.get("timeout_rate", 1.0)) > 0.80:
        reasons.append("timeout_rate > 0.80")
    return "; ".join(reasons)


def select_best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    eligible = [row for row in rows if not row.get("hard_reject")]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row.get(metric, float("-inf"))),
            float(row.get("route_completion_mean", float("-inf"))),
            -float(row.get("collision_rate", 0.0)) - float(row.get("offroad_rate", 0.0)),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", "selected", "hard_reject", "reject_reasons", *METRIC_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], selected: dict[str, Any] | None, metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Checkpoint Selection Report",
        "",
        f"Selection metric: `{metric}`",
        "",
        "| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["checkpoint"]),
                    str(bool(row.get("selected", False))),
                    str(bool(row.get("hard_reject", False))),
                    str(row.get("reject_reasons", "")),
                    f"{float(row.get('success_rate', 0.0)):.4f}",
                    f"{float(row.get('timeout_rate', 0.0)):.4f}",
                    f"{float(row.get('route_completion_mean', 0.0)):.4f}",
                    f"{float(row.get('collision_rate', 0.0)):.4f}",
                    f"{float(row.get('offroad_rate', 0.0)):.4f}",
                    f"{float(row.get('episode_cost_mean', 0.0)):.4f}",
                    f"{float(row.get('safety_efficiency_score', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Selected checkpoint", ""])
    if selected is None:
        lines.append("No checkpoint passed the hard reject screen.")
    else:
        lines.append(f"`{selected['checkpoint']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_checkpoint(
    checkpoint: Path,
    eval_config: str | Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    eval_run_dir = output_dir / "evals" / checkpoint.stem
    overrides = [
        f"experiment.name=select_{checkpoint.stem}",
        f"experiment.output_dir={eval_run_dir.as_posix()}",
        f"eval.n_episodes={args.eval_episodes}",
        f"metadrive.config.start_seed={args.eval_start_seed}",
        f"metadrive.config.num_scenarios={args.eval_num_scenarios}",
        f"metadrive.config.horizon={args.horizon}",
        f"metadrive.config.traffic_density={args.traffic_density}",
        "algorithm.params.device=cpu",
    ]
    cfg = load_config(eval_config, overrides)
    scenario_set = cfg.eval.get("scenario_set", "heldout_random")
    return Evaluator(cfg, eval_run_dir).evaluate_checkpoint(
        checkpoint.as_posix(),
        scenario_set,
        int(cfg.eval.n_episodes),
        bool(cfg.eval.get("deterministic", True)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="safety_efficiency_score")
    parser.add_argument("--eval-start-seed", type=int, required=True)
    parser.add_argument("--eval-num-scenarios", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--traffic-density", type=float, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {run_dir / 'checkpoints'}")

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        metrics = evaluate_checkpoint(checkpoint, args.eval_config, output_dir, args)
        reject_reasons = hard_reject(metrics)
        row = {
            "checkpoint": checkpoint.as_posix(),
            "selected": False,
            "hard_reject": bool(reject_reasons),
            "reject_reasons": reject_reasons,
            **metrics,
        }
        rows.append(row)

    selected = select_best(rows, args.metric)
    if selected is not None:
        selected["selected"] = True
        destination = run_dir / "checkpoints" / "selected_dev_best.zip"
        shutil.copyfile(selected["checkpoint"], destination)

    write_csv(output_dir / "checkpoint_selection.csv", rows)
    write_report(output_dir / "checkpoint_selection_report.md", rows, selected, args.metric)
    if selected is None:
        print("No checkpoint passed hard reject screen")
    else:
        print(f"Selected checkpoint: {selected['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
