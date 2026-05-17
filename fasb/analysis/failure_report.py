from __future__ import annotations

from pathlib import Path

import pandas as pd

from fasb.evaluation.metrics import summarize_episode_records
from fasb.utils.io import read_jsonl


def generate_failure_report(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(root / "logs" / "episodes.jsonl")
    df = pd.DataFrame(records)
    outputs: dict[str, Path] = {}
    summary = summarize_episode_records(records)
    summary_path = analysis / "failure_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    outputs["failure_summary"] = summary_path
    if not df.empty and "failure_mode" in df:
        by_mode = df.groupby("failure_mode", dropna=False).agg(
            episodes=("failure_mode", "size"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            offroad_rate=("offroad", "mean"),
            cost_mean=("cost", "mean"),
        )
        by_mode_path = analysis / "failure_by_mode.csv"
        by_mode.to_csv(by_mode_path)
        outputs["failure_by_mode"] = by_mode_path
    paper = analysis / "paper_numbers.md"
    paper.write_text(_paper_numbers(summary), encoding="utf-8")
    outputs["paper_numbers"] = paper
    return outputs


def _paper_numbers(summary: dict[str, float]) -> str:
    lines = ["# Paper Numbers", "", "| Metric | Value |", "|---|---:|"]
    for key, value in summary.items():
        lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines) + "\n"
