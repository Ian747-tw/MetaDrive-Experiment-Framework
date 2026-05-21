from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


MAIN_COLUMNS = [
    "method",
    "eval_run",
    "success_rate",
    "collision_rate",
    "offroad_rate",
    "timeout_rate",
    "route_completion_mean",
    "episode_cost_mean",
    "cost_violation_rate",
    "safety_efficiency_score",
    "csv_path",
]


def aggregate(root: str | Path, output: str | Path | None = None) -> tuple[Path, Path | None]:
    root_path = Path(root)
    eval_csvs = sorted(root_path.glob("eval_*/eval/heldout_random.csv"))
    if not eval_csvs:
        raise FileNotFoundError(f"no eval CSVs found under {root_path}/eval_*/eval/heldout_random.csv")

    rows: list[dict[str, object]] = []
    for csv_path in eval_csvs:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        eval_run = csv_path.parents[1].name
        method = eval_run.removeprefix("eval_").removesuffix("_s42")
        row.update({"method": method, "eval_run": eval_run, "csv_path": str(csv_path)})
        rows.append(row)
    if not rows:
        raise ValueError("eval CSVs were found, but none had rows")

    output_path = Path(output) if output else root_path / "summary_main_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    main_df = pd.DataFrame(rows)
    columns = [column for column in MAIN_COLUMNS if column in main_df.columns]
    extra_columns = [column for column in main_df.columns if column not in columns]
    main_df[columns + extra_columns].to_csv(output_path, index=False)

    mode_csvs = sorted(root_path.glob("eval_*/analysis/failure_by_mode.csv"))
    mode_output = None
    if mode_csvs:
        mode_rows = []
        for csv_path in mode_csvs:
            df = pd.read_csv(csv_path)
            eval_run = csv_path.parents[1].name
            df.insert(0, "eval_run", eval_run)
            df.insert(0, "method", eval_run.removeprefix("eval_").removesuffix("_s42"))
            df["csv_path"] = str(csv_path)
            mode_rows.append(df)
        mode_output = root_path / "summary_failure_by_mode.csv"
        pd.concat(mode_rows, ignore_index=True).to_csv(mode_output, index=False)
    else:
        print(f"WARNING: no failure_by_mode.csv files found under {root_path}/eval_*/analysis/")
    return output_path, mode_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        main_path, mode_path = aggregate(args.root, args.output)
    except Exception as exc:
        print(f"FAIL aggregate results: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote main summary: {main_path}")
    if mode_path:
        print(f"Wrote failure-by-mode summary: {mode_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
