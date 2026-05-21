from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    buffer_path = Path(path)
    if not buffer_path.exists():
        raise FileNotFoundError(f"failure buffer not found: {buffer_path}")
    records: list[dict[str, Any]] = []
    with buffer_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def summarize(records: list[dict[str, Any]], min_records: int) -> tuple[list[str], list[str], dict[str, Any]]:
    failures: list[str] = []
    warnings: list[str] = []
    if len(records) < min_records:
        failures.append(f"record count {len(records)} < {min_records}")
    seeds = [record.get("seed") for record in records if record.get("seed") is not None]
    if not seeds:
        failures.append("no records have seed")
    modes = Counter(str(record.get("failure_mode", "unknown")) for record in records)
    if len(modes) <= 1 and records:
        warnings.append("only one failure mode present")
    if records and all(bool(record.get("success", record.get("solved", False))) for record in records):
        warnings.append("all records are solved")
    if len(set(seeds)) <= 1 and records:
        warnings.append("all records share the same seed")
    risk_values = [
        float(record.get("risk_score", record.get("failure_score")))
        for record in records
        if isinstance(record.get("risk_score", record.get("failure_score")), (int, float))
    ]
    route_values = [
        float(record.get("route_completion"))
        for record in records
        if isinstance(record.get("route_completion"), (int, float))
    ]
    summary = {
        "record_count": len(records),
        "distinct_seed_count": len(set(seeds)),
        "failure_mode_counts": dict(modes),
        "average_risk_score": mean(risk_values) if risk_values else None,
        "average_route_completion": mean(route_values) if route_values else None,
    }
    return failures, warnings, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--min-records", type=int, default=30)
    args = parser.parse_args()

    try:
        records = load_records(args.buffer)
        failures, warnings, summary = summarize(records, args.min_records)
    except Exception as exc:
        print(f"FAIL failure buffer quality: {exc}", file=sys.stderr)
        return 1

    print(f"record_count: {summary['record_count']}")
    print(f"distinct_seed_count: {summary['distinct_seed_count']}")
    print(f"failure_mode_counts: {summary['failure_mode_counts']}")
    print(f"average_risk_score: {summary['average_risk_score']}")
    print(f"average_route_completion: {summary['average_route_completion']}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if failures:
        print("FAIL failure buffer quality:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS failure buffer quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
