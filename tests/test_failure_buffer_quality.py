from __future__ import annotations

import json

from scripts.check_failure_buffer_quality import load_records, summarize


def write_jsonl(path, records) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_failure_buffer_quality_passes_with_diverse_records(tmp_path) -> None:
    path = tmp_path / "failure_buffer.jsonl"
    records = [
        {
            "seed": i,
            "failure_mode": "collision" if i % 2 else "low_progress",
            "risk_score": 0.5,
            "route_completion": 0.4,
            "success": False,
        }
        for i in range(30)
    ]
    write_jsonl(path, records)
    failures, warnings, summary = summarize(load_records(path), min_records=30)
    assert failures == []
    assert warnings == []
    assert summary["record_count"] == 30
    assert summary["distinct_seed_count"] == 30


def test_failure_buffer_quality_hard_failures_and_warnings(tmp_path) -> None:
    path = tmp_path / "failure_buffer.jsonl"
    write_jsonl(
        path,
        [
            {"seed": 1, "failure_mode": "solved", "risk_score": 0.0, "route_completion": 1.0, "success": True},
            {"seed": 1, "failure_mode": "solved", "risk_score": 0.0, "route_completion": 1.0, "success": True},
        ],
    )
    failures, warnings, summary = summarize(load_records(path), min_records=30)
    assert failures == ["record count 2 < 30"]
    assert "only one failure mode present" in warnings
    assert "all records are solved" in warnings
    assert "all records share the same seed" in warnings
    assert summary["average_risk_score"] == 0.0
