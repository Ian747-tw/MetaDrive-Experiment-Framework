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
    failures, warnings, summary = summarize(load_records(path), min_records=30, require_multiple_modes=True)
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
    assert "record count 2 < 30" in failures
    assert "all records are solved" in failures
    assert "all records share the same seed" in failures
    assert "only one failure mode present" in warnings
    assert summary["average_risk_score"] == 0.0
    assert summary["solved_count"] == 2
    assert summary["unknown_count"] == 0


def test_failure_buffer_quality_fails_on_high_unknown_fraction(tmp_path) -> None:
    path = tmp_path / "failure_buffer.jsonl"
    records = [
        {
            "seed": i,
            "failure_mode": "unknown" if i < 20 else "collision",
            "risk_score": 0.5,
            "route_completion": 0.4,
            "success": False,
        }
        for i in range(30)
    ]
    write_jsonl(path, records)
    failures, warnings, summary = summarize(load_records(path), min_records=30, max_unknown_fraction=0.25)
    assert any("unknown failure mode fraction" in failure for failure in failures)
    assert warnings == []
    assert summary["unknown_count"] == 20


def test_failure_buffer_quality_counts_missing_and_null_modes_as_unknown(tmp_path) -> None:
    path = tmp_path / "failure_buffer.jsonl"
    write_jsonl(
        path,
        [
            {"seed": 1, "risk_score": 0.5, "route_completion": 0.4, "success": False},
            {"seed": 2, "failure_mode": None, "risk_score": 0.5, "route_completion": 0.4, "success": False},
            {"seed": 3, "failure_mode": "", "risk_score": 0.5, "route_completion": 0.4, "success": False},
            {"seed": 4, "failure_mode": "collision", "risk_score": 0.5, "route_completion": 0.4, "success": False},
        ],
    )
    failures, warnings, summary = summarize(load_records(path), min_records=4, max_unknown_fraction=0.25)
    assert any("unknown failure mode fraction" in failure for failure in failures)
    assert warnings == []
    assert summary["failure_mode_counts"]["unknown"] == 3
    assert summary["unknown_count"] == 3
