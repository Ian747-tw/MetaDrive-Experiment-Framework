from __future__ import annotations

import pytest

from scripts.check_base_checkpoint_quality import check_quality, read_metrics


def test_base_checkpoint_quality_passes(tmp_path) -> None:
    path = tmp_path / "heldout_random.csv"
    path.write_text(
        "success_rate,route_completion_mean,timeout_rate\n"
        "0.2,0.5,0.8\n",
        encoding="utf-8",
    )
    metrics = read_metrics(path)
    assert check_quality(metrics, 0.1, 0.35, 0.95) == []


def test_base_checkpoint_quality_fails_thresholds(tmp_path) -> None:
    path = tmp_path / "heldout_random.csv"
    path.write_text(
        "success_rate,route_completion_mean,timeout_rate\n"
        "0.0,0.2,1.0\n",
        encoding="utf-8",
    )
    failures = check_quality(read_metrics(path), 0.1, 0.35, 0.95)
    assert len(failures) == 3


def test_base_checkpoint_quality_requires_columns(tmp_path) -> None:
    path = tmp_path / "heldout_random.csv"
    path.write_text("success_rate\n0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required metrics"):
        read_metrics(path)
