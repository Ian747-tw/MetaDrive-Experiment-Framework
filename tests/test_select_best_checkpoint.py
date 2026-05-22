from __future__ import annotations

from scripts.select_best_checkpoint import discover_checkpoints, hard_reject, select_best


def test_discover_checkpoints_orders_periodic_then_best_then_final(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("final.zip", "latest_300000_steps.zip", "latest_100000_steps.zip", "best_mean_reward.zip"):
        (checkpoint_dir / name).write_text("x", encoding="utf-8")

    assert [path.name for path in discover_checkpoints(tmp_path)] == [
        "latest_100000_steps.zip",
        "latest_300000_steps.zip",
        "best_mean_reward.zip",
        "final.zip",
    ]


def test_hard_reject_progress_screens() -> None:
    assert hard_reject({"success_rate": 0.19, "route_completion_mean": 0.5, "timeout_rate": 0.5}) == "success_rate < 0.20"
    assert hard_reject({"success_rate": 0.2, "route_completion_mean": 0.39, "timeout_rate": 0.5}) == "route_completion_mean < 0.40"
    assert hard_reject({"success_rate": 0.2, "route_completion_mean": 0.4, "timeout_rate": 0.81}) == "timeout_rate > 0.80"
    assert hard_reject({"success_rate": 0.2, "route_completion_mean": 0.4, "timeout_rate": 0.8}) == ""


def test_select_best_uses_metric_then_route_then_safety_tiebreak() -> None:
    rows = [
        {"checkpoint": "a", "hard_reject": True, "safety_efficiency_score": 1.0, "route_completion_mean": 1.0, "collision_rate": 0.0, "offroad_rate": 0.0},
        {"checkpoint": "b", "hard_reject": False, "safety_efficiency_score": 0.5, "route_completion_mean": 0.6, "collision_rate": 0.2, "offroad_rate": 0.2},
        {"checkpoint": "c", "hard_reject": False, "safety_efficiency_score": 0.5, "route_completion_mean": 0.7, "collision_rate": 0.5, "offroad_rate": 0.5},
    ]
    assert select_best(rows, "safety_efficiency_score")["checkpoint"] == "c"

    rows[1]["route_completion_mean"] = 0.7
    assert select_best(rows, "safety_efficiency_score")["checkpoint"] == "b"


def test_select_best_returns_none_when_all_rejected() -> None:
    assert select_best([{"checkpoint": "a", "hard_reject": True}], "safety_efficiency_score") is None
