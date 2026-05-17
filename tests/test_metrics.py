from __future__ import annotations

from fasb.evaluation.metrics import summarize_episode_records


def test_metrics_summary() -> None:
    records = [
        {"success": True, "collision": False, "offroad": False, "timeout": False, "route_completion": 1.0, "reward": 10, "modified_reward": 9, "cost": 0, "episode_length": 10},
        {"success": False, "collision": True, "offroad": False, "timeout": False, "route_completion": 0.4, "reward": 2, "modified_reward": 1, "cost": 1, "episode_length": 20},
    ]
    metrics = summarize_episode_records(records)
    assert metrics["success_rate"] == 0.5
    assert metrics["collision_rate"] == 0.5
    assert metrics["episode_cost_mean"] == 0.5


def test_empty_metrics_stress() -> None:
    for _ in range(100):
        assert summarize_episode_records([])["n_episodes"] == 0
