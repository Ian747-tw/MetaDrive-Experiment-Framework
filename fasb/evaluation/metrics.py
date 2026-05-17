from __future__ import annotations

from typing import Any

import numpy as np


def summarize_episode_records(
    records: list[dict[str, Any]],
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.5,
) -> dict[str, float]:
    if not records:
        return {
            "n_episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "offroad_rate": 0.0,
            "timeout_rate": 0.0,
            "route_completion_mean": 0.0,
            "episode_reward_mean": 0.0,
            "episode_modified_reward_mean": 0.0,
            "episode_cost_mean": 0.0,
            "cost_violation_rate": 0.0,
            "avg_episode_length": 0.0,
            "safety_efficiency_score": 0.0,
        }

    def mean(key: str) -> float:
        return float(np.mean([float(r.get(key, 0.0) or 0.0) for r in records]))

    success = mean("success")
    collision = mean("collision")
    offroad = mean("offroad")
    timeout = mean("timeout")
    modified_rewards = [
        float(r.get("modified_reward") if r.get("modified_reward") is not None else r.get("reward", 0.0))
        for r in records
    ]
    return {
        "n_episodes": float(len(records)),
        "success_rate": success,
        "collision_rate": collision,
        "offroad_rate": offroad,
        "timeout_rate": timeout,
        "route_completion_mean": mean("route_completion"),
        "episode_reward_mean": mean("reward"),
        "episode_modified_reward_mean": float(np.mean(modified_rewards)),
        "episode_cost_mean": mean("cost"),
        "cost_violation_rate": float(np.mean([float(r.get("cost", 0.0) or 0.0) > float(r.get("safety_budget", 1e9) or 1e9) for r in records])),
        "avg_episode_length": mean("episode_length"),
        "safety_efficiency_score": success - beta * collision - gamma * offroad - delta * timeout,
    }
