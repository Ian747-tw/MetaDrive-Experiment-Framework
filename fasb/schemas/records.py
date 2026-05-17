from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EpisodeRecord:
    run_id: str
    episode_id: int
    seed: int | None
    scenario_id: str | None
    reward: float
    modified_reward: float | None
    cost: float
    success: bool
    collision: bool
    offroad: bool
    timeout: bool
    route_completion: float
    episode_length: int
    traffic_density: float | None = None
    failure_mode: str | None = None
    failure_score: float | None = None
    risk_score: float | None = None
    safety_budget: float | None = None
    penalty_coef: float | None = None
    min_vehicle_distance: float | None = None
    hard_brake_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
