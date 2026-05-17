from __future__ import annotations

from fasb.schemas.outputs import CostOutput


class CautiousDistanceCost:
    name = "CautiousDistanceCost"

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold

    def __call__(self, obs, action, reward, next_obs, terminated, truncated, info):
        distance = info.get("min_vehicle_distance")
        near = 0.0
        if isinstance(distance, (int, float)) and 0 <= distance < self.threshold:
            near = (self.threshold - float(distance)) / self.threshold
        crash = float(bool(info.get("crash") or info.get("collision")))
        return CostOutput(crash + 0.25 * near, {"crash": crash, "near_miss": near})
