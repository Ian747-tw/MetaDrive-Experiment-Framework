from __future__ import annotations

from fasb.plugins.cost import DefaultDrivingCost


class NearMissHeavyCost(DefaultDrivingCost):
    name = "NearMissHeavyCost"

    def __init__(self) -> None:
        super().__init__(
            collision_weight=1.0,
            offroad_weight=1.0,
            near_miss_weight=0.5,
            min_distance_threshold=5.0,
        )
