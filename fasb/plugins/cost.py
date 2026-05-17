from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import CostOutput


def truthy_info(info: dict[str, Any], *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)


class DefaultDrivingCost:
    name = "DefaultDrivingCost"

    def __init__(
        self,
        collision_weight: float = 1.0,
        offroad_weight: float = 1.0,
        near_miss_weight: float = 0.1,
        min_distance_threshold: float = 3.0,
    ) -> None:
        self.collision_weight = collision_weight
        self.offroad_weight = offroad_weight
        self.near_miss_weight = near_miss_weight
        self.min_distance_threshold = min_distance_threshold

    def __call__(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> CostOutput:
        collision = float(
            truthy_info(info, "crash", "crash_vehicle", "collision", "collided", "vehicle_collision")
        )
        offroad = float(truthy_info(info, "out_of_road", "offroad", "crash_sidewalk", "crash_offroad"))
        if "on_lane" in info and info.get("on_lane") is False:
            offroad = 1.0
        if truthy_info(info, "arrive_dest", "success"):
            offroad = 0.0
        min_distance = info.get("min_vehicle_distance", info.get("min_distance"))
        near_miss = 0.0
        if isinstance(min_distance, (int, float)) and 0 <= float(min_distance) < self.min_distance_threshold:
            near_miss = (self.min_distance_threshold - float(min_distance)) / self.min_distance_threshold
        cost = (
            self.collision_weight * collision
            + self.offroad_weight * offroad
            + self.near_miss_weight * near_miss
        )
        return CostOutput(
            cost=float(cost),
            components={"collision": collision, "offroad": offroad, "near_miss": near_miss},
        )
