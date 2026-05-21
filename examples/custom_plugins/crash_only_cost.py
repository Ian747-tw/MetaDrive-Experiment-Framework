from __future__ import annotations

from typing import Any

from fasb.plugins.cost import truthy_info
from fasb.schemas.outputs import CostOutput


class CrashOnlyCost:
    name = "CrashOnlyCost"

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
        return CostOutput(
            cost=float(collision + offroad),
            components={"collision": collision, "offroad": offroad},
        )
