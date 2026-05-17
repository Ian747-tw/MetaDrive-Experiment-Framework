from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import FailureScoreOutput


class DefaultFailureScorer:
    name = "DefaultFailureScorer"

    def score(self, episode_record: dict[str, Any]) -> FailureScoreOutput:
        collision = float(bool(episode_record.get("collision", False)))
        offroad = float(bool(episode_record.get("offroad", False)))
        timeout = float(bool(episode_record.get("timeout", False)))
        route_completion = float(episode_record.get("route_completion") or 0.0)
        near_miss = float(episode_record.get("near_miss", episode_record.get("min_vehicle_distance_risk", 0.0)) or 0.0)
        raw = 5.0 * collision + 4.0 * offroad + 2.0 * timeout + 1.0 * (1.0 - route_completion) + 2.0 * near_miss
        risk = min(max(raw / 8.0, 0.0), 1.0)
        return FailureScoreOutput(
            failure_score=float(raw),
            risk_score=float(risk),
            reason={
                "collision": collision,
                "offroad": offroad,
                "timeout": timeout,
                "route_completion": route_completion,
                "near_miss": near_miss,
            },
        )
