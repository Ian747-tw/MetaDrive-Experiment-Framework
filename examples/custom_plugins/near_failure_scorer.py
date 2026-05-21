from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import FailureScoreOutput


class NearFailureScorer:
    name = "NearFailureScorer"

    def score(self, episode_record: dict[str, Any]) -> FailureScoreOutput:
        collision = float(bool(episode_record.get("collision", False)))
        offroad = float(bool(episode_record.get("offroad", False)))
        timeout = float(bool(episode_record.get("timeout", False)))
        route_completion = float(episode_record.get("route_completion") or 0.0)
        near_miss = _near_miss_signal(episode_record)
        partial_progress_failure = (1.0 - route_completion) * float(not bool(episode_record.get("success", False)))
        raw = (
            2.0 * partial_progress_failure
            + 3.0 * near_miss
            + 1.5 * timeout
            + 1.0 * collision
            + 1.0 * offroad
        )
        risk = min(max(raw / 6.0, 0.0), 1.0)
        return FailureScoreOutput(
            failure_score=float(raw),
            risk_score=float(risk),
            reason={
                "partial_progress_failure": partial_progress_failure,
                "near_miss": near_miss,
                "timeout": timeout,
                "collision": collision,
                "offroad": offroad,
                "route_completion": route_completion,
            },
        )


def _near_miss_signal(record: dict[str, Any]) -> float:
    direct = record.get("near_miss", record.get("min_vehicle_distance_risk"))
    if isinstance(direct, (int, float)):
        return min(max(float(direct), 0.0), 1.0)
    distance = record.get("min_vehicle_distance")
    if isinstance(distance, (int, float)) and 0 <= float(distance) < 5.0:
        return (5.0 - float(distance)) / 5.0
    return 0.0
