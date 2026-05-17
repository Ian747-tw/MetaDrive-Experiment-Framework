from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import FailureLabelOutput, FailureScoreOutput


class DefaultFailureClassifier:
    name = "DefaultFailureClassifier"

    def __init__(self, low_progress_threshold: float = 0.8) -> None:
        self.low_progress_threshold = low_progress_threshold

    def classify(
        self, episode_record: dict[str, Any], score: FailureScoreOutput
    ) -> FailureLabelOutput:
        route_completion = float(episode_record.get("route_completion") or 0.0)
        near_miss = float(episode_record.get("near_miss", 0.0) or 0.0)
        if episode_record.get("collision"):
            mode = "collision"
        elif episode_record.get("offroad"):
            mode = "offroad"
        elif episode_record.get("timeout") and route_completion < self.low_progress_threshold:
            mode = "timeout_or_hesitation"
        elif route_completion < self.low_progress_threshold:
            mode = "low_progress"
        elif near_miss > 0:
            mode = "near_miss"
        elif episode_record.get("success"):
            mode = "solved"
        else:
            mode = "unknown"
        confidence = 0.9 if mode in {"collision", "offroad", "solved"} else 0.7
        return FailureLabelOutput(
            failure_mode=mode,
            confidence=confidence,
            reason={"route_completion": route_completion, "risk_score": score.risk_score},
        )
