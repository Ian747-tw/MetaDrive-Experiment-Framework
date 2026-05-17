from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import BudgetOutput, FailureLabelOutput, FailureScoreOutput, PenaltyOutput


class RiskPenaltyScheduler:
    name = "RiskPenaltyScheduler"

    def __init__(
        self,
        lambda_min: float = 0.1,
        lambda_max: float = 5.0,
        mode_multipliers: dict[str, float] | None = None,
    ) -> None:
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.mode_multipliers = mode_multipliers or {
            "collision": 1.2,
            "offroad": 1.1,
            "timeout_or_hesitation": 0.7,
            "low_progress": 0.6,
        }

    def get_penalty(
        self,
        scenario_record: dict[str, Any],
        budget: BudgetOutput,
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> PenaltyOutput:
        base = self.lambda_min + (self.lambda_max - self.lambda_min) * score.risk_score
        multiplier = float(self.mode_multipliers.get(label.failure_mode, 1.0))
        penalty = max(base * multiplier, 0.0)
        return PenaltyOutput(penalty, {"base": base, "multiplier": multiplier, "budget": budget.budget})
