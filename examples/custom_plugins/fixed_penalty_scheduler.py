from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import BudgetOutput, FailureLabelOutput, FailureScoreOutput, PenaltyOutput


class FixedPenaltyScheduler:
    name = "FixedPenaltyScheduler"

    def __init__(self, penalty_coef: float = 1.0) -> None:
        self.penalty_coef = penalty_coef

    def get_penalty(
        self,
        scenario_record: dict[str, Any],
        budget: BudgetOutput,
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> PenaltyOutput:
        return PenaltyOutput(
            penalty_coef=float(self.penalty_coef),
            reason={
                "budget": budget.budget,
                "risk_score": score.risk_score,
                "failure_mode": label.failure_mode,
            },
        )
