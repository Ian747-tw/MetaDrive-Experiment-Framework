from __future__ import annotations

from typing import Any

from fasb.schemas.outputs import BudgetOutput, FailureLabelOutput, FailureScoreOutput


class TimeoutRelaxedBudget:
    name = "TimeoutRelaxedBudget"

    def __init__(self, d_min: float = 0.02, d_max: float = 0.10, timeout_budget: float = 0.08) -> None:
        self.d_min = d_min
        self.d_max = d_max
        self.timeout_budget = timeout_budget

    def get_budget(
        self,
        scenario_record: dict[str, Any],
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> BudgetOutput:
        if label.failure_mode in {"timeout_or_hesitation", "low_progress"}:
            budget = self.timeout_budget
            mode = "relaxed_progress"
        elif label.failure_mode in {"collision", "offroad"}:
            budget = self.d_min
            mode = "strict_safety"
        else:
            budget = self.d_max - (self.d_max - self.d_min) * score.risk_score
            mode = "risk_adaptive"
        budget = min(max(float(budget), self.d_min), self.d_max)
        return BudgetOutput(
            budget=budget,
            mode=mode,
            reason={"failure_mode": label.failure_mode, "risk_score": score.risk_score},
        )
