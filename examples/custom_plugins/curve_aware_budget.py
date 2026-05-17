from __future__ import annotations

from fasb.schemas.outputs import BudgetOutput


class CurveAwareBudget:
    name = "CurveAwareBudget"

    def __init__(self, base_budget: float = 0.06) -> None:
        self.base_budget = base_budget

    def get_budget(self, scenario_record, score, label):
        curve = float(scenario_record.get("metadata", {}).get("curve_severity", 0.0) or 0.0)
        budget = max(self.base_budget - 0.02 * curve - 0.02 * score.risk_score, 0.01)
        return BudgetOutput(budget, "curve_aware", {"curve_severity": curve})
