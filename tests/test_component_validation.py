from __future__ import annotations

import pytest

from fasb.core.errors import ComponentValidationError
from fasb.core.validation import (
    validate_budget_output,
    validate_cost_output,
    validate_failure_label_output,
    validate_failure_score_output,
    validate_penalty_output,
    validate_scenario_sample,
)
from fasb.schemas.outputs import (
    BudgetOutput,
    CostOutput,
    FailureLabelOutput,
    FailureScoreOutput,
    PenaltyOutput,
    ScenarioSample,
)


def test_valid_outputs_pass() -> None:
    validate_cost_output(CostOutput(0.0, {"collision": 0.0}))
    validate_failure_score_output(FailureScoreOutput(1.0, 0.2, {}))
    validate_failure_label_output(FailureLabelOutput("collision", 0.9, {}))
    validate_budget_output(BudgetOutput(0.05, "adaptive", {}), 0.01, 0.1)
    validate_penalty_output(PenaltyOutput(1.0, {}))
    validate_scenario_sample(ScenarioSample(1, "random", 1.0, {}))


@pytest.mark.parametrize(
    "output,validator",
    [
        (CostOutput(-1.0, {}), validate_cost_output),
        (FailureScoreOutput(1.0, 2.0, {}), validate_failure_score_output),
        (FailureLabelOutput("", 0.5, {}), validate_failure_label_output),
        (BudgetOutput(0.0, "bad", {}), validate_budget_output),
        (PenaltyOutput(-0.1, {}), validate_penalty_output),
        (ScenarioSample(True, "bad", 1.0, {}), validate_scenario_sample),
    ],
)
def test_invalid_outputs_fail(output, validator) -> None:
    with pytest.raises(ComponentValidationError):
        validator(output)


def test_validation_stress_many_budget_bounds() -> None:
    for i in range(1000):
        value = 0.01 + i * 0.00001
        validate_budget_output(BudgetOutput(value, "stress", {}), 0.01, 0.03)
