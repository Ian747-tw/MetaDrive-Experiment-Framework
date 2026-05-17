from __future__ import annotations

import math
from dataclasses import is_dataclass

from fasb.core.errors import ComponentValidationError
from fasb.schemas.outputs import (
    BudgetOutput,
    CostOutput,
    FailureLabelOutput,
    FailureScoreOutput,
    PenaltyOutput,
    ScenarioSample,
)


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ComponentValidationError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ComponentValidationError(f"{name} must be finite, got {value!r}")
    return value


def _dataclass(output: object, cls: type, name: str) -> None:
    if not isinstance(output, cls) or not is_dataclass(output):
        raise ComponentValidationError(f"{name} must be {cls.__name__}, got {type(output).__name__}")


def validate_cost_output(output: CostOutput) -> None:
    _dataclass(output, CostOutput, "cost output")
    cost = _finite(output.cost, "cost")
    if cost < 0:
        raise ComponentValidationError(f"cost must be non-negative, got {cost}")
    if not isinstance(output.components, dict):
        raise ComponentValidationError("cost components must be a dict")
    for key, value in output.components.items():
        if not isinstance(key, str) or not key:
            raise ComponentValidationError("cost component names must be non-empty strings")
        _finite(value, f"cost component {key}")


def validate_failure_score_output(output: FailureScoreOutput) -> None:
    _dataclass(output, FailureScoreOutput, "failure score output")
    score = _finite(output.failure_score, "failure_score")
    risk = _finite(output.risk_score, "risk_score")
    if score < 0:
        raise ComponentValidationError(f"failure_score must be non-negative, got {score}")
    if not 0 <= risk <= 1:
        raise ComponentValidationError(f"risk_score must be in [0, 1], got {risk}")
    if not isinstance(output.reason, dict):
        raise ComponentValidationError("failure score reason must be a dict")


def validate_failure_label_output(output: FailureLabelOutput) -> None:
    _dataclass(output, FailureLabelOutput, "failure label output")
    if not isinstance(output.failure_mode, str) or not output.failure_mode:
        raise ComponentValidationError("failure_mode must be a non-empty string")
    confidence = _finite(output.confidence, "confidence")
    if not 0 <= confidence <= 1:
        raise ComponentValidationError(f"confidence must be in [0, 1], got {confidence}")
    if not isinstance(output.reason, dict):
        raise ComponentValidationError("failure label reason must be a dict")


def validate_budget_output(
    output: BudgetOutput, d_min: float | None = None, d_max: float | None = None
) -> None:
    _dataclass(output, BudgetOutput, "budget output")
    budget = _finite(output.budget, "budget")
    if budget <= 0:
        raise ComponentValidationError(f"budget must be positive, got {budget}")
    if d_min is not None and budget < d_min:
        raise ComponentValidationError(f"budget must be >= {d_min}, got {budget}")
    if d_max is not None and budget > d_max:
        raise ComponentValidationError(f"budget must be <= {d_max}, got {budget}")
    if not isinstance(output.mode, str) or not output.mode:
        raise ComponentValidationError("budget mode must be a non-empty string")
    if not isinstance(output.reason, dict):
        raise ComponentValidationError("budget reason must be a dict")


def validate_penalty_output(output: PenaltyOutput) -> None:
    _dataclass(output, PenaltyOutput, "penalty output")
    penalty = _finite(output.penalty_coef, "penalty_coef")
    if penalty < 0:
        raise ComponentValidationError(f"penalty_coef must be non-negative, got {penalty}")
    if not isinstance(output.reason, dict):
        raise ComponentValidationError("penalty reason must be a dict")


def validate_scenario_sample(output: ScenarioSample) -> None:
    _dataclass(output, ScenarioSample, "scenario sample")
    if not isinstance(output.seed, int) or isinstance(output.seed, bool):
        raise ComponentValidationError(f"scenario seed must be int, got {output.seed!r}")
    if not isinstance(output.source, str) or not output.source:
        raise ComponentValidationError("sampler source must be a non-empty string")
    priority = _finite(output.priority, "priority")
    if priority < 0:
        raise ComponentValidationError(f"priority must be non-negative, got {priority}")
    if not isinstance(output.metadata, dict):
        raise ComponentValidationError("scenario metadata must be a dict")
