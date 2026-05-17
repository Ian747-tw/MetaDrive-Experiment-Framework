from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostOutput:
    cost: float
    components: dict[str, float]


@dataclass
class FailureScoreOutput:
    failure_score: float
    risk_score: float
    reason: dict[str, float] = field(default_factory=dict)


@dataclass
class FailureLabelOutput:
    failure_mode: str
    confidence: float
    reason: dict[str, float | str] = field(default_factory=dict)


@dataclass
class BudgetOutput:
    budget: float
    mode: str
    reason: dict[str, float | str] = field(default_factory=dict)


@dataclass
class PenaltyOutput:
    penalty_coef: float
    reason: dict[str, float | str] = field(default_factory=dict)


@dataclass
class ScenarioSample:
    seed: int
    source: str
    priority: float
    metadata: dict[str, Any] = field(default_factory=dict)
