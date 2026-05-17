from __future__ import annotations

from typing import Any, Protocol

from fasb.schemas.outputs import (
    BudgetOutput,
    CostOutput,
    FailureLabelOutput,
    FailureScoreOutput,
    PenaltyOutput,
    ScenarioSample,
)


class CostFunction(Protocol):
    name: str

    def __call__(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> CostOutput: ...


class FailureScorer(Protocol):
    name: str

    def score(self, episode_record: dict[str, Any]) -> FailureScoreOutput: ...


class FailureClassifier(Protocol):
    name: str

    def classify(
        self, episode_record: dict[str, Any], score: FailureScoreOutput
    ) -> FailureLabelOutput: ...


class SafetyBudgeter(Protocol):
    name: str

    def get_budget(
        self,
        scenario_record: dict[str, Any],
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> BudgetOutput: ...


class PenaltyScheduler(Protocol):
    name: str

    def get_penalty(
        self,
        scenario_record: dict[str, Any],
        budget: BudgetOutput,
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> PenaltyOutput: ...


class ScenarioSampler(Protocol):
    name: str

    def next(self) -> ScenarioSample: ...
