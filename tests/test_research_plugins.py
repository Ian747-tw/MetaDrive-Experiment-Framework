from __future__ import annotations

from examples.custom_plugins.crash_only_cost import CrashOnlyCost
from examples.custom_plugins.first_seed_sampler import FirstSeedSampler
from examples.custom_plugins.fixed_penalty_scheduler import FixedPenaltyScheduler
from examples.custom_plugins.near_failure_scorer import NearFailureScorer
from examples.custom_plugins.near_miss_heavy_cost import NearMissHeavyCost
from examples.custom_plugins.timeout_relaxed_budget import TimeoutRelaxedBudget
from fasb.core.validation import (
    validate_budget_output,
    validate_cost_output,
    validate_failure_score_output,
    validate_penalty_output,
    validate_scenario_sample,
)
from fasb.schemas.outputs import FailureLabelOutput


def test_research_cost_plugins_validate() -> None:
    info = {"crash": True, "out_of_road": True, "min_vehicle_distance": 2.5}
    for plugin in (CrashOnlyCost(), NearMissHeavyCost()):
        output = plugin(None, None, 0.0, None, False, False, info)
        validate_cost_output(output)
        assert output.cost >= 0.0


def test_near_failure_scorer_validates_and_clamps_risk() -> None:
    output = NearFailureScorer().score(
        {
            "seed": 7,
            "success": False,
            "collision": False,
            "offroad": False,
            "timeout": True,
            "route_completion": 0.4,
            "min_vehicle_distance": 1.0,
        }
    )
    validate_failure_score_output(output)
    assert 0.0 <= output.risk_score <= 1.0


def test_timeout_relaxed_budget_validates_modes() -> None:
    scorer = NearFailureScorer()
    score = scorer.score({"route_completion": 0.2, "timeout": True, "success": False})
    budgeter = TimeoutRelaxedBudget()
    label = FailureLabelOutput("timeout_or_hesitation", 0.8)
    output = budgeter.get_budget({"seed": 1}, score, label)
    validate_budget_output(output)
    assert output.mode == "relaxed_progress"

    strict = budgeter.get_budget({"seed": 1}, score, FailureLabelOutput("collision", 0.9))
    validate_budget_output(strict)
    assert strict.mode == "strict_safety"


def test_fixed_penalty_scheduler_validates() -> None:
    score = NearFailureScorer().score({"collision": True, "route_completion": 0.2, "success": False})
    label = FailureLabelOutput("collision", 0.9)
    budget = TimeoutRelaxedBudget().get_budget({"seed": 1}, score, label)
    output = FixedPenaltyScheduler(penalty_coef=2.0).get_penalty({"seed": 1}, budget, score, label)
    validate_penalty_output(output)
    assert output.penalty_coef == 2.0


def test_first_seed_sampler_validates() -> None:
    sample = FirstSeedSampler(start_seed=1234).next()
    validate_scenario_sample(sample)
    assert sample.seed == 1234
