from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None

from fasb.core.validation import (
    validate_budget_output,
    validate_cost_output,
    validate_failure_label_output,
    validate_failure_score_output,
    validate_penalty_output,
)
from fasb.core.plugin_runtime import safe_call_component
from fasb.failure.record_utils import build_training_scenario_record
from fasb.plugins.failure_classifier import DefaultFailureClassifier
from fasb.plugins.failure_scorer import DefaultFailureScorer
from fasb.plugins.penalty_scheduler import RiskPenaltyScheduler
from fasb.plugins.safety_budget import AdaptiveSafetyBudget
class _FallbackWrapper:
    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


BaseWrapper = gym.Wrapper if gym is not None else _FallbackWrapper


def _wrapper_init(self: Any, env: Any) -> None:
    if gym is not None:
        gym.Wrapper.__init__(self, env)
    else:  # pragma: no cover
        _FallbackWrapper.__init__(self, env)


class ScenarioMetadataWrapper(BaseWrapper):
    def __init__(self, env: Any, scenario_sampler: Any | None = None, run_context: dict[str, Any] | None = None) -> None:
        _wrapper_init(self, env)
        self.scenario_sampler = scenario_sampler
        self.run_context = run_context or {}
        self.current_sample = None
        self.episode_id = -1

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.episode_id += 1
        if self.scenario_sampler is not None:
            sample = self.scenario_sampler.next()
            self.current_sample = sample
            kwargs.setdefault("seed", sample.seed)
            if hasattr(self.env, "config") and isinstance(getattr(self.env, "config"), dict):
                self.env.config["start_seed"] = sample.seed
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) and len(result) == 2 else (result, {})
        info = dict(info)
        info["fasb_scenario"] = self._scenario_metadata(info)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
        info = dict(info)
        info["fasb_scenario"] = self._scenario_metadata(info)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _scenario_metadata(self, info: dict[str, Any]) -> dict[str, Any]:
        seed = None
        source = "env"
        priority = 1.0
        metadata = {}
        if self.current_sample is not None:
            seed = self.current_sample.seed
            source = self.current_sample.source
            priority = self.current_sample.priority
            metadata = dict(self.current_sample.metadata)
        seed = seed if seed is not None else info.get("seed", info.get("scenario_seed"))
        scenario_id = info.get("scenario_id") or (f"seed_{seed}" if seed is not None else None)
        return {
            "episode_id": self.episode_id,
            "seed": seed,
            "scenario_id": scenario_id,
            "source": source,
            "priority": priority,
            "traffic_density": self.run_context.get("traffic_density"),
            "metadata": metadata,
        }


class CostFunctionWrapper(BaseWrapper):
    def __init__(
        self,
        env: Any,
        cost_function: Any,
        error_dir: str | Any | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        _wrapper_init(self, env)
        self.cost_function = cost_function
        self.error_dir = error_dir
        self.run_context = run_context or {}
        self.last_obs = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        output = safe_call_component(
            self.cost_function,
            None,
            "cost_function",
            self.run_context,
            self.error_dir,
            self.last_obs,
            action,
            reward,
            obs,
            terminated,
            truncated,
            info,
        )
        validate_cost_output(output)
        info = dict(info)
        info["fasb_cost"] = output.cost
        info["fasb_cost_components"] = output.components
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


class AdaptiveRewardPenaltyWrapper(BaseWrapper):
    def __init__(
        self,
        env: Any,
        failure_scorer: Any | None = None,
        failure_classifier: Any | None = None,
        safety_budget: Any | None = None,
        penalty_scheduler: Any | None = None,
        error_dir: str | Any | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        _wrapper_init(self, env)
        self.failure_scorer = failure_scorer or DefaultFailureScorer()
        self.failure_classifier = failure_classifier or DefaultFailureClassifier()
        self.safety_budget = safety_budget or AdaptiveSafetyBudget()
        self.penalty_scheduler = penalty_scheduler or RiskPenaltyScheduler()
        self.error_dir = error_dir
        self.run_context = run_context or {}
        self.current_budget = None
        self.current_label = None
        self.current_score = None
        self.current_penalty = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        record = build_training_scenario_record(info)
        score = safe_call_component(
            self.failure_scorer, "score", "failure_scorer", self.run_context, self.error_dir, record
        )
        validate_failure_score_output(score)
        label = safe_call_component(
            self.failure_classifier,
            "classify",
            "failure_classifier",
            self.run_context,
            self.error_dir,
            record,
            score,
        )
        validate_failure_label_output(label)
        budget = safe_call_component(
            self.safety_budget,
            "get_budget",
            "safety_budget",
            self.run_context,
            self.error_dir,
            record,
            score,
            label,
        )
        validate_budget_output(budget)
        penalty = safe_call_component(
            self.penalty_scheduler,
            "get_penalty",
            "penalty_scheduler",
            self.run_context,
            self.error_dir,
            record,
            budget,
            score,
            label,
        )
        validate_penalty_output(penalty)
        self.current_score = score
        self.current_label = label
        self.current_budget = budget
        self.current_penalty = penalty
        info = self._annotate_info(info)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(info.get("fasb_cost", 0.0) or 0.0)
        penalty_coef = float(self.current_penalty.penalty_coef if self.current_penalty else 0.0)
        modified_reward = float(reward) - penalty_coef * cost
        info = dict(info)
        info["fasb_original_reward"] = float(reward)
        info["fasb_modified_reward"] = modified_reward
        info = self._annotate_info(info)
        return obs, modified_reward, terminated, truncated, info

    def _annotate_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info = dict(info)
        if self.current_budget is not None:
            info["fasb_budget"] = self.current_budget.budget
            info["fasb_budget_mode"] = self.current_budget.mode
            info["fasb_budget_reason"] = self.current_budget.reason
        if self.current_penalty is not None:
            info["fasb_penalty_coef"] = self.current_penalty.penalty_coef
            info["fasb_penalty_reason"] = self.current_penalty.reason
        if self.current_label is not None:
            info["fasb_failure_mode"] = self.current_label.failure_mode
        if self.current_score is not None:
            info["fasb_risk_score"] = self.current_score.risk_score
            info["fasb_failure_score"] = self.current_score.failure_score
        return info


class RandomPolicy:
    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, None]:
        return self.action_space.sample(), None

    def __call__(self, obs: Any) -> Any:
        if hasattr(self.action_space, "sample"):
            return self.action_space.sample()
        return np.zeros(1, dtype=np.float32)
