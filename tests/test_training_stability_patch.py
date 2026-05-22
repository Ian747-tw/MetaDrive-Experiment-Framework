from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from fasb.buffers.failure_buffer import FailureBuffer
from fasb.core.plugin_runtime import safe_call_component
from fasb.envs.wrappers import CostFunctionWrapper
from fasb.failure.record_utils import build_training_scenario_record
from fasb.plugins.failure_classifier import DefaultFailureClassifier
from fasb.plugins.failure_scorer import DefaultFailureScorer
from fasb.plugins.safety_budget import AdaptiveSafetyBudget
from fasb.plugins.sampler import MixedFailureSampler, UniformSampler
from fasb.envs.wrappers import ScenarioMetadataWrapper
from fasb.training.callbacks import build_episode_log_record
from fasb.training.sb3_trainer import SB3Trainer

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None


def test_training_scenario_record_exposes_metadata_failure_fields() -> None:
    record = build_training_scenario_record(
        {
            "fasb_scenario": {
                "seed": 7,
                "scenario_id": "seed_7",
                "source": "failure_buffer",
                "priority": 0.8,
                "metadata": {
                    "collision": True,
                    "route_completion": 0.2,
                    "risk_score": 0.9,
                    "failure_mode": "collision",
                    "source": "historical",
                },
            }
        }
    )
    assert record["collision"] is True
    assert record["route_completion"] == 0.2
    assert record["risk_score"] == 0.9
    assert record["failure_mode"] == "collision"
    assert record["source"] == "historical"
    assert record["scenario_source"] == "failure_buffer"


BaseDummySeedEnv = gym.Env if gym is not None else object


class DummySeedEnv(BaseDummySeedEnv):
    def reset(self, **kwargs):
        return "obs", {}

    def step(self, action):
        return "obs", 0.0, True, False, {}


def test_scenario_metadata_wrapper_preserves_reset_seed_without_sampler() -> None:
    env = ScenarioMetadataWrapper(DummySeedEnv())
    _, info = env.reset(seed=1001)
    record = build_training_scenario_record(info)
    assert record["seed"] == 1001
    assert record["scenario_id"] == "seed_1001"


def test_training_scenario_record_handles_missing_and_malformed_metadata() -> None:
    missing = build_training_scenario_record({"fasb_scenario": {"seed": 3, "scenario_id": "seed_3", "source": "random"}})
    malformed = build_training_scenario_record({"fasb_scenario": {"seed": 4, "metadata": ["bad"]}})
    assert missing["seed"] == 3
    assert missing["scenario_id"] == "seed_3"
    assert missing["source"] == "random"
    assert malformed["seed"] == 4
    assert malformed["metadata"] == {}


def test_failure_metadata_drives_risk_and_adaptive_budget() -> None:
    scorer = DefaultFailureScorer()
    classifier = DefaultFailureClassifier()
    budgeter = AdaptiveSafetyBudget(d_min=0.02, d_max=0.10)
    hard = build_training_scenario_record(
        {
            "fasb_scenario": {
                "seed": 1,
                "metadata": {"collision": True, "route_completion": 0.2, "success": False},
            }
        }
    )
    easy = build_training_scenario_record(
        {
            "fasb_scenario": {
                "seed": 2,
                "metadata": {"collision": False, "offroad": False, "route_completion": 1.0, "success": True},
            }
        }
    )
    hard_score = scorer.score(hard)
    easy_score = scorer.score(easy)
    hard_label = classifier.classify(hard, hard_score)
    easy_label = classifier.classify(easy, easy_score)
    assert hard_score.risk_score > easy_score.risk_score
    assert budgeter.get_budget(hard, hard_score, hard_label).budget < budgeter.get_budget(easy, easy_score, easy_label).budget


def test_sampler_uses_config_target_and_injects_failure_buffer_path(tmp_path) -> None:
    failure_path = tmp_path / "failure_buffer.jsonl"
    failure_path.write_text('{"seed": 11, "risk_score": 1.0, "failure_mode": "collision"}\n', encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "test", "mode": "fasb_ppo", "output_dir": str(tmp_path)},
            "mode": "fasb_ppo",
            "metadrive": {"config": {"start_seed": 100, "num_scenarios": 10}},
            "vec_env": {"n_envs": 2},
            "failure_buffer": {"path": str(failure_path)},
            "sampler": {"_target_": "fasb.plugins.sampler.MixedFailureSampler", "failure_ratio": 0.5},
        }
    )
    sampler = SB3Trainer(cfg)._build_sampler(0)
    assert isinstance(sampler, MixedFailureSampler)
    assert len(sampler.failure_buffer) == 1


def test_mixed_failure_sampler_only_emits_in_range_failure_seeds() -> None:
    buffer = FailureBuffer(
        [
            {"seed": 11, "risk_score": 1.0, "failure_mode": "collision"},
            {"seed": 105, "risk_score": 1.0, "failure_mode": "offroad"},
        ]
    )
    sampler = MixedFailureSampler(
        failure_buffer=buffer,
        start_seed=100,
        num_scenarios=10,
        failure_ratio=1.0,
    )

    for _ in range(20):
        sample = sampler.next()
        assert sample.seed == 105
        assert sample.source == "failure_buffer"


def test_mixed_failure_sampler_falls_back_when_buffer_has_no_in_range_seeds() -> None:
    buffer = FailureBuffer([{"seed": 11, "risk_score": 1.0, "failure_mode": "collision"}])
    sampler = MixedFailureSampler(
        failure_buffer=buffer,
        start_seed=100,
        num_scenarios=10,
        failure_ratio=1.0,
    )

    for _ in range(20):
        sample = sampler.next()
        assert 100 <= sample.seed < 110
        assert sample.source == "random"


def test_uniform_sampler_config_target_and_sharding(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "test", "mode": "naive_ft", "output_dir": str(tmp_path)},
            "mode": "naive_ft",
            "metadrive": {"config": {"start_seed": 100, "num_scenarios": 10}},
            "vec_env": {"n_envs": 3},
            "failure_buffer": {"path": str(tmp_path / "unused.jsonl")},
            "sampler": {"_target_": "fasb.plugins.sampler.UniformSampler"},
        }
    )
    trainer = SB3Trainer(cfg)
    sampler = trainer._build_sampler(2)
    assert isinstance(sampler, UniformSampler)
    assert 100 <= sampler.start_seed < 110
    assert sampler.start_seed + sampler.num_scenarios <= 110


class RaisingPlugin:
    name = "RaisingPlugin"

    def score(self, record):
        raise RuntimeError("plugin exploded")


def test_safe_call_component_logs_and_reraises(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        safe_call_component(RaisingPlugin(), "score", "failure_scorer", {"experiment": "x"}, tmp_path, {})
    jsonl = tmp_path / "plugin_errors.jsonl"
    log = tmp_path / "plugin_errors.log"
    assert jsonl.exists()
    assert log.exists()
    record = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert record["component_type"] == "failure_scorer"
    assert record["error_type"] == "RuntimeError"
    assert record["message"] == "plugin exploded"


def test_build_episode_log_record_rich_fields_and_missing_optional_keys() -> None:
    record = build_episode_log_record(
        {
            "fasb_scenario": {
                "episode_id": 1,
                "seed": 5,
                "scenario_id": "seed_5",
                "source": "failure_buffer",
                "priority": 0.7,
            },
            "success": False,
            "collision": True,
            "route_completion": 0.25,
            "fasb_cost": 2.0,
            "fasb_budget": 0.03,
            "fasb_budget_mode": "risk_adaptive",
            "fasb_penalty_coef": 4.2,
            "fasb_risk_score": 0.8,
            "fasb_failure_score": 6.0,
            "fasb_failure_mode": "collision",
            "fasb_original_reward": 1.0,
            "fasb_modified_reward": -1.0,
            "fasb_cost_components": {"collision": 1.0},
            "episode": {"r": 3.0, "l": 12},
        },
        9,
    )
    assert record["episode_id"] == 9
    assert record["scenario_episode_id"] == 1
    assert record["seed"] == 5
    assert record["scenario_source"] == "failure_buffer"
    assert record["collision"] is True
    assert record["safety_budget"] == 0.03
    assert record["budget_mode"] == "risk_adaptive"
    assert record["monitor_return"] == 3.0
    assert build_episode_log_record({}, 10)["episode_id"] == 10


class ResettableCost:
    name = "ResettableCost"

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(self, obs, action, reward, next_obs, terminated, truncated, info):
        from fasb.schemas.outputs import CostOutput

        return CostOutput(cost=0.0, components={})


def test_cost_function_wrapper_calls_plugin_reset() -> None:
    cost = ResettableCost()
    env = CostFunctionWrapper(DummySeedEnv(), cost)
    env.reset(seed=100)
    env.reset(seed=101)
    assert cost.reset_calls == 2


def test_checkpoint_finetune_honors_configured_algorithm_params(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_text("placeholder", encoding="utf-8")
    calls = {}

    class FakePPO:
        @classmethod
        def load(cls, path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return "loaded-model"

    import stable_baselines3

    monkeypatch.setattr(stable_baselines3, "PPO", FakePPO)
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "test", "output_dir": str(tmp_path)},
            "algorithm": {
                "checkpoint_path": str(checkpoint),
                "params": {
                    "learning_rate": 0.00003,
                    "device": "cpu",
                    "batch_size": 64,
                    "policy_kwargs": {"net_arch": [256, 256]},
                },
            },
        }
    )

    model = SB3Trainer(cfg)._build_model(env="env")

    assert model == "loaded-model"
    assert calls["path"] == str(checkpoint)
    assert calls["kwargs"]["env"] == "env"
    assert calls["kwargs"]["device"] == "cpu"
    assert calls["kwargs"]["learning_rate"] == 0.00003
    assert calls["kwargs"]["batch_size"] == 64
    assert "policy_kwargs" not in calls["kwargs"]
