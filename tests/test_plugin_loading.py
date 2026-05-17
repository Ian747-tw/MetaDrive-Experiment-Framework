from __future__ import annotations

from omegaconf import OmegaConf

from fasb.core.imports import instantiate_from_config


def test_hydra_instantiates_builtin_plugins() -> None:
    cfg = OmegaConf.create({"_target_": "fasb.plugins.safety_budget.AdaptiveSafetyBudget", "d_min": 0.02, "d_max": 0.1})
    plugin = instantiate_from_config(cfg)
    assert plugin.name == "AdaptiveSafetyBudget"


def test_validate_components_script_config_loads() -> None:
    cfg = OmegaConf.load("configs/train/fasb_ppo.yaml")
    assert instantiate_from_config(cfg.cost_function).name == "DefaultDrivingCost"
    assert instantiate_from_config(cfg.failure_scorer).name == "DefaultFailureScorer"
    assert instantiate_from_config(cfg.failure_classifier).name == "DefaultFailureClassifier"
    assert instantiate_from_config(cfg.safety_budget).name == "AdaptiveSafetyBudget"
