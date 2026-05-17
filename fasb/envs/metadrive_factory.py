from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from fasb.core.imports import import_string
from fasb.envs.api_compat import APICompatibilityWrapper
from fasb.envs.wrappers import ScenarioMetadataWrapper


def make_metadrive_env(
    metadrive_config: dict[str, Any] | DictConfig,
    wrappers: list[Any] | None = None,
    scenario_sampler: Any | None = None,
    run_context: dict[str, Any] | None = None,
):
    cfg = OmegaConf.to_container(metadrive_config, resolve=True) if isinstance(metadrive_config, DictConfig) else dict(metadrive_config)
    env_class_path = cfg.get("env_class", "metadrive.envs.MetaDriveEnv")
    pass_through = dict(cfg.get("config", {}))
    env_cls = import_string(str(env_class_path))
    env = env_cls(pass_through)
    env = APICompatibilityWrapper(env)
    env = ScenarioMetadataWrapper(env, scenario_sampler=scenario_sampler, run_context=run_context)
    for wrapper in wrappers or []:
        if isinstance(wrapper, tuple):
            wrapper_cls, kwargs = wrapper
            env = wrapper_cls(env, **kwargs)
        else:
            env = wrapper(env)
    return env
