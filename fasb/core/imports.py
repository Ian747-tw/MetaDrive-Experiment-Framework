from __future__ import annotations

import importlib
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fasb.core.errors import ConfigurationError


def import_string(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ConfigurationError(f"Invalid import path: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def instantiate_from_config(config: dict | DictConfig | None, **kwargs: Any) -> Any:
    if config is None:
        return None
    cfg = OmegaConf.create(config) if isinstance(config, dict) else config
    if "_target_" not in cfg:
        raise ConfigurationError("Component config requires _target_")
    return instantiate(cfg, **kwargs)
