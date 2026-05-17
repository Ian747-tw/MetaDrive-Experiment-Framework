from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg
