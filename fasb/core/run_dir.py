from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fasb.utils.io import ensure_dir


RUN_SUBDIRS = ["checkpoints", "logs", "buffers", "errors", "eval", "analysis", "analysis/plots"]


def create_run_dir(output_dir: str | Path, config: Any | None = None, run_id: str | None = None) -> Path:
    root = ensure_dir(output_dir)
    for subdir in RUN_SUBDIRS:
        ensure_dir(root / subdir)
    metadata = {"run_id": run_id or root.name, "created_at": time.time()}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if config is not None:
        save_resolved_config(config, root / "config_resolved.yaml")
    return root


def save_resolved_config(config: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, DictConfig):
        content = OmegaConf.to_yaml(config, resolve=True)
    elif isinstance(config, dict):
        content = OmegaConf.to_yaml(OmegaConf.create(config), resolve=True)
    elif is_dataclass(config):
        content = OmegaConf.to_yaml(OmegaConf.create(asdict(config)), resolve=True)
    else:
        content = str(config)
    p.write_text(content, encoding="utf-8")
