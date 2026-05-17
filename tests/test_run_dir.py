from __future__ import annotations

from omegaconf import OmegaConf

from fasb.core.run_dir import create_run_dir


def test_create_run_dir(tmp_path) -> None:
    cfg = OmegaConf.create({"experiment": {"name": "x"}})
    root = create_run_dir(tmp_path / "run", cfg, "x")
    assert (root / "config_resolved.yaml").exists()
    assert (root / "metadata.json").exists()
    assert (root / "checkpoints").is_dir()
    assert (root / "logs").is_dir()
    assert (root / "errors").is_dir()
