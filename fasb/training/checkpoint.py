from __future__ import annotations

from pathlib import Path


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    candidates = sorted((Path(run_dir) / "checkpoints").glob("*.zip"))
    return candidates[-1] if candidates else None
