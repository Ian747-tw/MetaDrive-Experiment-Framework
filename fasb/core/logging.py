from __future__ import annotations

import traceback as tb
from pathlib import Path
from typing import Any

from fasb.utils.io import append_jsonl


def log_plugin_error(
    run_dir: str | Path,
    *,
    run_id: str,
    component_type: str,
    component_name: str,
    error: BaseException,
    seed: int | None = None,
    episode_id: int | None = None,
    step_id: int | None = None,
    config_path: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "component_type": component_type,
        "component_name": component_name,
        "seed": seed,
        "episode_id": episode_id,
        "step_id": step_id,
        "error_type": type(error).__name__,
        "message": str(error),
        "config_path": config_path,
        "traceback": "".join(tb.format_exception(type(error), error, error.__traceback__)),
        "context": context or {},
    }
    root = Path(run_dir)
    append_jsonl(root / "errors" / "plugin_errors.jsonl", record)
    with (root / "errors" / "plugin_errors.log").open("a", encoding="utf-8") as f:
        f.write(f"{record['error_type']}: {record['message']}\n{record['traceback']}\n")
    return record
