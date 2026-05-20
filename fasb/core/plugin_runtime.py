from __future__ import annotations

import json
import traceback as traceback_module
from pathlib import Path
from typing import Any

from fasb.utils.io import ensure_dir, to_jsonable


def safe_call_component(
    component: Any,
    method_name: str | None,
    component_type: str,
    run_context: dict[str, Any] | None,
    error_dir: str | Path | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        if method_name is None:
            return component(*args, **kwargs)
        return getattr(component, method_name)(*args, **kwargs)
    except Exception as exc:
        tb = traceback_module.format_exc()
        if error_dir is not None:
            path = ensure_dir(error_dir)
            record = {
                "component_type": component_type,
                "component": getattr(component, "name", component.__class__.__name__),
                "component_class": f"{component.__class__.__module__}.{component.__class__.__qualname__}",
                "method_name": method_name,
                "error_type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": tb,
                **(run_context or {}),
            }
            with (path / "plugin_errors.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
            with (path / "plugin_errors.log").open("a", encoding="utf-8") as f:
                f.write(
                    f"[{record.get('component_type')}] {record.get('component_class')}"
                    f".{method_name or '__call__'} failed: {record.get('error_type')}: {record.get('message')}\n"
                )
                f.write(tb)
                f.write("\n")
        raise
