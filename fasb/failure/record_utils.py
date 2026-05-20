from __future__ import annotations

from typing import Any


def build_training_scenario_record(info: dict[str, Any]) -> dict[str, Any]:
    """Flatten FASB scenario metadata for training-time scoring.

    Failure-buffer samples store the original episode/failure fields inside
    ``fasb_scenario.metadata``. Training plugins need those fields directly.
    """
    safe_info = info if isinstance(info, dict) else {}
    scenario_raw = safe_info.get("fasb_scenario", {})
    scenario = scenario_raw if isinstance(scenario_raw, dict) else {}
    metadata_raw = scenario.get("metadata", {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    record: dict[str, Any] = {}

    for key in (
        "seed",
        "scenario_id",
        "source",
        "priority",
        "traffic_density",
        "episode_id",
    ):
        if key in scenario:
            record[key] = scenario.get(key)

    if "source" in scenario:
        record["scenario_source"] = scenario.get("source")
    if "priority" in scenario:
        record["scenario_priority"] = scenario.get("priority")

    for key, value in scenario.items():
        if key != "metadata" and key not in record:
            record[key] = value

    record.update(metadata)

    if "source" not in record and "scenario_source" in record:
        record["source"] = record["scenario_source"]
    if "priority" not in record and "scenario_priority" in record:
        record["priority"] = record["scenario_priority"]

    record["metadata"] = metadata
    return record
