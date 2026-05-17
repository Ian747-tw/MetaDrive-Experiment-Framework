from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioStore:
    records: dict[int, dict[str, Any]] = field(default_factory=dict)

    def put(self, seed: int, record: dict[str, Any]) -> None:
        self.records[int(seed)] = dict(record)

    def get(self, seed: int) -> dict[str, Any] | None:
        return self.records.get(int(seed))
