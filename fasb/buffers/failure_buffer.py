from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from fasb.utils.io import read_jsonl, write_jsonl


class FailureBuffer:
    def __init__(self, records: list[dict[str, Any]] | None = None, max_size: int = 1000) -> None:
        self.records = list(records or [])
        self.max_size = max_size
        self._trim()

    def add(self, record: dict[str, Any]) -> None:
        if "seed" not in record:
            raise ValueError("failure buffer record requires seed")
        normalized = dict(record)
        normalized.setdefault("priority", normalized.get("risk_score", normalized.get("failure_score", 1.0)))
        normalized.setdefault("metadata", {})
        self.records.append(normalized)
        self._trim()

    def sample(self, n: int = 1) -> list[dict[str, Any]]:
        if not self.records:
            return []
        return [dict(r) for r in random.choices(self.records, k=n)]

    def sample_priority(self, alpha: float = 0.7) -> dict[str, Any]:
        if not self.records:
            raise ValueError("cannot priority-sample empty failure buffer")
        weights = []
        for record in self.records:
            risk = float(record.get("risk_score", record.get("priority", 0.0)) or 0.0)
            learnability = float(record.get("learnability", 1.0) or 0.0)
            mode = record.get("failure_mode")
            if mode == "solved":
                learnability *= 0.1
            if record.get("always_fails"):
                learnability *= 0.3
            weights.append(max((1e-6 + risk) ** alpha * learnability, 1e-9))
        return dict(random.choices(self.records, weights=weights, k=1)[0])

    def save(self, path: str | Path) -> None:
        write_jsonl(path, self.records)

    @classmethod
    def load(cls, path: str | Path | None, max_size: int = 1000) -> "FailureBuffer":
        if path is None:
            return cls(max_size=max_size)
        return cls(read_jsonl(path), max_size=max_size)

    def _trim(self) -> None:
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size :]

    def __len__(self) -> int:
        return len(self.records)
