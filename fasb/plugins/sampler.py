from __future__ import annotations

import random
from typing import Any

from fasb.buffers.failure_buffer import FailureBuffer
from fasb.schemas.outputs import ScenarioSample


class UniformSampler:
    name = "UniformSampler"

    def __init__(self, start_seed: int = 0, num_scenarios: int = 100, source: str = "random") -> None:
        self.start_seed = start_seed
        self.num_scenarios = num_scenarios
        self.source = source

    def next(self) -> ScenarioSample:
        seed = self.start_seed + random.randrange(max(self.num_scenarios, 1))
        return ScenarioSample(seed=seed, source=self.source, priority=1.0, metadata={})


class MixedFailureSampler:
    name = "MixedFailureSampler"

    def __init__(
        self,
        failure_buffer: FailureBuffer | None = None,
        failure_buffer_path: str | None = None,
        start_seed: int = 0,
        num_scenarios: int = 100,
        failure_ratio: float = 0.6,
        alpha: float = 0.7,
        max_too_hard_ratio: float = 0.15,
    ) -> None:
        self.failure_buffer = failure_buffer or (
            FailureBuffer.load(failure_buffer_path) if failure_buffer_path else FailureBuffer()
        )
        self.uniform = UniformSampler(start_seed, num_scenarios)
        self.start_seed = int(start_seed)
        self.num_scenarios = int(num_scenarios)
        self._eligible_failure_buffer = FailureBuffer(
            [record for record in self.failure_buffer.records if self._in_seed_range(record.get("seed"))],
            max_size=self.failure_buffer.max_size,
        )
        self.failure_ratio = failure_ratio
        self.alpha = alpha
        self.max_too_hard_ratio = max_too_hard_ratio

    def _in_seed_range(self, seed: Any) -> bool:
        try:
            value = int(seed)
        except (TypeError, ValueError):
            return False
        return self.start_seed <= value < self.start_seed + max(self.num_scenarios, 1)

    def next(self) -> ScenarioSample:
        if len(self._eligible_failure_buffer) and random.random() < self.failure_ratio:
            record = self._eligible_failure_buffer.sample_priority(alpha=self.alpha)
            return ScenarioSample(
                seed=int(record["seed"]),
                source=str(record.get("source", "failure_buffer")),
                priority=float(record.get("priority", record.get("risk_score", 1.0))),
                metadata=dict(record),
            )
        return self.uniform.next()
