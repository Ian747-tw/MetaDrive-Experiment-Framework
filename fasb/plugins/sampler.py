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
        self.failure_ratio = failure_ratio
        self.alpha = alpha
        self.max_too_hard_ratio = max_too_hard_ratio

    def next(self) -> ScenarioSample:
        if len(self.failure_buffer) and random.random() < self.failure_ratio:
            record = self.failure_buffer.sample_priority(alpha=self.alpha)
            return ScenarioSample(
                seed=int(record["seed"]),
                source=str(record.get("source", "failure_buffer")),
                priority=float(record.get("priority", record.get("risk_score", 1.0))),
                metadata=dict(record),
            )
        return self.uniform.next()
