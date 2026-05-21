from __future__ import annotations

from fasb.schemas.outputs import ScenarioSample


class FirstSeedSampler:
    name = "FirstSeedSampler"

    def __init__(self, start_seed: int = 1000, num_scenarios: int = 1) -> None:
        self.start_seed = start_seed
        self.num_scenarios = num_scenarios

    def next(self) -> ScenarioSample:
        return ScenarioSample(
            seed=int(self.start_seed),
            source="first_seed",
            priority=1.0,
            metadata={"num_scenarios": int(self.num_scenarios)},
        )
