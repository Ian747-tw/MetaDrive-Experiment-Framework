from __future__ import annotations


def scenario_seeds(name: str, start_seed: int, n_episodes: int) -> list[int]:
    # MetaDrive validates reset seeds against the env's configured range. The
    # scenario-set config chooses the range; this helper only enumerates within it.
    return [start_seed + i for i in range(n_episodes)]
