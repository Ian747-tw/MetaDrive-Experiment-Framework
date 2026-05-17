from __future__ import annotations

from pathlib import Path
from typing import Any

from fasb.utils.io import append_jsonl


try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover
    BaseCallback = object


class EpisodeJSONLCallback(BaseCallback):
    def __init__(self, log_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.episode_id = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) if hasattr(self, "locals") else []
        dones = self.locals.get("dones", []) if hasattr(self, "locals") else []
        for done, info in zip(dones, infos):
            if done:
                append_jsonl(
                    self.log_path,
                    {
                        "episode_id": self.episode_id,
                        "seed": info.get("fasb_scenario", {}).get("seed"),
                        "scenario_id": info.get("fasb_scenario", {}).get("scenario_id"),
                        "success": bool(info.get("success", info.get("arrive_dest", False))),
                        "collision": bool(info.get("collision", info.get("crash", False))),
                        "offroad": bool(info.get("offroad", info.get("out_of_road", False))),
                        "cost": float(info.get("fasb_cost", 0.0) or 0.0),
                        "safety_budget": info.get("fasb_budget"),
                        "penalty_coef": info.get("fasb_penalty_coef"),
                    },
                )
                self.episode_id += 1
        return True


def checkpoint_callback(save_freq: int, save_path: str | Path):
    from stable_baselines3.common.callbacks import CheckpointCallback

    return CheckpointCallback(save_freq=max(int(save_freq), 1), save_path=str(save_path), name_prefix="latest")
