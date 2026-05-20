from __future__ import annotations

from pathlib import Path
import shutil
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
                append_jsonl(self.log_path, build_episode_log_record(info, self.episode_id))
                self.episode_id += 1
        return True


def build_episode_log_record(info: dict[str, Any], episode_id: int) -> dict[str, Any]:
    scenario = info.get("fasb_scenario", {})
    if not isinstance(scenario, dict):
        scenario = {}
    metadata = scenario.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    monitor_episode = info.get("episode", {})
    if not isinstance(monitor_episode, dict):
        monitor_episode = {}

    record = {
        "episode_id": info.get("episode_id", scenario.get("episode_id", episode_id)),
        "seed": info.get("seed", scenario.get("seed", metadata.get("seed"))),
        "scenario_id": info.get("scenario_id", scenario.get("scenario_id", metadata.get("scenario_id"))),
        "scenario_source": info.get("scenario_source", scenario.get("source", metadata.get("source"))),
        "scenario_priority": info.get("scenario_priority", scenario.get("priority", metadata.get("priority"))),
        "success": bool(info.get("success", info.get("arrive_dest", metadata.get("success", False)))),
        "collision": bool(
            info.get("collision", info.get("crash", info.get("crash_vehicle", metadata.get("collision", False))))
        ),
        "offroad": bool(
            info.get("offroad", info.get("out_of_road", info.get("crash_offroad", metadata.get("offroad", False))))
        ),
        "timeout": bool(info.get("timeout", info.get("max_step", metadata.get("timeout", False)))),
        "route_completion": float(
            info.get("route_completion", info.get("route_completion_rate", metadata.get("route_completion", 0.0))) or 0.0
        ),
        "cost": float(info.get("fasb_cost", info.get("cost", metadata.get("cost", 0.0))) or 0.0),
        "safety_budget": info.get("fasb_budget", info.get("safety_budget")),
        "budget_mode": info.get("fasb_budget_mode", info.get("budget_mode")),
        "penalty_coef": info.get("fasb_penalty_coef", info.get("penalty_coef")),
        "risk_score": info.get("fasb_risk_score", info.get("risk_score", metadata.get("risk_score"))),
        "failure_score": info.get("fasb_failure_score", info.get("failure_score", metadata.get("failure_score"))),
        "failure_mode": info.get("fasb_failure_mode", info.get("failure_mode", metadata.get("failure_mode"))),
        "original_reward": info.get("fasb_original_reward", info.get("original_reward", metadata.get("reward"))),
        "modified_reward": info.get("fasb_modified_reward", info.get("modified_reward")),
        "cost_components": info.get("fasb_cost_components", info.get("cost_components")),
        "episode_length": info.get("episode_length", info.get("l", monitor_episode.get("l"))),
    }
    for key in ("r", "l", "t"):
        if key in monitor_episode:
            record[f"monitor_{key}"] = monitor_episode[key]
    if "r" in monitor_episode:
        record["monitor_return"] = monitor_episode["r"]
    if "l" in monitor_episode:
        record["monitor_length"] = monitor_episode["l"]
    return record


def checkpoint_callback(save_freq: int, save_path: str | Path):
    from stable_baselines3.common.callbacks import CheckpointCallback

    return CheckpointCallback(save_freq=max(int(save_freq), 1), save_path=str(save_path), name_prefix="latest")


class BestMeanRewardEvalCallback:
    def __new__(
        cls,
        eval_env: Any,
        eval_freq: int,
        best_model_save_path: str | Path,
        log_path: str | Path,
        deterministic: bool = True,
        n_eval_episodes: int = 5,
        verbose: int = 0,
    ):
        from stable_baselines3.common.callbacks import EvalCallback

        class _Callback(EvalCallback):
            def _on_step(self) -> bool:
                previous_best = self.best_mean_reward
                should_continue = super()._on_step()
                best_path = Path(self.best_model_save_path) / "best_model.zip"
                named_path = Path(self.best_model_save_path) / "best_mean_reward.zip"
                if self.best_mean_reward > previous_best and best_path.exists():
                    shutil.copyfile(best_path, named_path)
                return should_continue

        return _Callback(
            eval_env,
            best_model_save_path=str(best_model_save_path),
            log_path=str(log_path),
            eval_freq=max(int(eval_freq), 1),
            deterministic=deterministic,
            n_eval_episodes=max(int(n_eval_episodes), 1),
            verbose=verbose,
        )
