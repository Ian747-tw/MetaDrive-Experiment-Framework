from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fasb.core.imports import instantiate_from_config
from fasb.envs.metadrive_factory import make_metadrive_env
from fasb.envs.wrappers import CostFunctionWrapper, RandomPolicy
from fasb.evaluation.metrics import summarize_episode_records
from fasb.evaluation.scenario_sets import scenario_seeds
from fasb.plugins.failure_classifier import DefaultFailureClassifier
from fasb.plugins.failure_scorer import DefaultFailureScorer
from fasb.schemas.records import EpisodeRecord
from fasb.utils.io import append_jsonl


def extract_episode_record(
    run_id: str,
    episode_id: int,
    reward: float,
    modified_reward: float | None,
    cost: float,
    length: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    scenario = info.get("fasb_scenario", {})
    record = EpisodeRecord(
        run_id=run_id,
        episode_id=episode_id,
        seed=scenario.get("seed"),
        scenario_id=scenario.get("scenario_id"),
        reward=float(reward),
        modified_reward=modified_reward,
        cost=float(cost),
        success=bool(info.get("success", info.get("arrive_dest", False))),
        collision=bool(info.get("collision", info.get("crash", info.get("crash_vehicle", False)))),
        offroad=bool(info.get("offroad", info.get("out_of_road", info.get("crash_offroad", False)))),
        timeout=bool(info.get("timeout", info.get("max_step", False))),
        route_completion=float(info.get("route_completion", info.get("route_completion_rate", 0.0)) or 0.0),
        episode_length=int(length),
        traffic_density=scenario.get("traffic_density"),
        safety_budget=info.get("fasb_budget"),
        penalty_coef=info.get("fasb_penalty_coef"),
        min_vehicle_distance=info.get("min_vehicle_distance"),
        metadata={"scenario_source": scenario.get("source"), **dict(scenario.get("metadata", {}))},
    ).to_dict()
    scorer = DefaultFailureScorer()
    classifier = DefaultFailureClassifier()
    score = scorer.score(record)
    label = classifier.classify(record, score)
    record["failure_score"] = score.failure_score
    record["risk_score"] = score.risk_score
    record["failure_mode"] = label.failure_mode
    return record


class Evaluator:
    def __init__(self, config: DictConfig | dict[str, Any], run_dir: str | Path | None = None) -> None:
        self.config = OmegaConf.create(config)
        self.run_dir = Path(run_dir or self.config.experiment.get("output_dir", "runs/eval"))

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | None,
        scenario_set: str,
        n_episodes: int,
        deterministic: bool = True,
    ) -> dict[str, float]:
        metadrive_cfg = self.config.metadrive
        cost = instantiate_from_config(self.config.get("cost_function")) if self.config.get("cost_function") else None
        wrappers = [(CostFunctionWrapper, {"cost_function": cost})] if cost else []
        env = make_metadrive_env(metadrive_cfg, wrappers=wrappers, run_context={"traffic_density": metadrive_cfg.config.get("traffic_density")})
        model = self._load_policy(checkpoint_path, env)
        records = []
        start_seed = int(metadrive_cfg.config.get("start_seed", 0))
        for i, seed in enumerate(scenario_seeds(scenario_set, start_seed, n_episodes)):
            obs, info = env.reset(seed=seed)
            total_reward = 0.0
            total_modified = 0.0
            total_cost = 0.0
            length = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(info.get("fasb_original_reward", reward))
                total_modified += float(info.get("fasb_modified_reward", reward))
                total_cost += float(info.get("fasb_cost", 0.0) or 0.0)
                length += 1
                done = terminated or truncated
            record = extract_episode_record(self.run_dir.name, i, total_reward, total_modified, total_cost, length, info)
            records.append(record)
            append_jsonl(self.run_dir / "logs" / "episodes.jsonl", record)
        metrics = summarize_episode_records(records)
        self._write_metrics_csv(scenario_set, metrics)
        if hasattr(env, "close"):
            env.close()
        return metrics

    def _load_policy(self, checkpoint_path: str | None, env: Any):
        if checkpoint_path:
            from stable_baselines3 import PPO

            path = Path(checkpoint_path)
            if path.exists():
                return PPO.load(str(path), env=env)
        return RandomPolicy(env.action_space)

    def _write_metrics_csv(self, scenario_set: str, metrics: dict[str, float]) -> None:
        path = self.run_dir / "eval" / f"{scenario_set}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
