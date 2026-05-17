from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fasb.core.imports import instantiate_from_config
from fasb.core.run_dir import create_run_dir
from fasb.envs.metadrive_factory import make_metadrive_env
from fasb.envs.vec_env import make_vec_env
from fasb.envs.wrappers import AdaptiveRewardPenaltyWrapper, CostFunctionWrapper
from fasb.plugins.sampler import MixedFailureSampler, UniformSampler
from fasb.training.callbacks import EpisodeJSONLCallback, checkpoint_callback
from fasb.utils.seed import seed_everything


class SB3Trainer:
    def __init__(self, config: DictConfig | dict[str, Any]) -> None:
        self.config = OmegaConf.create(config)
        self.run_dir = Path(self.config.experiment.output_dir)

    def train(self) -> Path:
        seed_everything(int(self.config.experiment.get("seed", 0)))
        create_run_dir(self.run_dir, self.config, self.config.experiment.name)
        env = self._build_vec_env()
        model = self._build_model(env)
        callbacks = [
            EpisodeJSONLCallback(self.run_dir / "logs" / "episodes.jsonl"),
            checkpoint_callback(int(self.config.training.get("save_every_steps", 10000)), self.run_dir / "checkpoints"),
        ]
        model.learn(total_timesteps=int(self.config.training.total_timesteps), callback=callbacks)
        final_path = self.run_dir / "checkpoints" / "final.zip"
        model.save(str(final_path))
        env.close()
        return final_path

    def _build_vec_env(self):
        n_envs = int(self.config.vec_env.get("n_envs", 1))
        return make_vec_env([self._make_env_fn(i) for i in range(n_envs)], self.config.vec_env.get("type", "dummy"), self.config.vec_env.get("start_method", "forkserver"))

    def _make_env_fn(self, rank: int):
        def factory():
            sampler = self._build_sampler(rank)
            wrappers = self._build_wrappers()
            return make_metadrive_env(
                self.config.metadrive,
                wrappers=wrappers,
                scenario_sampler=sampler,
                run_context={"traffic_density": self.config.metadrive.config.get("traffic_density")},
            )

        return factory

    def _build_sampler(self, rank: int):
        mode = self.config.get("mode", self.config.experiment.get("mode", "fasb_ppo"))
        start_seed = int(self.config.metadrive.config.get("start_seed", 0)) + rank * 10000
        num_scenarios = int(self.config.metadrive.config.get("num_scenarios", 100))
        if mode in {"fixed_budget_ft", "fasb_ppo", "fasb_ppo_lagrangian_stretch"}:
            failure_path = self.config.get("failure_buffer", {}).get("path")
            return MixedFailureSampler(
                failure_buffer_path=failure_path,
                start_seed=start_seed,
                num_scenarios=num_scenarios,
                failure_ratio=float(self.config.sampler.get("failure_ratio", 0.6)) if self.config.get("sampler") else 0.6,
                alpha=float(self.config.sampler.get("alpha", 0.7)) if self.config.get("sampler") else 0.7,
            )
        return UniformSampler(start_seed, num_scenarios)

    def _build_wrappers(self) -> list[tuple[Any, dict[str, Any]]]:
        mode = self.config.get("mode", self.config.experiment.get("mode", "fasb_ppo"))
        wrappers: list[tuple[Any, dict[str, Any]]] = []
        cost = instantiate_from_config(self.config.get("cost_function")) if self.config.get("cost_function") else None
        if cost is not None and mode in {"fixed_budget_ft", "fasb_ppo", "fasb_ppo_lagrangian_stretch"}:
            wrappers.append((CostFunctionWrapper, {"cost_function": cost}))
        if mode in {"fixed_budget_ft", "fasb_ppo", "fasb_ppo_lagrangian_stretch"}:
            wrappers.append(
                (
                    AdaptiveRewardPenaltyWrapper,
                    {
                        "failure_scorer": instantiate_from_config(self.config.get("failure_scorer")),
                        "failure_classifier": instantiate_from_config(self.config.get("failure_classifier")),
                        "safety_budget": instantiate_from_config(self.config.get("safety_budget")),
                        "penalty_scheduler": instantiate_from_config(self.config.get("penalty_scheduler")),
                    },
                )
            )
        return wrappers

    def _build_model(self, env: Any):
        from stable_baselines3 import PPO

        checkpoint_path = self.config.algorithm.get("checkpoint_path")
        if checkpoint_path and Path(str(checkpoint_path)).exists():
            return PPO.load(str(checkpoint_path), env=env)
        params = OmegaConf.to_container(self.config.algorithm.get("params", {}), resolve=True)
        return PPO(self.config.algorithm.get("policy", "MlpPolicy"), env, **params)
