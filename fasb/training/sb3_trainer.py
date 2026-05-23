from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fasb.core.imports import import_string, instantiate_from_config
from fasb.core.run_dir import create_run_dir
from fasb.envs.metadrive_factory import make_metadrive_env
from fasb.envs.vec_env import make_vec_env
from fasb.envs.wrappers import AdaptiveRewardPenaltyWrapper, CostFunctionWrapper
from fasb.plugins.sampler import UniformSampler
from fasb.training.callbacks import BestMeanRewardEvalCallback, EpisodeJSONLCallback, checkpoint_callback
from fasb.utils.seed import seed_everything


def _coerce_schedule(value: Any) -> Any:
    # "linear:3e-4" → SB3 schedule that decays peak → 0 over training.
    if isinstance(value, str) and value.startswith("linear:"):
        peak = float(value.split(":", 1)[1])
        return lambda frac: float(frac) * peak
    return value


def _accepts_kwarg(target_path: str, kwarg: str) -> bool:
    target = import_string(target_path)
    signature = inspect.signature(target)
    return kwarg in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


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
        eval_env = None
        eval_every_steps = int(self.config.training.get("eval_every_steps", 0) or 0)
        if eval_every_steps > 0:
            eval_env = self._build_eval_env()
            callbacks.append(
                BestMeanRewardEvalCallback(
                    eval_env,
                    eval_freq=eval_every_steps,
                    best_model_save_path=self.run_dir / "checkpoints",
                    log_path=self.run_dir / "eval",
                    deterministic=bool(self.config.training.get("deterministic_eval", True)),
                    n_eval_episodes=int(self.config.get("eval", {}).get("n_episodes", 5)),
                )
            )
        model.learn(total_timesteps=int(self.config.training.total_timesteps), callback=callbacks)
        final_path = self.run_dir / "checkpoints" / "final.zip"
        model.save(str(final_path))
        env.close()
        if eval_env is not None:
            eval_env.close()
        return final_path

    def _build_vec_env(self):
        n_envs = int(self.config.vec_env.get("n_envs", 1))
        return make_vec_env([self._make_env_fn(i) for i in range(n_envs)], self.config.vec_env.get("type", "dummy"), self.config.vec_env.get("start_method", "forkserver"))

    def _make_env_fn(self, rank: int):
        def factory():
            sampler = self._build_sampler(rank)
            wrappers = self._build_wrappers(rank)
            return make_metadrive_env(
                self.config.metadrive,
                wrappers=wrappers,
                scenario_sampler=sampler,
                run_context={"traffic_density": self.config.metadrive.config.get("traffic_density")},
            )

        return factory

    def _worker_seed_shard(self, rank: int) -> tuple[int, int]:
        base_seed = int(self.config.metadrive.config.get("start_seed", 0))
        total = max(int(self.config.metadrive.config.get("num_scenarios", 100)), 1)
        n_envs = max(int(self.config.vec_env.get("n_envs", 1)), 1)
        shard_size = max((total + n_envs - 1) // n_envs, 1)
        offset = min(max(rank, 0) * shard_size, total - 1)
        count = max(min(shard_size, total - offset), 1)
        return base_seed + offset, count

    def _build_sampler(self, rank: int):
        start_seed, num_scenarios = self._worker_seed_shard(rank)
        sampler_cfg = self.config.get("sampler")
        if sampler_cfg and "_target_" in sampler_cfg:
            cfg = OmegaConf.create(OmegaConf.to_container(sampler_cfg, resolve=True))
            cfg.start_seed = start_seed
            cfg.num_scenarios = num_scenarios
            target_path = str(cfg.get("_target_"))
            if "failure_buffer_path" not in cfg and _accepts_kwarg(target_path, "failure_buffer_path"):
                failure_path = self.config.get("failure_buffer", {}).get("path")
                if failure_path:
                    cfg.failure_buffer_path = failure_path
            sampler = instantiate_from_config(cfg)
            if not hasattr(sampler, "next"):
                raise TypeError(f"Sampler {sampler.__class__.__name__} must define next()")
            return sampler
        return UniformSampler(start_seed, num_scenarios)

    def _build_wrappers(self, rank: int | None = None) -> list[tuple[Any, dict[str, Any]]]:
        mode = self.config.get("mode", self.config.experiment.get("mode", "fasb_ppo"))
        wrappers: list[tuple[Any, dict[str, Any]]] = []
        run_context = {
            "experiment": self.config.experiment.get("name"),
            "mode": mode,
            "rank": rank,
        }
        runtime_kwargs = {"error_dir": self.run_dir / "errors", "run_context": run_context}
        cost = instantiate_from_config(self.config.get("cost_function")) if self.config.get("cost_function") else None
        # Cost is read-only signal; attach for every mode so naive_ft can log info["fasb_cost"]
        # for downstream failure mining without applying any reward shaping.
        if cost is not None:
            wrappers.append((CostFunctionWrapper, {"cost_function": cost, **runtime_kwargs}))
        if mode in {"fixed_budget_ft", "fasb_ppo", "fasb_ppo_lagrangian_stretch"}:
            wrappers.append(
                (
                    AdaptiveRewardPenaltyWrapper,
                    {
                        "failure_scorer": instantiate_from_config(self.config.get("failure_scorer")),
                        "failure_classifier": instantiate_from_config(self.config.get("failure_classifier")),
                        "safety_budget": instantiate_from_config(self.config.get("safety_budget")),
                        "penalty_scheduler": instantiate_from_config(self.config.get("penalty_scheduler")),
                        **runtime_kwargs,
                    },
                )
            )
        return wrappers

    def _build_eval_env(self):
        eval_config = OmegaConf.create(OmegaConf.to_container(self.config.metadrive, resolve=True))
        train_start = int(eval_config.config.get("start_seed", 0))
        train_count = int(eval_config.config.get("num_scenarios", 100))
        n_eval = int(self.config.get("eval", {}).get("n_episodes", 5))
        eval_config.config.start_seed = train_start + train_count
        eval_config.config.num_scenarios = max(n_eval, 1)

        def factory():
            return make_metadrive_env(
                eval_config,
                wrappers=self._build_wrappers(rank=None),
                scenario_sampler=UniformSampler(int(eval_config.config.start_seed), int(eval_config.config.num_scenarios)),
                run_context={"traffic_density": eval_config.config.get("traffic_density")},
            )

        return make_vec_env([factory], "dummy", self.config.vec_env.get("start_method", "forkserver"))

    def _build_model(self, env: Any):
        from stable_baselines3 import PPO

        params = OmegaConf.to_container(self.config.algorithm.get("params", {}), resolve=True) or {}
        for key in ("learning_rate", "clip_range", "clip_range_vf"):
            if key in params:
                params[key] = _coerce_schedule(params[key])

        checkpoint_path = self.config.algorithm.get("checkpoint_path")
        if checkpoint_path and Path(str(checkpoint_path)).exists():
            load_params = {key: value for key, value in params.items() if key not in {"device", "policy_kwargs"}}
            return PPO.load(
                str(checkpoint_path),
                env=env,
                device=params.get("device", "auto"),
                **load_params,
            )
        return PPO(self.config.algorithm.get("policy", "MlpPolicy"), env, **params)
