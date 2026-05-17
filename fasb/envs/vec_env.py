from __future__ import annotations

from typing import Callable


def make_vec_env(env_fns: list[Callable], vec_type: str = "dummy", start_method: str = "forkserver"):
    if vec_type == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(env_fns, start_method=start_method)
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv(env_fns)
