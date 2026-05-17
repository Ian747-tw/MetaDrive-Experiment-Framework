from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasb.core.config import load_config
from fasb.core.imports import instantiate_from_config
from fasb.envs.metadrive_factory import make_metadrive_env
from fasb.envs.vec_env import make_vec_env
from fasb.envs.wrappers import CostFunctionWrapper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subproc", action="store_true")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    cost = instantiate_from_config(cfg.get("cost_function")) if cfg.get("cost_function") else None
    wrappers = [(CostFunctionWrapper, {"cost_function": cost})] if cost else []
    env = make_metadrive_env(cfg.metadrive, wrappers=wrappers, run_context={"traffic_density": cfg.metadrive.config.get("traffic_density")})
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("PASS raw/wrapped reset-step")
    if hasattr(env, "close"):
        env.close()

    def fn():
        return make_metadrive_env(cfg.metadrive, wrappers=wrappers)

    vec = make_vec_env([fn], "dummy")
    vec.reset()
    vec.step([vec.action_space.sample()])
    vec.close()
    print("PASS DummyVecEnv")

    if args.subproc:
        vec = make_vec_env([fn], "subproc")
        vec.reset()
        vec.close()
        print("PASS SubprocVecEnv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
