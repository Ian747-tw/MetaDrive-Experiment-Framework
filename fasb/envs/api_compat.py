from __future__ import annotations

from typing import Any

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None


def normalize_reset(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, {}


def normalize_step(result: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, dict(info)
    raise TypeError(f"Unsupported step result shape: {type(result).__name__}")


if gym is not None:

    class APICompatibilityWrapper(gym.Wrapper):
        def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            return normalize_reset(self.env.reset(**kwargs))

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return normalize_step(self.env.step(action))

else:

    class APICompatibilityWrapper:  # pragma: no cover
        def __init__(self, env: Any) -> None:
            self.env = env

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)

        def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            return normalize_reset(self.env.reset(**kwargs))

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return normalize_step(self.env.step(action))
