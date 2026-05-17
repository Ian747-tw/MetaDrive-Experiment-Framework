from __future__ import annotations

from enum import Enum


class FineTuneMode(str, Enum):
    BASE_EVAL = "base_eval"
    NAIVE_FT = "naive_ft"
    FIXED_BUDGET_FT = "fixed_budget_ft"
    FASB_PPO = "fasb_ppo"
    FASB_PPO_LAGRANGIAN_STRETCH = "fasb_ppo_lagrangian_stretch"
