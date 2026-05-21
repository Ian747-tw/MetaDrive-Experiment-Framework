# Plugin Authoring

Plugins are regular Python classes referenced from YAML with `_target_`. They are the intended extension point for research changes; teammates should customize configs and plugins instead of editing the training loop.

Plugin outputs must return dataclasses from `fasb.schemas.outputs`. Invalid outputs fail fast through validation or runtime checks. Runtime plugin errors are logged in `runs/<experiment>/errors/`.

Validate plugin wiring before training:

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
```

## Required Interfaces

```text
cost_function.__call__(obs, action, reward, next_obs, terminated, truncated, info) -> CostOutput
failure_scorer.score(episode_record) -> FailureScoreOutput
failure_classifier.classify(episode_record, score) -> FailureLabelOutput
safety_budget.get_budget(scenario_record, score, label) -> BudgetOutput
penalty_scheduler.get_penalty(scenario_record, budget, score, label) -> PenaltyOutput
sampler.next() -> ScenarioSample
```

`safety_budget.__call__` and `sampler.sample` are not framework extension points.

## Copy-Paste Examples

```python
from fasb.plugins.cost import DefaultDrivingCost
from fasb.schemas.outputs import (
    BudgetOutput,
    CostOutput,
    FailureLabelOutput,
    FailureScoreOutput,
    PenaltyOutput,
    ScenarioSample,
)


class CrashOnlyCost:
    name = "CrashOnlyCost"

    def __call__(self, obs, action, reward, next_obs, terminated, truncated, info):
        collision = float(bool(info.get("crash") or info.get("collision") or info.get("crash_vehicle")))
        offroad = float(bool(info.get("out_of_road") or info.get("offroad") or info.get("crash_offroad")))
        return CostOutput(
            cost=collision + offroad,
            components={"collision": collision, "offroad": offroad},
        )


class NearMissHeavyCost(DefaultDrivingCost):
    name = "NearMissHeavyCost"

    def __init__(self):
        super().__init__(
            collision_weight=1.0,
            offroad_weight=1.0,
            near_miss_weight=0.5,
            min_distance_threshold=5.0,
        )


class NearFailureScorer:
    name = "NearFailureScorer"

    def score(self, episode_record):
        collision = float(bool(episode_record.get("collision")))
        offroad = float(bool(episode_record.get("offroad")))
        route_completion = float(episode_record.get("route_completion") or 0.0)
        near_miss = float(episode_record.get("near_miss", episode_record.get("min_vehicle_distance_risk", 0.0)) or 0.0)
        low_progress = max(0.0, 1.0 - route_completion)
        raw = 3.0 * near_miss + 2.0 * low_progress + 2.0 * collision + 1.5 * offroad
        risk = min(max(raw / 6.0, 0.0), 1.0)
        return FailureScoreOutput(
            failure_score=float(raw),
            risk_score=float(risk),
            reason={
                "near_miss": near_miss,
                "low_progress": low_progress,
                "collision": collision,
                "offroad": offroad,
            },
        )


class TimeoutRelaxedBudget:
    name = "TimeoutRelaxedBudget"

    def __init__(self, d_min=0.02, d_max=0.10, timeout_budget=0.08):
        self.d_min = d_min
        self.d_max = d_max
        self.timeout_budget = timeout_budget

    def get_budget(self, scenario_record, score, label):
        if label.failure_mode in {"timeout_or_hesitation", "low_progress"}:
            budget = self.timeout_budget
            mode = "relaxed_progress"
        elif label.failure_mode in {"collision", "offroad"}:
            budget = self.d_min
            mode = "strict_safety"
        else:
            budget = self.d_max - (self.d_max - self.d_min) * score.risk_score
            mode = "risk_adaptive"
        budget = min(max(float(budget), self.d_min), self.d_max)
        return BudgetOutput(
            budget=budget,
            mode=mode,
            reason={"failure_mode": label.failure_mode, "risk_score": score.risk_score},
        )


class FixedPenaltyScheduler:
    name = "FixedPenaltyScheduler"

    def __init__(self, penalty_coef=1.0):
        self.penalty_coef = penalty_coef

    def get_penalty(self, scenario_record, budget, score, label):
        return PenaltyOutput(
            penalty_coef=float(self.penalty_coef),
            reason={"budget": budget.budget, "failure_mode": label.failure_mode},
        )


class FirstSeedSampler:
    name = "FirstSeedSampler"

    def __init__(self, start_seed=1000):
        self.start_seed = start_seed

    def next(self):
        return ScenarioSample(
            seed=int(self.start_seed),
            source="first_seed",
            priority=1.0,
            metadata={"policy": "first_seed"},
        )
```

Example YAML:

```yaml
cost_function:
  _target_: examples.custom_plugins.crash_only_cost.CrashOnlyCost

failure_scorer:
  _target_: examples.custom_plugins.near_failure_scorer.NearFailureScorer

safety_budget:
  _target_: examples.custom_plugins.timeout_relaxed_budget.TimeoutRelaxedBudget

penalty_scheduler:
  _target_: examples.custom_plugins.fixed_penalty_scheduler.FixedPenaltyScheduler
  penalty_coef: 1.0

sampler:
  _target_: examples.custom_plugins.first_seed_sampler.FirstSeedSampler
  start_seed: 1000
```

## Import Paths

Keep plugin modules importable from the active environment. For local packages, install the repo or plugin package in editable mode.

If custom imports are intentionally allowed, keep this in config:

```yaml
error_policy:
  allow_custom_imports: true
```

If validation fails, check the `_target_` path first, then confirm the plugin returns the exact output dataclass expected for that extension point.
