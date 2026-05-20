# Plugin Authoring

Plugins are regular Python classes referenced from YAML with `_target_`. They are the intended extension point for most research changes; avoid editing the training loop for ordinary experiments.

Plugin outputs must return dataclasses from `fasb.schemas.outputs`. Invalid outputs fail fast through validation or runtime checks. Runtime plugin errors are logged in `runs/<experiment>/errors/`.

Validate plugin wiring before training:

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
```

## Custom Cost Function

```python
from fasb.schemas.outputs import CostOutput


class CrashCost:
    name = "CrashCost"

    def __call__(self, obs, action, reward, next_obs, terminated, truncated, info):
        crash_cost = 1.0 if info.get("crash") else 0.0
        out_cost = 0.5 if info.get("out_of_road") else 0.0
        cost = crash_cost + out_cost
        return CostOutput(
            cost=cost,
            components={"crash": crash_cost, "out_of_road": out_cost},
        )
```

```yaml
cost_function:
  _target_: my_project.plugins.CrashCost
```

## Custom Safety Budget

```python
from fasb.schemas.outputs import BudgetOutput


class ConservativeBudget:
    name = "ConservativeBudget"

    def __init__(self, default_budget=0.03, hard_budget=0.01):
        self.default_budget = default_budget
        self.hard_budget = hard_budget

    def __call__(self, scenario_metadata=None, failure_stats=None):
        metadata = scenario_metadata or {}
        too_hard = bool(metadata.get("too_hard", False))
        budget = self.hard_budget if too_hard else self.default_budget
        return BudgetOutput(
            budget=budget,
            mode="hard" if too_hard else "default",
            reason={"too_hard": str(too_hard)},
        )
```

```yaml
safety_budget:
  _target_: my_project.plugins.ConservativeBudget
  default_budget: 0.03
  hard_budget: 0.01
```

## Custom Sampler

```python
from fasb.schemas.outputs import ScenarioSample


class FirstSeedSampler:
    name = "FirstSeedSampler"

    def __init__(self, start_seed=1000):
        self.start_seed = start_seed

    def sample(self, failure_buffer=None, rng=None):
        return ScenarioSample(
            seed=int(self.start_seed),
            source="custom",
            priority=1.0,
            metadata={"policy": "first_seed"},
        )
```

```yaml
sampler:
  _target_: my_project.plugins.FirstSeedSampler
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
