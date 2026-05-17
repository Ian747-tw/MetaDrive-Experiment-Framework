# Plugin Authoring

Plugins are regular Python classes referenced from YAML with `_target_`.

## Custom Cost

```python
from fasb.schemas.outputs import CostOutput

class MyCost:
    name = "MyCost"

    def __call__(self, obs, action, reward, next_obs, terminated, truncated, info):
        cost = 1.0 if info.get("crash") else 0.0
        return CostOutput(cost=cost, components={"crash": cost})
```

```yaml
cost_function:
  _target_: my_package.my_plugins.MyCost
```

## Custom Safety Budget

Return `BudgetOutput` with a positive finite budget. The validator fails fast on invalid values.

## Custom Failure Scorer

Return `FailureScoreOutput` with `risk_score` in `[0, 1]`.

## Custom Sampler

Return `ScenarioSample(seed=int, source=str, priority=float, metadata=dict)`.

Validate before training:

```bash
python scripts/validate_components.py --config configs/train/my_custom_budget.yaml
```

Enable custom import paths in config:

```yaml
error_policy:
  allow_custom_imports: true
```
