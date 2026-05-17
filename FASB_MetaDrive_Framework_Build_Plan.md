# FASB-MetaDrive Framework Build Plan for Codex CLI

**Project name:** `fasb-metadrive`  
**Working method:** Failure-Aware Safety-Budgeted Fine-Tuning for generalized MetaDrive RL policies  
**Primary backend:** MetaDrive + Gymnasium-style wrappers + Stable-Baselines3 PPO + Hydra/OmegaConf configs  
**Build goal:** A lightweight, stable, research-flexible backend that teammates can clone, configure, and run without rewriting training scripts.

---

## 0. Context for Codex

Build a framework, not a pile of scripts. The framework should support our research project:

> Start from a generalized MetaDrive driving policy/checkpoint, evaluate it to discover failure modes, fine-tune it with failure-aware scenario sampling and adaptive safety budgets, then evaluate/benchmark against baselines.

The framework must be:

- **Stable:** shared training/eval/checkpoint/logging core; no one-off scripts that silently diverge.
- **Lightweight:** use existing tools instead of rebuilding RL infrastructure.
- **Research-flexible:** cost functions, safety budgets, failure scoring, failure classification, samplers, and metrics must be swappable plugins.
- **Strict:** custom plugin outputs must be validated; invalid outputs fail fast by default and write useful error logs.
- **MetaDrive-compatible:** pass through MetaDrive settings instead of hiding them behind hardcoded choices.
- **SB3-compatible:** use Stable-Baselines3 for PPO, vectorized envs, callbacks, saving/loading, and basic training mechanics.

Do **not** build a full RL library. Build a MetaDrive + SB3 experiment layer for failure-aware safety specialization.

---

## 1. External frameworks to use, not rebuild

Use these existing tools:

| Need | Use | Framework responsibility | Our framework responsibility |
|---|---|---|---|
| Driving simulator | MetaDrive | Procedural driving envs, scenario seeds, traffic settings | Env factory, scenario metadata, seed splits, failure logging |
| Env augmentation | Gymnasium-style wrappers | Modular env wrapping | Cost wrapper, scenario logging wrapper, adaptive reward penalty wrapper |
| RL algorithm | Stable-Baselines3 | PPO/SAC implementations, rollout collection, model save/load | FASB-specific fine-tuning orchestration |
| Vectorized envs | SB3 `DummyVecEnv` / `SubprocVecEnv` | Parallel env execution | Scenario-aware env creation per worker |
| Callbacks | SB3 callbacks | Training hooks, checkpoints, eval hooks | Custom failure/eval/checkpoint callbacks |
| Configs | Hydra + OmegaConf | Config composition and object instantiation | Research plugin configs and run configs |
| Data logs | JSONL + CSV + TensorBoard optional | Simple durable run logs | Episode records, failure buffers, benchmark summaries |
| Analysis | Pandas + Matplotlib | Data aggregation and plots | Failure-mode reports and paper tables |

Do **not** implement PPO/SAC, rollout buffers, neural-network optimizers, vectorized env internals, or generic checkpoint systems from scratch.

---

## 2. MVP vs stretch

### MVP method: `FASB-PPO`

Use:

```text
Stable-Baselines3 PPO
+ failure-aware scenario sampler
+ configurable cost function plugin
+ adaptive safety-budget plugin
+ adaptive reward-penalty wrapper
+ failure-mode evaluation
```

Training signal:

\[
R'_t = R_t - \lambda_i C_t
\]

where \(C_t\) comes from the cost plugin and \(\lambda_i\) is derived from scenario risk / budget / failure mode.

This is implementable with SB3 PPO and wrappers without rewriting PPO internals.

### Stretch method: `FASB-PPO-Lagrangian`

Only after MVP works:

```text
Separate cost tracking
Scenario-conditioned budget d_i
Episode-level lambda update
Optional custom PPO-Lagrangian trainer or callback-enhanced approximation
```

Do not start with the stretch method. A perfect PPO-Lagrangian implementation is useless if failure logging, evaluation, and baselines are broken.

---

## 3. Expected user commands

After build, teammates should run:

```bash
# 1. Validate that env + plugins + configs work
python scripts/smoke_test_env.py --config configs/env/metadrive_generalization.yaml
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml

# 2. Discover failure modes of a base checkpoint
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml

# 3. Train baselines and FASB
python scripts/train.py --config configs/train/naive_ft.yaml
python scripts/train.py --config configs/train/fixed_budget_ft.yaml
python scripts/train.py --config configs/train/fasb_ppo.yaml

# 4. Evaluate a checkpoint
python scripts/evaluate.py --config configs/eval/heldout_random.yaml --checkpoint runs/fasb_ppo/checkpoints/best_score.zip

# 5. Analyze failures and produce paper tables
python scripts/analyze_failures.py --run runs/fasb_ppo

# 6. Run the full benchmark suite
python scripts/benchmark.py --config configs/benchmark/final.yaml
```

---

## 4. Repository layout

Create this structure:

```text
fasb-metadrive/
  README.md
  pyproject.toml
  BUILD_PLAN.md
  LICENSE

  configs/
    env/
      metadrive_generalization.yaml
      metadrive_debug.yaml
    explore/
      base_checkpoint.yaml
    train/
      naive_ft.yaml
      fixed_budget_ft.yaml
      fasb_ppo.yaml
      fasb_ppo_lagrangian_stretch.yaml
    eval/
      heldout_random.yaml
      discovered_failures.yaml
      heldout_failure_modes.yaml
    benchmark/
      final.yaml
    components/
      cost/default_driving_cost.yaml
      cost/distance_aware_cost.yaml
      scorer/default_failure_scorer.yaml
      classifier/default_failure_classifier.yaml
      budget/fixed_budget.yaml
      budget/adaptive_budget.yaml
      sampler/uniform_sampler.yaml
      sampler/mixed_failure_sampler.yaml

  scripts/
    smoke_test_env.py
    validate_components.py
    explore_failures.py
    train.py
    evaluate.py
    analyze_failures.py
    benchmark.py

  fasb/
    __init__.py

    core/
      __init__.py
      config.py
      registry.py
      validation.py
      errors.py
      logging.py
      run_dir.py
      imports.py

    envs/
      __init__.py
      metadrive_factory.py
      api_compat.py
      wrappers.py
      vec_env.py

    schemas/
      __init__.py
      records.py
      outputs.py

    plugins/
      __init__.py
      base.py
      cost.py
      failure_scorer.py
      failure_classifier.py
      scenario_risk.py
      safety_budget.py
      penalty_scheduler.py
      sampler.py
      metrics.py

    buffers/
      __init__.py
      failure_buffer.py
      scenario_store.py

    training/
      __init__.py
      sb3_trainer.py
      fine_tune.py
      callbacks.py
      baselines.py
      checkpoint.py

    evaluation/
      __init__.py
      evaluator.py
      metrics.py
      scenario_sets.py
      benchmark_suite.py

    analysis/
      __init__.py
      failure_report.py
      tables.py
      plots.py

    utils/
      __init__.py
      seed.py
      io.py
      timing.py

  examples/
    custom_plugins/
      curve_aware_budget.py
      cautious_distance_cost.py
    minimal_run.md
    plugin_authoring.md

  tests/
    test_component_validation.py
    test_failure_buffer.py
    test_env_smoke.py
    test_plugin_loading.py
    test_metrics.py
    test_run_dir.py
```

---

## 5. Dependency plan

Use a minimal dependency set in `pyproject.toml`.

Recommended dependencies:

```toml
[project]
name = "fasb-metadrive"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "metadrive-simulator",
  "stable-baselines3[extra]",
  "gymnasium",
  "hydra-core",
  "omegaconf",
  "numpy",
  "pandas",
  "matplotlib",
  "pyyaml",
  "rich",
  "tensorboard",
]

[project.optional-dependencies]
dev = [
  "pytest",
  "ruff",
  "mypy",
]
```

Notes for Codex:

- If MetaDrive installation uses package name `metadrive` instead of `metadrive-simulator` in the target environment, adapt dependency and document it clearly.
- Do not pin versions too aggressively unless compatibility breaks.
- Add a `requirements-dev.txt` only if project tooling needs it; otherwise keep dependencies centralized in `pyproject.toml`.

---

## 6. Core design contract

The framework must guarantee:

```text
1. Same config + same seed ranges -> reproducible run structure.
2. Every run saves its resolved config.
3. Every episode saves scenario + failure data.
4. Every checkpoint has evaluation metrics.
5. Baselines and FASB use the same evaluator.
6. Research components are swappable via config.
7. Every plugin output is validated before affecting training.
8. Plugin failures are logged with component name, seed, episode, step, config, and traceback.
9. Invalid plugin outputs fail fast by default.
10. MetaDrive settings are passed through, not reimplemented.
```

---

## 7. Config system

Use Hydra/OmegaConf.

### Main train config example

`configs/train/fasb_ppo.yaml`:

```yaml
experiment:
  name: fasb_ppo
  seed: 42
  output_dir: runs/fasb_ppo
  save_resolved_config: true

metadrive:
  env_class: metadrive.envs.MetaDriveEnv
  config:
    start_seed: 1000
    num_scenarios: 200
    traffic_density: 0.1
    random_traffic: true
    use_render: false
    # Keep this as pass-through. Users may add MetaDrive settings here.

vec_env:
  type: dummy        # dummy | subproc
  n_envs: 1
  start_method: forkserver

algorithm:
  backend: sb3
  name: PPO
  policy: MlpPolicy
  checkpoint_path: checkpoints/base_generalized.zip
  params:
    learning_rate: 0.0001
    n_steps: 2048
    batch_size: 256
    gamma: 0.99
    verbose: 1
    tensorboard_log: runs/tensorboard

training:
  total_timesteps: 100000
  save_every_steps: 10000
  eval_every_steps: 10000
  deterministic_eval: true

failure_buffer:
  path: runs/base_explore/failure_buffer.jsonl
  max_size: 1000

cost_function:
  _target_: fasb.plugins.cost.DefaultDrivingCost

failure_scorer:
  _target_: fasb.plugins.failure_scorer.DefaultFailureScorer

failure_classifier:
  _target_: fasb.plugins.failure_classifier.DefaultFailureClassifier

safety_budget:
  _target_: fasb.plugins.safety_budget.AdaptiveSafetyBudget
  d_min: 0.02
  d_max: 0.10
  timeout_budget: 0.07

penalty_scheduler:
  _target_: fasb.plugins.penalty_scheduler.RiskPenaltyScheduler
  lambda_min: 0.1
  lambda_max: 5.0

sampler:
  _target_: fasb.plugins.sampler.MixedFailureSampler
  failure_ratio: 0.6
  alpha: 0.7
  max_too_hard_ratio: 0.15

error_policy:
  plugin_error: fail_fast        # fail_fast | fallback
  validation_error: fail_fast
  allow_custom_imports: true
  max_fallbacks: 0

eval:
  n_episodes: 200
  scenario_sets:
    - heldout_random
    - discovered_failures
    - heldout_failure_modes
```

### Config rules

- All MetaDrive settings must live under `metadrive.config` and be passed through to the env constructor.
- All swappable research components must be created from config.
- Every run must save the fully resolved config at `runs/<exp>/config_resolved.yaml`.
- Use default built-in configs, but allow custom plugin paths when `allow_custom_imports: true`.

---

## 8. Data schemas

Create dataclasses in `fasb/schemas/records.py` and `fasb/schemas/outputs.py`.

### Episode record

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class EpisodeRecord:
    run_id: str
    episode_id: int
    seed: int | None
    scenario_id: str | None
    reward: float
    modified_reward: float | None
    cost: float
    success: bool
    collision: bool
    offroad: bool
    timeout: bool
    route_completion: float
    episode_length: int
    traffic_density: float | None = None
    failure_mode: str | None = None
    failure_score: float | None = None
    risk_score: float | None = None
    safety_budget: float | None = None
    penalty_coef: float | None = None
    min_vehicle_distance: float | None = None
    hard_brake_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Plugin outputs

```python
@dataclass
class CostOutput:
    cost: float
    components: dict[str, float]

@dataclass
class FailureScoreOutput:
    failure_score: float
    risk_score: float
    reason: dict[str, float]

@dataclass
class FailureLabelOutput:
    failure_mode: str
    confidence: float
    reason: dict[str, float | str]

@dataclass
class BudgetOutput:
    budget: float
    mode: str
    reason: dict[str, float | str]

@dataclass
class PenaltyOutput:
    penalty_coef: float
    reason: dict[str, float | str]

@dataclass
class ScenarioSample:
    seed: int
    source: str
    priority: float
    metadata: dict
```

---

## 9. Plugin interfaces

Use `typing.Protocol` in `fasb/plugins/base.py` or separate files.

### Cost function

```python
class CostFunction(Protocol):
    name: str

    def __call__(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> CostOutput:
        ...
```

### Failure scorer

```python
class FailureScorer(Protocol):
    name: str

    def score(self, episode_record: dict) -> FailureScoreOutput:
        ...
```

### Failure classifier

```python
class FailureClassifier(Protocol):
    name: str

    def classify(self, episode_record: dict, score: FailureScoreOutput) -> FailureLabelOutput:
        ...
```

### Safety budgeter

```python
class SafetyBudgeter(Protocol):
    name: str

    def get_budget(
        self,
        scenario_record: dict,
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> BudgetOutput:
        ...
```

### Penalty scheduler

```python
class PenaltyScheduler(Protocol):
    name: str

    def get_penalty(
        self,
        scenario_record: dict,
        budget: BudgetOutput,
        score: FailureScoreOutput,
        label: FailureLabelOutput,
    ) -> PenaltyOutput:
        ...
```

### Scenario sampler

```python
class ScenarioSampler(Protocol):
    name: str

    def next(self) -> ScenarioSample:
        ...
```

---

## 10. Built-in plugin implementations

### `DefaultDrivingCost`

File: `fasb/plugins/cost.py`

Responsibilities:

- Extract collision/offroad signals from common MetaDrive `info` keys.
- Return non-negative finite cost.
- Include component breakdown.

Expected behavior:

```python
cost = collision + offroad + 0.1 * near_miss_optional
```

Use defensive `info.get(...)` checks because MetaDrive info keys can vary across env/task versions.

### `DefaultFailureScorer`

File: `fasb/plugins/failure_scorer.py`

Default formula:

```python
raw = (
    5.0 * collision
    + 4.0 * offroad
    + 2.0 * timeout
    + 1.0 * (1.0 - route_completion)
    + 2.0 * near_miss
)
risk = min(raw / 8.0, 1.0)
```

If `near_miss` is unavailable, treat it as `0.0`.

### `DefaultFailureClassifier`

File: `fasb/plugins/failure_classifier.py`

Rules:

```text
collision true -> collision
else offroad true -> offroad
else timeout true and route_completion low -> timeout_or_hesitation
else route_completion < threshold -> low_progress
else near_miss > 0 -> near_miss
else success true -> solved
else -> unknown
```

### `AdaptiveSafetyBudget`

File: `fasb/plugins/safety_budget.py`

Formula:

\[
d_i = d_{max} - (d_{max} - d_{min}) r_i
\]

Special case:

```text
If failure mode is timeout_or_hesitation or low_progress, use moderate budget instead of blindly strict budget.
```

### `RiskPenaltyScheduler`

File: `fasb/plugins/penalty_scheduler.py`

MVP behavior:

```python
penalty = lambda_min + (lambda_max - lambda_min) * risk_score
```

Optional: allow failure-mode multipliers:

```yaml
mode_multipliers:
  collision: 1.2
  offroad: 1.1
  timeout_or_hesitation: 0.7
  low_progress: 0.6
```

### `MixedFailureSampler`

File: `fasb/plugins/sampler.py`

Mixed sampling:

```text
failure_ratio from failure buffer
1 - failure_ratio from random seed pool
```

Priority:

\[
P(i) \propto (\epsilon + r_i)^\alpha L_i
\]

Learnability filter:

```text
solved -> low priority
sometimes fails -> high priority
always fails -> capped priority
```

---

## 11. Validation layer

File: `fasb/core/validation.py`

Validate all plugin outputs before they affect training.

### Required validators

```python
def validate_cost_output(output: CostOutput) -> None: ...
def validate_failure_score_output(output: FailureScoreOutput) -> None: ...
def validate_failure_label_output(output: FailureLabelOutput) -> None: ...
def validate_budget_output(output: BudgetOutput, d_min: float | None = None, d_max: float | None = None) -> None: ...
def validate_penalty_output(output: PenaltyOutput) -> None: ...
def validate_scenario_sample(output: ScenarioSample) -> None: ...
```

### Validation rules

```text
cost: finite float, >= 0
cost components: dict[str, finite float]
failure_score: finite float, >= 0
risk_score: finite float in [0, 1]
failure_mode: non-empty string
confidence: finite float in [0, 1]
budget: finite float, > 0, optionally within configured range
penalty_coef: finite float, >= 0
scenario seed: int
sampler source: non-empty string
```

If validation fails, raise `ComponentValidationError`.

---

## 12. Error logging

File: `fasb/core/errors.py` and `fasb/core/logging.py`

### Error records

When a plugin fails, write JSONL to:

```text
runs/<exp>/errors/plugin_errors.jsonl
runs/<exp>/errors/plugin_errors.log
```

Each error record must include:

```json
{
  "run_id": "fasb_ppo",
  "component_type": "SafetyBudgeter",
  "component_name": "AdaptiveSafetyBudget",
  "seed": 1042,
  "episode_id": 17,
  "step_id": 88,
  "error_type": "ComponentValidationError",
  "message": "budget must be positive, got -0.4",
  "config_path": "configs/train/fasb_ppo.yaml",
  "traceback": "..."
}
```

### Error policy

Default:

```yaml
error_policy:
  plugin_error: fail_fast
  validation_error: fail_fast
  allow_fallback: false
```

Fallback mode is allowed only if explicitly configured:

```yaml
error_policy:
  plugin_error: fallback
  fallback_component: default_adaptive_budget
  max_fallbacks: 3
```

Do not silently continue after invalid plugin outputs.

---

## 13. MetaDrive compatibility layer

File: `fasb/envs/metadrive_factory.py`

### Responsibilities

- Import env class from config.
- Pass through `metadrive.config` exactly.
- Apply scenario seed overrides from samplers.
- Create train/eval envs consistently.
- Avoid modifying MetaDrive internals.

### Factory API

```python
def make_metadrive_env(
    metadrive_config: dict,
    wrappers: list | None = None,
    scenario_sampler: ScenarioSampler | None = None,
    run_context: dict | None = None,
):
    ...
```

### MetaDrive config pass-through

Support config like:

```yaml
metadrive:
  env_class: metadrive.envs.MetaDriveEnv
  config:
    start_seed: 1000
    num_scenarios: 200
    traffic_density: 0.1
    random_traffic: true
    use_render: false
```

Do not force users into our own limited config schema.

### Reset/step API compatibility

File: `fasb/envs/api_compat.py`

Support both:

```text
old Gym: reset() -> obs; step() -> obs, reward, done, info
new Gymnasium: reset() -> obs, info; step() -> obs, reward, terminated, truncated, info
```

Normalize internally so wrappers/training code see a consistent API.

---

## 14. Wrapper design

File: `fasb/envs/wrappers.py`

Recommended order:

```text
Raw MetaDriveEnv
-> APICompatibilityWrapper if needed
-> ScenarioMetadataWrapper
-> CostFunctionWrapper
-> AdaptiveRewardPenaltyWrapper
-> Monitor / VecEnv
-> SB3 PPO
```

### `ScenarioMetadataWrapper`

Tracks:

```text
current seed
scenario id
scenario source: random/failure_buffer/heldout
traffic density
optional map/scenario metadata if accessible
```

Adds metadata to `info`.

### `CostFunctionWrapper`

Computes safety cost using configured plugin.

Adds to `info`:

```python
info["fasb_cost"] = cost
info["fasb_cost_components"] = components
```

### `AdaptiveRewardPenaltyWrapper`

At scenario reset:

1. Read scenario metadata.
2. Get failure score/risk if metadata exists.
3. Get safety budget.
4. Get penalty coefficient.

At each step:

```python
modified_reward = original_reward - penalty_coef * cost
```

Adds to `info`:

```python
info["fasb_original_reward"] = original_reward
info["fasb_modified_reward"] = modified_reward
info["fasb_budget"] = budget
info["fasb_penalty_coef"] = penalty_coef
info["fasb_failure_mode"] = failure_mode
```

---

## 15. Failure buffer

File: `fasb/buffers/failure_buffer.py`

### API

```python
class FailureBuffer:
    def add(self, record: dict) -> None: ...
    def sample(self, n: int = 1) -> list[dict]: ...
    def sample_priority(self) -> dict: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "FailureBuffer": ...
    def __len__(self) -> int: ...
```

### Storage format

Use JSONL:

```text
runs/<exp>/failure_buffer.jsonl
```

Each record:

```json
{
  "seed": 1042,
  "scenario_id": "seed_1042",
  "failure_score": 7.0,
  "risk_score": 0.85,
  "failure_mode": "collision",
  "route_completion": 0.42,
  "success": false,
  "priority": 0.74,
  "metadata": {}
}
```

---

## 16. Training system

File: `fasb/training/sb3_trainer.py`

### Trainer responsibilities

- Load config.
- Build run directory.
- Save resolved config.
- Build env via MetaDrive factory.
- Load SB3 checkpoint if provided.
- Instantiate SB3 PPO if checkpoint absent.
- Attach callbacks.
- Train.
- Save checkpoints.
- Save final model.

### Fine-tuning mode support

Create method enum:

```text
base_eval
naive_ft
fixed_budget_ft
fasb_ppo
fasb_ppo_lagrangian_stretch
```

### Mode behavior

| Mode | Scenario sampler | Cost wrapper | Safety budget | Reward penalty |
|---|---|---|---|---|
| `base_eval` | eval only | optional logging | no | no |
| `naive_ft` | failure-aware or failure-only | logging only | no | no |
| `fixed_budget_ft` | mixed sampler | yes | fixed budget | fixed/risk-neutral penalty |
| `fasb_ppo` | mixed failure sampler | yes | adaptive budget | adaptive penalty |
| `fasb_ppo_lagrangian_stretch` | mixed failure sampler | yes | adaptive budget | lambda update |

---

## 17. Callbacks

File: `fasb/training/callbacks.py`

Use SB3 callbacks instead of custom training loops.

### Required callbacks

1. `EpisodeJSONLCallback`
   - Collect per-episode info from env.
   - Write to `episodes.jsonl`.

2. `FASBCheckpointCallback`
   - Save latest checkpoint.
   - Save best by success.
   - Save best by safety.
   - Save best by combined score.

3. `FailureBufferUpdateCallback`
   - During exploration or training, add useful failure records to buffer.

4. `FASBEvalCallback`
   - Run evaluation periodically using our evaluator.
   - Save eval metrics CSV.

Use SB3 `CheckpointCallback` and `EvalCallback` if possible; extend only where FASB-specific metrics are needed.

---

## 18. Evaluation system

File: `fasb/evaluation/evaluator.py`

### API

```python
class Evaluator:
    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        scenario_set: str,
        n_episodes: int,
        deterministic: bool = True,
    ) -> dict:
        ...
```

### Scenario sets

File: `fasb/evaluation/scenario_sets.py`

Support:

```text
heldout_random
 discovered_failures
heldout_failure_modes
train_random_debug
```

### Metrics

File: `fasb/evaluation/metrics.py`

Report:

```text
success_rate
collision_rate
offroad_rate
timeout_rate
route_completion_mean
episode_reward_mean
episode_modified_reward_mean
episode_cost_mean
cost_violation_rate
avg_episode_length
safety_efficiency_score
generalization_drop
```

Safety-efficiency score example:

\[
SES = success\_rate - \beta collision\_rate - \gamma offroad\_rate - \delta timeout\_rate
\]

Make coefficients configurable.

---

## 19. Failure-mode analysis

File: `fasb/analysis/failure_report.py`

Generate:

```text
runs/<exp>/analysis/failure_summary.csv
runs/<exp>/analysis/failure_by_mode.csv
runs/<exp>/analysis/failure_by_scenario_source.csv
runs/<exp>/analysis/before_after_table.csv
runs/<exp>/analysis/paper_numbers.md
runs/<exp>/analysis/*.png
```

Tables:

1. Overall method comparison.
2. Failure-mode breakdown.
3. Forgetting/generalization preservation.
4. Ablation comparison.

Example output table:

| Method | Success ↑ | Collision ↓ | Offroad ↓ | Timeout ↓ | Completion ↑ | Cost ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Base checkpoint | | | | | | |
| Naive FT | | | | | | |
| Fixed-budget FT | | | | | | |
| FASB-PPO | | | | | | |

---

## 20. Benchmark runner

File: `scripts/benchmark.py`

Runs:

```text
1. base checkpoint evaluation
2. failure exploration
3. naive fine-tuning
4. fixed-budget fine-tuning
5. FASB-PPO fine-tuning
6. evaluation on all scenario sets
7. analysis report generation
```

Command:

```bash
python scripts/benchmark.py --config configs/benchmark/final.yaml
```

Final output:

```text
runs/final_benchmark/
  configs/
  checkpoints/
  logs/
  eval/
  analysis/
    paper_numbers.md
    tables/
    plots/
```

---

## 21. Scripts

### `scripts/smoke_test_env.py`

Must test:

```text
raw MetaDrive env creation
reset
random action step
wrapped env step
DummyVecEnv creation
optional SubprocVecEnv creation
```

### `scripts/validate_components.py`

Must instantiate configured plugins and run synthetic validation inputs.

Output:

```text
PASS cost_function: DefaultDrivingCost
PASS failure_scorer: DefaultFailureScorer
PASS failure_classifier: DefaultFailureClassifier
PASS safety_budget: AdaptiveSafetyBudget
PASS sampler: MixedFailureSampler
```

If invalid:

```text
FAIL safety_budget: budget must be positive, got -0.4
```

### `scripts/explore_failures.py`

Runs checkpoint on scenario seeds, logs episode records, creates failure buffer.

### `scripts/train.py`

Runs selected mode from config.

### `scripts/evaluate.py`

Evaluates checkpoint using shared evaluator.

### `scripts/analyze_failures.py`

Generates analysis from run logs.

### `scripts/benchmark.py`

Runs full benchmark suite.

---

## 22. Run directory structure

Every run must look like:

```text
runs/<experiment_name>/
  config_resolved.yaml
  metadata.json
  checkpoints/
    latest.zip
    best_success.zip
    best_safety.zip
    best_score.zip
    final.zip
  logs/
    episodes.jsonl
    train_metrics.csv
    eval_metrics.csv
  buffers/
    failure_buffer.jsonl
  errors/
    plugin_errors.jsonl
    plugin_errors.log
  eval/
    heldout_random.csv
    discovered_failures.csv
    heldout_failure_modes.csv
  analysis/
    paper_numbers.md
    failure_summary.csv
    failure_by_mode.csv
    plots/
```

---

## 23. Tests and acceptance criteria

### Unit tests

Implement:

```text
test_component_validation.py
test_failure_buffer.py
test_plugin_loading.py
test_metrics.py
test_run_dir.py
```

### Integration tests

Implement, but mark slow tests if needed:

```text
test_env_smoke.py
```

### Minimum passing checks

Before declaring MVP complete:

```bash
pytest tests/test_component_validation.py
pytest tests/test_failure_buffer.py
pytest tests/test_plugin_loading.py
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=1000 vec_env.type=dummy vec_env.n_envs=1
python scripts/evaluate.py --config configs/eval/heldout_random.yaml eval.n_episodes=5
```

### Framework acceptance criteria

The framework is acceptable only if:

```text
1. It can evaluate a checkpoint and produce episode logs.
2. It can create a failure buffer from logs.
3. It can run naive fine-tuning using SB3 PPO.
4. It can run FASB-PPO with adaptive cost penalty.
5. It can save and reload checkpoints.
6. It can evaluate all methods with the same evaluator.
7. It can generate failure-mode tables.
8. Invalid plugin outputs fail fast and write error logs.
9. MetaDrive config pass-through works.
10. Plugin swaps work from YAML config.
```

---

## 24. Build phases for Codex

### Phase 0 — Inspect and scaffold

Codex tasks:

```text
1. Create repo structure.
2. Add pyproject.toml.
3. Add minimal README.md.
4. Add BUILD_PLAN.md copy of this file.
5. Add package imports.
```

Acceptance:

```bash
python -m compileall fasb
```

### Phase 1 — Schemas, validation, run dirs

Codex tasks:

```text
1. Implement dataclasses for records/outputs.
2. Implement validation helpers.
3. Implement error classes.
4. Implement run directory creation.
5. Implement JSONL writer.
```

Acceptance:

```bash
pytest tests/test_component_validation.py
pytest tests/test_run_dir.py
```

### Phase 2 — Plugin system

Codex tasks:

```text
1. Implement plugin protocols.
2. Implement built-in cost/scorer/classifier/budget/penalty/sampler plugins.
3. Implement Hydra/OmegaConf instantiation utility.
4. Implement plugin validation script.
```

Acceptance:

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
pytest tests/test_plugin_loading.py
```

### Phase 3 — MetaDrive env factory and wrappers

Codex tasks:

```text
1. Implement MetaDrive env factory with config pass-through.
2. Implement reset/step API compatibility wrapper.
3. Implement ScenarioMetadataWrapper.
4. Implement CostFunctionWrapper.
5. Implement AdaptiveRewardPenaltyWrapper.
6. Implement smoke test script.
```

Acceptance:

```bash
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
```

### Phase 4 — Failure exploration

Codex tasks:

```text
1. Implement evaluator-like rollout loop for checkpoint or random policy.
2. Implement episode record logging.
3. Implement failure scoring/classification during exploration.
4. Implement FailureBuffer save/load.
5. Implement explore_failures.py.
```

Acceptance:

```bash
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml eval.n_episodes=5
ls runs/*/buffers/failure_buffer.jsonl
```

### Phase 5 — SB3 training backend

Codex tasks:

```text
1. Implement SB3 trainer.
2. Support loading checkpoint if present.
3. Support creating PPO if checkpoint absent.
4. Implement callbacks for logging/checkpointing.
5. Implement train.py.
```

Acceptance:

```bash
python scripts/train.py --config configs/train/naive_ft.yaml training.total_timesteps=1000 vec_env.type=dummy vec_env.n_envs=1
```

### Phase 6 — FASB-PPO fine-tuning

Codex tasks:

```text
1. Implement mixed failure sampler integration.
2. Implement adaptive budget + penalty wrapper integration.
3. Support modes: naive_ft, fixed_budget_ft, fasb_ppo.
4. Ensure same trainer/evaluator works for all methods.
```

Acceptance:

```bash
python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=1000 vec_env.type=dummy vec_env.n_envs=1
```

### Phase 7 — Evaluation and analysis

Codex tasks:

```text
1. Implement Evaluator.
2. Implement driving metrics.
3. Implement scenario set configs.
4. Implement analyze_failures.py.
5. Implement plots/tables.
```

Acceptance:

```bash
python scripts/evaluate.py --config configs/eval/heldout_random.yaml eval.n_episodes=5
python scripts/analyze_failures.py --run runs/fasb_ppo
```

### Phase 8 — Benchmark runner

Codex tasks:

```text
1. Implement final benchmark runner.
2. Run all methods from one config.
3. Aggregate results into paper_numbers.md.
```

Acceptance:

```bash
python scripts/benchmark.py --config configs/benchmark/final.yaml --dry-run
```

---

## 25. README requirements

Create `README.md` with:

```text
1. What the framework does.
2. What it does not do.
3. Installation.
4. Quickstart.
5. Commands.
6. Config explanation.
7. Plugin authoring.
8. Experiment modes.
9. Output directory structure.
10. Troubleshooting.
```

README must be teammate-friendly.

---

## 26. Plugin authoring guide

Create `examples/plugin_authoring.md`.

Must include examples for:

```text
custom cost function
custom safety budget
custom failure scorer
custom sampler
```

Also include:

```text
how to validate custom plugin
what errors mean
how to enable custom imports
```

Example command:

```bash
python scripts/validate_components.py --config configs/train/my_custom_budget.yaml
```

---

## 27. Important engineering warnings

Do not build:

```text
custom PPO from scratch
custom vectorized env system
web dashboard
database server
vision/RGB pipeline
VLM/language interface
multi-agent RL backend
distributed training cluster
```

Do build:

```text
stable config-driven env construction
strict plugin system
failure buffer
adaptive reward penalty wrapper
SB3-compatible training
shared evaluator
failure-mode analysis
benchmark runner
```

---

## 28. Research experiment modes to support

### Baseline 1: Base checkpoint

No training. Evaluate only.

### Baseline 2: Naive failure fine-tuning

Fine-tune on failure scenarios, no adaptive budget.

Purpose: prove FASB is not just fine-tuning.

### Baseline 3: Fixed-budget fine-tuning

Use safety cost and one fixed penalty/budget.

Purpose: prove adaptive scenario-specific safety matters.

### Main: FASB-PPO

Use:

```text
failure-aware sampler
adaptive budget
adaptive penalty
same SB3 PPO backend
```

Purpose: proposed method.

---

## 29. Final expected research output

Framework should produce enough for the paper:

```text
1. Overall performance table.
2. Failure-mode performance table.
3. Forgetting/generalization preservation table.
4. Ablation table.
5. Plots for success/collision/offroad/timeout/completion.
6. Failure buffer summary.
7. Clear reproducible command list.
```

The paper claim supported by this framework:

> FASB specializes a generalized MetaDrive policy by discovering its failure modes, prioritizing learnable near-failure scenarios, and applying scenario-conditioned safety penalties/budgets during fine-tuning, improving safety on failure modes while preserving general driving ability.

---

## 30. Source references used for design

These are design references, not code dependencies:

- MetaDrive official site: https://metadriverse.github.io/metadrive/
- MetaDrive paper: https://arxiv.org/abs/2109.12674
- MetaDrive RL environments docs: https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html
- MetaDrive training docs: https://metadrive-simulator.readthedocs.io/en/latest/training.html
- MetaDrive config docs: https://metadrive-simulator.readthedocs.io/en/latest/config_system.html
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- SB3 callbacks: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
- SB3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- SB3 evaluation helper: https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
- Gymnasium wrappers: https://gymnasium.farama.org/api/wrappers/
- Hydra object instantiation: https://hydra.cc/docs/advanced/instantiate_objects/overview/
- OpenAI Codex CLI docs: https://developers.openai.com/codex/cli

---

## 31. Codex execution instruction

When Codex CLI reads this file:

1. Build in phases. Do not jump to training before scaffolding, validation, env smoke tests, and plugin validation exist.
2. Prefer small working commits over large rewrites.
3. Use existing frameworks. Do not rebuild PPO/SAC/env-vectorization/checkpoint basics.
4. Every phase must add or update tests.
5. If MetaDrive/SB3 API mismatch appears, add compatibility wrappers and document it.
6. Keep MVP focused on FASB-PPO. Put PPO-Lagrangian in stretch only.
7. Never silently ignore plugin errors.
8. Preserve config-driven extensibility.
9. Preserve MetaDrive config pass-through.
10. Make teammate commands easy and documented.
