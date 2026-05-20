# FASB MetaDrive

FASB MetaDrive is a lightweight experiment framework for failure-aware, safety-budgeted fine-tuning of MetaDrive driving policies with Stable-Baselines3.

The framework is meant to be cloned, configured, and extended. Most users should customize YAML configs and small plugin classes instead of editing the training loop.

The current main method is **FASB-PPO: SB3 PPO + failure-aware sampler + adaptive safety penalty**. It is an adaptive reward-penalty PPO workflow, not a full PPO-Lagrangian implementation. PPO-Lagrangian with a learned multiplier update is a stretch/future direction unless that update is explicitly implemented.

It provides:

- MetaDrive environment construction with pass-through simulator config
- Strict plugin validation for costs, failure scoring, failure labels, safety budgets, penalties, and samplers
- Failure exploration and JSONL failure buffers
- SB3 PPO training and checkpointing
- Shared evaluation, metrics, and failure analysis outputs

It does not provide a custom PPO implementation, a dashboard, a database server, distributed training, or a full RL library.

## First-Time Setup

Clone the repo:

```bash
git clone git@github.com:Ian747-tw/MetaDrive-Experiment-Framework.git
cd MetaDrive-Experiment-Framework
```

Create a project-local Python environment and install the framework:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
```

Why two commands?

- `./scripts/bootstrap.sh` creates `.venv` and installs this project plus dependencies.
- `source .venv/bin/activate` activates that environment in your current terminal. Shell activation inside a script cannot persist after the script exits.

For later terminal sessions, only run:

```bash
source .venv/bin/activate
```

## Recommended Workflow

Run this sequence when starting a new experiment:

```bash
make validate
make smoke
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml
python scripts/train.py --config configs/train/fasb_ppo.yaml
python scripts/evaluate.py --config configs/eval/heldout_random.yaml --checkpoint runs/fasb_ppo/checkpoints/final.zip
python scripts/analyze_failures.py --run runs/heldout_random_eval
```

For a short end-to-end stress run:

```bash
make stress
```

For a dry run of the full benchmark sequence:

```bash
make benchmark-dry-run
```

## Tool Commands

### Validate Plugins

Checks that configured plugins instantiate correctly and return valid outputs.

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
```

Shortcut:

```bash
make validate
```

Use this after changing any plugin or `_target_` path.

### Smoke-Test MetaDrive

Creates a MetaDrive env, resets it, takes one action, applies wrappers, and checks `DummyVecEnv`.

```bash
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
```

Shortcut:

```bash
make smoke
```

Use this after changing `metadrive.config`, env wrappers, or environment setup.

### Explore Failures

Runs a checkpoint or random policy through scenarios, writes episode logs, scores failures, and creates a failure buffer.

```bash
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml
```

Outputs:

```text
runs/base_explore/logs/episodes.jsonl
runs/base_explore/buffers/failure_buffer.jsonl
```

For a quick run:

```bash
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml eval.n_episodes=5 metadrive.config.horizon=100
```

### Train

Runs SB3 PPO with the configured experiment mode.

```bash
python scripts/train.py --config configs/train/fasb_ppo.yaml
```

Outputs:

```text
runs/fasb_ppo/config_resolved.yaml
runs/fasb_ppo/logs/episodes.jsonl
runs/fasb_ppo/checkpoints/final.zip
```

For a quick sanity run:

```bash
python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=1000 vec_env.n_envs=1
```

### Evaluate

Evaluates a checkpoint with the shared evaluator.

```bash
python scripts/evaluate.py \
  --config configs/eval/heldout_random.yaml \
  --checkpoint runs/fasb_ppo/checkpoints/final.zip
```

Output:

```text
runs/heldout_random_eval/eval/heldout_random.csv
```

### Analyze

Generates tables and paper-number summaries from run logs.

```bash
python scripts/analyze_failures.py --run runs/heldout_random_eval
```

Outputs:

```text
runs/heldout_random_eval/analysis/failure_summary.csv
runs/heldout_random_eval/analysis/failure_by_mode.csv
runs/heldout_random_eval/analysis/paper_numbers.md
```

### Benchmark

Prints or runs the full benchmark sequence.

```bash
python scripts/benchmark.py --config configs/benchmark/final.yaml --dry-run
```

Shortcut:

```bash
make benchmark-dry-run
```

## Customizing Experiments

Start with YAML configs. You usually do not need to edit Python code for common experiment changes.

Important config files:

```text
configs/env/metadrive_debug.yaml             small env smoke-test config
configs/env/metadrive_generalization.yaml    larger generalization env config
configs/explore/base_checkpoint.yaml         failure discovery config
configs/train/naive_ft.yaml                  baseline fine-tuning config
configs/train/fixed_budget_ft.yaml           fixed safety penalty baseline
configs/train/fasb_ppo.yaml                  main FASB-PPO config
configs/eval/heldout_random.yaml             heldout evaluation config
configs/benchmark/final.yaml                 benchmark sequence config
configs/components/*                         reusable plugin configs
```

MetaDrive settings must stay under:

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

Everything under `metadrive.config` is passed to MetaDrive directly.

You can override config values from the command line:

```bash
python scripts/train.py \
  --config configs/train/fasb_ppo.yaml \
  training.total_timesteps=5000 \
  metadrive.config.traffic_density=0.2 \
  safety_budget.d_min=0.01
```

## Customizing Plugins

Plugins are small Python classes referenced from YAML with `_target_`.

Built-in plugin types:

```text
cost_function          computes safety cost per step
failure_scorer         scores episode failure severity
failure_classifier     assigns failure mode labels
safety_budget          chooses scenario safety budget
penalty_scheduler      converts risk/budget into reward penalty
sampler                chooses scenario seeds
```

Built-in implementations live in:

```text
fasb/plugins/cost.py
fasb/plugins/failure_scorer.py
fasb/plugins/failure_classifier.py
fasb/plugins/safety_budget.py
fasb/plugins/penalty_scheduler.py
fasb/plugins/sampler.py
```

Example custom plugins live in:

```text
examples/custom_plugins/
```

To use a custom cost function:

```yaml
cost_function:
  _target_: examples.custom_plugins.cautious_distance_cost.CautiousDistanceCost
  threshold: 5.0
```

Then validate it:

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
```

Plugin outputs are validated before they affect training. Invalid outputs raise `ComponentValidationError` instead of silently continuing.

More examples are in:

```text
examples/plugin_authoring.md
```

### What Is Config-Driven?

The training loop stays small and delegates experiment behavior to configured components. These are plugin-driven through YAML `_target_` entries:

- `cost_function`
- `failure_scorer`
- `failure_classifier`
- `safety_budget`
- `penalty_scheduler`
- `sampler`

Sampler construction is also config-driven: `sampler._target_` chooses the sampler class, while the trainer injects runtime seed shards and the configured failure-buffer path when needed.

### Error Logs

Plugin failures are fail-fast, but the framework writes context first. During training, wrapper plugin errors are recorded under:

```text
runs/<experiment>/errors/plugin_errors.jsonl
runs/<experiment>/errors/plugin_errors.log
```

The JSONL file is intended for scripted inspection. The `.log` file contains readable tracebacks for debugging.

## Output Structure

Each run writes a resolved config and artifacts:

```text
runs/<experiment>/
  config_resolved.yaml
  metadata.json
  checkpoints/
    final.zip
  logs/
    episodes.jsonl
  buffers/
    failure_buffer.jsonl
  errors/
    plugin_errors.jsonl
    plugin_errors.log
  eval/
    heldout_random.csv
  analysis/
    failure_summary.csv
    failure_by_mode.csv
    paper_numbers.md
```

Generated `runs/` data is ignored by git.

## Common Development Checks

```bash
make test
make validate
make smoke
```

Equivalent direct commands:

```bash
python -m pytest tests/test_component_validation.py tests/test_failure_buffer.py tests/test_plugin_loading.py tests/test_metrics.py tests/test_run_dir.py -q
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
```

## Troubleshooting

If `python` cannot import `fasb`, activate the project environment:

```bash
source .venv/bin/activate
```

If MetaDrive import or env creation fails, rerun setup:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
make smoke
```

If a plugin fails validation, check:

- The YAML `_target_` import path is correct.
- The plugin returns the expected dataclass from `fasb.schemas.outputs`.
- Numeric values are finite and within the documented range.

If training is slow, start with shorter overrides:

```bash
python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=1000 metadrive.config.horizon=100
```
