# FASB MetaDrive

FASB MetaDrive is a lightweight experiment framework for failure-aware fine-tuning of MetaDrive driving policies with Stable-Baselines3 PPO.

The repo is meant to be cloned or pulled, installed into a compatible Python environment, and used directly for experiments. Teammates should usually customize YAML configs and plugin classes, not edit the training loop.

## What This Is

- A reproducible MetaDrive experiment scaffold.
- A config-driven SB3 PPO training pipeline.
- Failure exploration, failure-buffer construction, evaluation, and analysis tooling.
- A plugin system for research ideas around costs, failure scoring, budgets, penalties, and sampling.

## What This Is Not

- Not a custom PPO implementation.
- Not a full PPO-Lagrangian implementation.
- Not a general RL framework, dashboard, database service, or distributed trainer.

## Current Method

The current implemented method is:

```text
FASB-PPO = SB3 PPO + failure-aware sampler + adaptive safety penalty
```

This is adaptive reward-penalty PPO. Full PPO-Lagrangian with a learned multiplier update is future/stretch work unless that update is explicitly implemented.

## Installation

### A. Recommended: Existing MetaDrive Venv

Use this path if you already have a MetaDrive-compatible environment.

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
```

`--no-deps` keeps pip from reinstalling MetaDrive or old transitive dependencies inside a venv that already works.

### B. Fresh Setup

Use Python 3.10 or 3.11 if possible. Python 3.12 may fail because `metadrive-simulator` can pull `gym==0.19.0`, which has known packaging issues on newer Python environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
python scripts/check_env.py --require-metadrive
```

If `pip install -e .` fails on MetaDrive or Gym, do not hide the failure. Use a MetaDrive-compatible environment, or install dependencies manually with compatible pins for your machine.

## First Validation Commands

Run these after installing or pulling major changes:

```bash
python scripts/check_env.py --require-metadrive
python -m pytest tests/test_component_validation.py tests/test_failure_buffer.py tests/test_metrics.py tests/test_training_stability_patch.py -q
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
python scripts/run_e2e_stress.py --clean-runs
```

Useful Make targets:

```bash
make check-env
make check-env-metadrive
make validate
make smoke
make stress
make e2e-stress
make test
```

`make stress` keeps the shorter command sequence. `make e2e-stress` runs the checked Python E2E stress script and cleans only the known stress run directories.

## Minimal Workflow

```bash
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml
python scripts/train.py --config configs/train/fasb_ppo.yaml
python scripts/evaluate.py --config configs/eval/heldout_random.yaml --checkpoint runs/fasb_ppo/checkpoints/final.zip
python scripts/analyze_failures.py --run runs/heldout_random_eval
```

Expected primary outputs:

```text
runs/base_explore/buffers/failure_buffer.jsonl
runs/fasb_ppo/checkpoints/final.zip
runs/fasb_ppo/logs/episodes.jsonl
runs/heldout_random_eval/eval/heldout_random.csv
runs/heldout_random_eval/analysis/
```

See [docs/experiment_workflow.md](docs/experiment_workflow.md) for a one-page teammate guide covering baselines, FASB-PPO, evaluation, and comparison outputs.

## Quick Overrides

Hydra-style CLI overrides let you change settings without editing YAML:

```bash
python scripts/train.py \
  --config configs/train/fasb_ppo.yaml \
  training.total_timesteps=10000 \
  vec_env.type=dummy \
  vec_env.n_envs=1 \
  algorithm.params.device=cpu
```

`training.total_timesteps=32` is only for stress tests. Real experiments need larger budgets and multiple seeds.

## Configs

Start with YAML configs:

```text
configs/explore/base_checkpoint.yaml
configs/train/naive_ft.yaml
configs/train/fixed_budget_ft.yaml
configs/train/fasb_ppo.yaml
configs/eval/heldout_random.yaml
configs/benchmark/final.yaml
configs/components/
```

MetaDrive settings live under:

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

Everything under `metadrive.config` is passed directly to MetaDrive.

## Plugins

Built-in research extension points:

```text
cost_function
failure_scorer
failure_classifier
safety_budget
penalty_scheduler
sampler
```

Plugins are Python classes loaded from YAML `_target_` fields. Plugin outputs must return dataclasses from `fasb.schemas.outputs`; invalid outputs fail fast during validation or runtime checks. Plugin errors are logged under `runs/<experiment>/errors/`.

Read [examples/plugin_authoring.md](examples/plugin_authoring.md) before adding a plugin.

Validate changes with:

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
```

## Tests And CI

Default CI intentionally avoids installing MetaDrive. It installs the framework with `--no-deps`, adds lightweight test dependencies, compiles `fasb` and `tests`, runs component/plugin/config tests, and validates the FASB-PPO config.

Local MetaDrive checks are still required before running real experiments:

```bash
python scripts/check_env.py --require-metadrive
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
python scripts/run_e2e_stress.py --clean-runs
```

The GitHub Actions workflow also includes a manual `workflow_dispatch` MetaDrive smoke job. It is experimental and not required for normal pull request CI.

## Troubleshooting

If MetaDrive does not import, use a known compatible venv first. Python 3.10/3.11 may be safer than Python 3.12 because of old Gym packaging constraints pulled by MetaDrive.

If component validation fails, check YAML `_target_` paths and plugin return dataclasses.

If training cannot find failures, run `explore_failures.py` first and confirm `runs/base_explore/buffers/failure_buffer.jsonl` exists.

If evaluation fails on the checkpoint path, confirm `runs/fasb_ppo/checkpoints/final.zip` exists and was produced by the same compatible environment.

Do not commit generated artifacts from `runs/`, `.venv/`, caches, checkpoints, TensorBoard, or W&B output.
