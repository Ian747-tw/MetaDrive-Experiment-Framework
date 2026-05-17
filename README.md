# FASB MetaDrive

FASB MetaDrive is a lightweight experiment layer for failure-aware, safety-budgeted fine-tuning of MetaDrive driving policies with Stable-Baselines3.

It provides config-driven environment creation, strict plugin validation, failure buffers, shared training/evaluation entrypoints, and analysis tables. It does not implement PPO, vectorized env internals, a dashboard, a database, or a distributed training system.

## Install

Clone the repo and install it into a local virtual environment:

```bash
git clone git@github.com:Ian747-tw/MetaDrive-Experiment-Framework.git
cd MetaDrive-Experiment-Framework

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or run the bootstrap helper:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
```

The `~/metadrive/.venv` environment used during development was only a local shortcut. New users should use the project-local `.venv` above unless they already maintain a separate MetaDrive environment.

## Customize

Most changes should happen in YAML configs or plugin classes:

```text
configs/train/fasb_ppo.yaml                  main FASB training settings
configs/env/metadrive_debug.yaml             quick simulator settings
configs/components/*                         reusable plugin configs
examples/custom_plugins/                     examples to copy and edit
```

Create a custom plugin by adding a Python class and pointing YAML at it:

```yaml
cost_function:
  _target_: examples.custom_plugins.cautious_distance_cost.CautiousDistanceCost
  threshold: 5.0
```

## Quickstart

```bash
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml eval.n_episodes=5
python scripts/train.py --config configs/train/fasb_ppo.yaml training.total_timesteps=1000 vec_env.n_envs=1
python scripts/evaluate.py --config configs/eval/heldout_random.yaml eval.n_episodes=5
python scripts/analyze_failures.py --run runs/fasb_ppo
python scripts/benchmark.py --config configs/benchmark/final.yaml --dry-run
```

The same commands are available through `make`:

```bash
make validate
make smoke
make stress
make benchmark-dry-run
```

## Configuration

MetaDrive settings live under `metadrive.config` and are passed to the configured env class unchanged. Research components use Hydra `_target_` paths and are validated before affecting training.

Experiment outputs follow:

```text
runs/<experiment>/
  config_resolved.yaml
  metadata.json
  checkpoints/
  logs/
  buffers/
  errors/
  eval/
  analysis/
```

## Plugin Authoring

Implement one of the protocols in `fasb.plugins.base`, expose it by import path, and reference it in YAML with `_target_`. Run `scripts/validate_components.py` before training. Invalid output raises `ComponentValidationError` and records a JSONL error when run through the framework logging helpers.
