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
The E2E stress script writes to `runs/e2e_stress/` so it does not remove canonical experiment outputs such as `runs/fasb_ppo/`.

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

## Research V1 Foundation

Research V1 separates committed source foundation from generated experiment artifacts.

Committed source foundation:

```text
docs/research_v1_protocol.md
docs/research_v1_commands.md
configs/research_v1/
examples/custom_plugins/
scripts/aggregate_results.py
scripts/check_base_checkpoint_quality.py
scripts/check_failure_buffer_quality.py
scripts/check_research_v1_ready.py
```

Generated local experiment foundation:

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
runs/research_v1/eval_base_pretrain/eval/heldout_random.csv
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

Generated artifacts under `runs/` are not committed. Teammates must not start the five research axes until the readiness checker passes.

Make targets:

```bash
make research-v1-base-train
make research-v1-base-eval
make research-v1-build-buffer
make research-v1-check
```

Manual equivalents:

```bash
python scripts/train.py --config configs/research_v1/base_pretrain.yaml
python scripts/evaluate.py --config configs/research_v1/base_eval.yaml --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip
python scripts/explore_failures.py --config configs/research_v1/base_explore_large.yaml
python scripts/check_research_v1_ready.py --min-failures 1000
```

The base checkpoint must pass the minimum success and route-completion thresholds in `scripts/check_base_checkpoint_quality.py`. The failure buffer must have enough records and basic diversity according to `scripts/check_failure_buffer_quality.py`.

Detailed protocol and copy-paste axis commands:

```text
docs/research_v1_protocol.md
docs/research_v1_commands.md
```

## Research V1 Final Readiness

There are two foundations:

```text
source foundation: docs, configs, scripts, tests, and example plugins committed to git
local experiment foundation: checkpoint and failure buffers generated under runs/ and not committed
```

Use the known MetaDrive environment when available:

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
```

Fresh installs can work, but MetaDrive and old Gym packaging are sensitive to Python version and dependency pins. If a fresh install fails, use a known compatible MetaDrive venv instead of changing framework code.

Final validation:

```bash
python scripts/check_env.py --require-metadrive
python scripts/run_e2e_stress.py --clean-runs
python scripts/check_research_v1_ready.py \
  --root runs/research_v1 \
  --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip \
  --eval-csv runs/research_v1/eval_base_pretrain/eval/heldout_random.csv \
  --buffer runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl \
  --min-failures 1000 \
  --min-episodes 100
```

Common operations:

```bash
python scripts/train.py --config configs/research_v1/axis1_fasb_final.yaml
CUDA_VISIBLE_DEVICES= python scripts/explore_failures.py --config configs/research_v1/base_explore_large.yaml
python scripts/evaluate.py --config configs/research_v1/base_eval.yaml --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip
python scripts/analyze_failures.py --run runs/research_v1/eval_base_pretrain
python scripts/aggregate_results.py --root runs/research_v1
```

Research customization should stay plugin/config-driven. Users should customize plugin classes and YAML `_target_` fields; normal research should not edit the trainer. Axis researchers only change the variables assigned to their axis. Basic PPO, environment, seed, horizon, traffic, buffer, checkpoint, and evaluation settings stay locked for comparability unless the axis explicitly studies that variable.

Final guides:

```text
docs/research_v1_final_runbook.md
docs/research_v1_axis_specs.md
docs/research_v1_results.md
docs/research_v1_high_level_plan.md
```

## Research V1 Shared Artifacts

Axis 1-4 must use the same base checkpoint and canonical large failure buffer. These files are generated artifacts and are not committed to git; they are distributed through GitHub Release assets. Source code, docs, configs, and scripts live in git. Generated checkpoint, eval CSV, and failure buffer files live in the release archive.

Required release:

```text
research-v1-foundation-v1
```

Download flow for teammates:

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
git checkout main
git pull origin main
pip install -e . --no-deps

python scripts/download_research_v1_artifacts.py --release research-v1-foundation-v1
```

Manual fallback:

```bash
gh release download research-v1-foundation-v1 \
  --pattern "research_v1_foundation_artifacts.tar.gz"

tar -xzf research_v1_foundation_artifacts.tar.gz

python scripts/check_research_v1_ready.py \
  --root runs/research_v1 \
  --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip \
  --eval-csv runs/research_v1/eval_base_pretrain/eval/heldout_random.csv \
  --buffer runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl \
  --min-failures 1000 \
  --min-episodes 100 \
  --min-success-rate 0.10 \
  --min-route-completion 0.35 \
  --max-timeout-rate 0.95
```

Do not run Axis 1-4 against different locally regenerated checkpoints or buffers unless the experiment explicitly studies checkpoint or buffer construction. That breaks comparability.

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
