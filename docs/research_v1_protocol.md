# Research V1 Protocol

## Purpose

This protocol prevents experiment drift across teammates. All five research axes should share the same environment setup, seed ranges, base checkpoint, failure buffer, first-pass budgets, metrics, and naming conventions unless the axis explicitly studies one of those variables.

## Shared Local Environment

Use the existing MetaDrive venv:

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
python scripts/run_e2e_stress.py --clean-runs
```

## Shared Root

```text
runs/research_v1/
```

## Shared Checkpoint

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
```

## Shared Base Checkpoint Quality Requirement

Minimum first-pass quality gates:

```text
eval_n_episodes >= 100
route_completion_mean >= 0.35
success_rate >= 0.10
safety_efficiency_score > random_baseline_safety_efficiency_score
```

Stronger target quality:

```text
route_completion_mean >= 0.50
success_rate >= 0.20
timeout_rate not equal to 1.0
```

If the minimum gate fails, do not start research axes. Continue training or adjust pretraining settings. The target gate is preferred, but the minimum gate is the hard stop.

## Shared Failure Buffer

```text
runs/research_v1/base_explore/buffers/failure_buffer.jsonl
```

Quality gate:

```text
line_count >= 30
at least 2 distinct failure modes if possible
not all records are solved
not all records are same seed
must contain seed, risk_score/failure_score, failure_mode, route_completion or failure stats
```

## Seed Ranges

```text
base pretraining:       start_seed=1000, num_scenarios=500
base checkpoint eval:   start_seed=4000, num_scenarios=200
failure exploration:    start_seed=0,    num_scenarios=500
fine-tuning training:   start_seed=2000, num_scenarios=500
heldout evaluation:     start_seed=5000, num_scenarios=200
dense traffic eval:     start_seed=7000, num_scenarios=200
easy traffic eval:      start_seed=8000, num_scenarios=200
```

## First-Pass Budget

```text
training.total_timesteps=100000 for axis runs
metadrive.config.horizon=500
eval.n_episodes=100
vec_env.type=dummy
vec_env.n_envs=1
algorithm.params.device=cpu
```

## Base Checkpoint Budget

For robust base checkpoint:

```text
preferred total_timesteps=1000000
minimum total_timesteps=300000
vec_env.type=subproc
vec_env.n_envs=4 when stable
```

If subproc has environment issues, use `dummy` with `n_envs=1`, but expect longer wall-clock.

## Shared Metrics

```text
success_rate
collision_rate
offroad_rate
timeout_rate
route_completion_mean
episode_cost_mean
cost_violation_rate
safety_efficiency_score
failure_by_mode
```

## Naming Convention

```text
runs/research_v1/<axis>_<variant>_s42/
runs/research_v1/eval_<axis>_<variant>_s42/
```

## What Not To Change

Unless the research axis explicitly studies it:

```text
eval seed range
training seed range
horizon
timesteps
traffic density
metrics
failure buffer path
base checkpoint path
```
