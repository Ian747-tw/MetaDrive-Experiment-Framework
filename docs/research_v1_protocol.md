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
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

Quality gate:

```text
line_count >= 1000
at least 2 distinct failure modes
unknown failure_mode fraction <= 0.25
not all records are solved
not all records are same seed
must contain seed, risk_score/failure_score, failure_mode, route_completion or failure stats
```

## Seed Ranges

```text
base pretraining:       start_seed=1000, num_scenarios=500
base checkpoint eval:   start_seed=4000, num_scenarios=200
failure exploration:    start_seed=0,    num_scenarios=3000
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

## Locked Settings Vs Axis Variables

These settings are locked for all main/final comparisons.

Base checkpoint:

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
```

Final canonical failure buffer:

```text
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

Fine-tuning training:

```text
training.total_timesteps=300000 for final runs
metadrive.config.start_seed=2000
metadrive.config.num_scenarios=500
metadrive.config.horizon=500
metadrive.config.traffic_density=0.1
vec_env.type=dummy or subproc, but must be same within a comparison
vec_env.n_envs=1 or 4, but must be same within a comparison
algorithm.params.device=cpu
```

Heldout evaluation:

```text
eval.n_episodes=100
metadrive.config.start_seed=5000
metadrive.config.num_scenarios=200
metadrive.config.horizon=500
metadrive.config.traffic_density=0.1
```

Axis researchers may not change these unless their axis explicitly studies that variable. If someone changes basic PPO or environment settings for stability, they must apply the same change to all compared methods and document it. Main final claims require the same base checkpoint, same final buffer, same train/eval seeds, and same timesteps.

Axis variables:

```text
Axis 1: method: naive_ft vs fixed_budget_ft vs full FASB-PPO
Axis 2: sampler._target_, sampler.failure_ratio, sampler.alpha
Axis 3: safety_budget.*, penalty_scheduler.*
Axis 4: cost_function.*
Axis 5: failure_scorer.*, failure_classifier.*, eval traffic_density, and axis-specific buffers only if explicitly studying buffer/distribution shift
```

Do not compare FASB 300k vs Naive 100k. That proves nothing.

## Shared Artifact Distribution

Required release:

```text
research-v1-foundation-v1
```

Assets:

```text
research_v1_foundation_artifacts.tar.gz
research_v1_artifact_manifest.json
```

Extracted paths:

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
runs/research_v1/eval_base_pretrain/eval/heldout_random.csv
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

Verification command:

```bash
make validate-research-v1-artifacts
```

Axis 1-4 must use the release-provided checkpoint and canonical large buffer. Axis 5 may use axis-specific buffers only when the research question explicitly studies buffer or distribution shift.
