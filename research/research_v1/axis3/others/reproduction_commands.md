# Axis 3 Reproduction Commands

This note records the commands used to reproduce the Axis 3 experiments and to regenerate the key artifacts.

All commands below assume you run them from the repository root.

## Environment (ws7)

```bash
cd /tmp2/b14902068/MetaDrive-Experiment-Framework
source /tmp2/b14902068/.venv/bin/activate
python scripts/check_env.py --require-metadrive
```

## Train (Single Run)

### Fixed budget (example: 0.05)

```bash
python scripts/train.py \
  --config configs/research_v1/axis3_budget_fixed_default_final.yaml \
  experiment.name=axis3_fixed005_s42 \
  experiment.output_dir=runs/research_v1/axis3_fixed005_s42 \
  training.total_timesteps=300000 \
  training.save_every_steps=50000 \
  safety_budget.mode=fixed \
  safety_budget.budget=0.05 \
  experiment.seed=42
```

### Adaptive (example: lambda_max=0.10)

```bash
python scripts/train.py \
  --config configs/research_v1/axis3_budget_adaptive_default_final.yaml \
  experiment.name=axis3_adaptive_lmax010_s42 \
  experiment.output_dir=runs/research_v1/axis3_adaptive_lmax010_s42 \
  training.total_timesteps=300000 \
  training.save_every_steps=50000 \
  penalty_scheduler.lambda_min=0.0 \
  penalty_scheduler.lambda_max=0.10 \
  experiment.seed=42
```

## Evaluate (Heldout Random)

```bash
python scripts/evaluate.py \
  --config configs/eval/heldout_random.yaml \
  --checkpoint runs/research_v1/<train_run>/checkpoints/final.zip \
  experiment.name=eval_<train_run> \
  experiment.output_dir=runs/research_v1/eval_<train_run> \
  eval.n_episodes=100 \
  metadrive.config.start_seed=5000 \
  metadrive.config.num_scenarios=200 \
  metadrive.config.horizon=500
```

Outputs:

```text
runs/research_v1/eval_<train_run>/eval/heldout_random.csv
runs/research_v1/eval_<train_run>/logs/episodes.jsonl
```

## Failure Analysis

```bash
python scripts/analyze_failures.py --run runs/research_v1/eval_<train_run>
```

Outputs:

```text
runs/research_v1/eval_<train_run>/analysis/failure_summary.csv
runs/research_v1/eval_<train_run>/analysis/failure_by_mode.csv
runs/research_v1/eval_<train_run>/analysis/paper_numbers.md
```

Batch (all Axis 3 eval runs):

```bash
for d in runs/research_v1/eval_axis3_*_rep??; do
  python scripts/analyze_failures.py --run "$d"
done
```

## Batch Suite (6 variants x 10 reps)

This runs: `fixed003`, `fixed005`, `fixed010`, `adaptive_default`, `adaptive_strict`, `adaptive_loose`.

```bash
python scripts/batch_axis3_suite.py --reps 10 --workers 6 --train-timesteps 100000
```

Outputs:

```text
runs/research_v1/axis3_suite_summary_s42_<timestamp>/results.csv
runs/research_v1/axis3_suite_summary_s42_<timestamp>/best_overall.txt
```

