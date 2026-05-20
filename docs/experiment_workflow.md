# Experiment Workflow

This is the short teammate guide for running the framework from a cloned repo.

## Before You Run

Activate a compatible environment, install the repo, and validate it:

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
python scripts/validate_components.py --config configs/train/fasb_ppo.yaml
python scripts/smoke_test_env.py --config configs/env/metadrive_debug.yaml
```

## Base Failure Exploration

Build the failure buffer used by failure-aware methods:

```bash
python scripts/explore_failures.py --config configs/explore/base_checkpoint.yaml
```

Expected output:

```text
runs/base_explore/
runs/base_explore/buffers/failure_buffer.jsonl
runs/base_explore/logs/episodes.jsonl
```

## Naive Baseline

Run ordinary SB3 PPO fine-tuning with uniform sampling:

```bash
python scripts/train.py --config configs/train/naive_ft.yaml
```

Expected output:

```text
runs/naive_ft/
runs/naive_ft/checkpoints/final.zip
```

## Fixed-Budget Baseline

Run the fixed safety-penalty/budget baseline:

```bash
python scripts/train.py --config configs/train/fixed_budget_ft.yaml
```

Expected output:

```text
runs/fixed_budget_ft/
runs/fixed_budget_ft/checkpoints/final.zip
```

## FASB-PPO

Run the current main method, adaptive-penalty FASB-PPO:

```bash
python scripts/train.py --config configs/train/fasb_ppo.yaml
```

Expected output:

```text
runs/fasb_ppo/
runs/fasb_ppo/checkpoints/final.zip
runs/fasb_ppo/logs/episodes.jsonl
```

## Evaluate

Evaluate a checkpoint on the heldout random set:

```bash
python scripts/evaluate.py \
  --config configs/eval/heldout_random.yaml \
  --checkpoint runs/fasb_ppo/checkpoints/final.zip
```

Expected output:

```text
runs/heldout_random_eval/
runs/heldout_random_eval/eval/heldout_random.csv
```

Swap the checkpoint path to compare baselines:

```bash
--checkpoint runs/naive_ft/checkpoints/final.zip
--checkpoint runs/fixed_budget_ft/checkpoints/final.zip
--checkpoint runs/fasb_ppo/checkpoints/final.zip
```

## Analyze And Compare

Generate summary tables:

```bash
python scripts/analyze_failures.py --run runs/heldout_random_eval
```

Important result files:

```text
runs/heldout_random_eval/eval/heldout_random.csv
runs/heldout_random_eval/analysis/failure_summary.csv
runs/heldout_random_eval/analysis/failure_by_mode.csv
runs/heldout_random_eval/analysis/paper_numbers.md
```

For quick local verification, run:

```bash
python scripts/run_e2e_stress.py --clean-runs
```

The stress settings are intentionally tiny. Do not use them as experiment results.

## What Not To Commit

Do not commit generated artifacts:

```text
runs/
.venv/
__pycache__/
*.pyc
.pytest_cache/
wandb/
checkpoints/
tensorboard/
model checkpoints
```
