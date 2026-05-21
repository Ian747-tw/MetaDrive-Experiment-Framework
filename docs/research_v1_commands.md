# Research V1 Commands

## 1. Shared Setup

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
python scripts/run_e2e_stress.py --clean-runs
```

## 2. Train Robust Base Checkpoint

```bash
python scripts/train.py \
  --config configs/research_v1/base_pretrain.yaml
```

## 3. Evaluate Base Checkpoint Quality

```bash
python scripts/evaluate.py \
  --config configs/research_v1/base_eval.yaml
```

```bash
python scripts/check_base_checkpoint_quality.py \
  --eval-csv runs/research_v1/eval_base_pretrain/eval/heldout_random.csv \
  --min-success-rate 0.10 \
  --min-route-completion 0.35
```

## 4. Continue Base Training If Quality Gate Fails

```bash
python scripts/train.py \
  --config configs/research_v1/base_pretrain.yaml \
  algorithm.checkpoint_path=runs/research_v1/base_pretrain_s42/checkpoints/final.zip \
  training.total_timesteps=300000
```

Then rerun evaluation and the base checkpoint quality check.

## 5. Build Shared Failure Buffer

```bash
python scripts/explore_failures.py \
  --config configs/research_v1/base_explore.yaml
```

## 6. Validate Shared Failure Buffer

```bash
python scripts/check_failure_buffer_quality.py \
  --buffer runs/research_v1/base_explore/buffers/failure_buffer.jsonl \
  --min-records 30
```

```bash
python scripts/check_research_v1_ready.py --min-failures 30
```

## 7. Axis 1 Commands - Main Baselines

```bash
python scripts/train.py --config configs/research_v1/axis1_naive.yaml
python scripts/train.py --config configs/research_v1/axis1_fixed_budget.yaml
python scripts/train.py --config configs/research_v1/axis1_fasb.yaml
```

## 8. Axis 2 Commands - Sampler Ablation

```bash
python scripts/train.py --config configs/research_v1/axis2_sampler_uniform.yaml
python scripts/train.py --config configs/research_v1/axis2_sampler_mixed030.yaml
python scripts/train.py --config configs/research_v1/axis2_sampler_mixed060.yaml
python scripts/train.py --config configs/research_v1/axis2_sampler_mixed090.yaml
python scripts/train.py --config configs/research_v1/axis2_sampler_mixed100.yaml
```

## 9. Axis 3 Commands - Budget/Penalty Ablation

```bash
python scripts/train.py --config configs/research_v1/axis3_budget_fixed003.yaml
python scripts/train.py --config configs/research_v1/axis3_budget_fixed005.yaml
python scripts/train.py --config configs/research_v1/axis3_budget_fixed010.yaml
python scripts/train.py --config configs/research_v1/axis3_budget_adaptive_default.yaml
python scripts/train.py --config configs/research_v1/axis3_budget_adaptive_strict.yaml
python scripts/train.py --config configs/research_v1/axis3_budget_adaptive_loose.yaml
```

## 10. Axis 4 Commands - Cost-Function Ablation

```bash
python scripts/train.py --config configs/research_v1/axis4_cost_crash_only.yaml
python scripts/train.py --config configs/research_v1/axis4_cost_default.yaml
python scripts/train.py --config configs/research_v1/axis4_cost_nearmiss_heavy.yaml
```

## 11. Axis 5 Commands - Failure Scorer/Generalization

```bash
python scripts/train.py --config configs/research_v1/axis5_near_failure_scorer.yaml
```

## 12. Evaluation Template

```bash
python scripts/evaluate.py \
  --config configs/research_v1/base_eval.yaml \
  --checkpoint runs/research_v1/<axis>_<variant>_s42/checkpoints/final.zip \
  experiment.name=eval_<axis>_<variant>_s42 \
  experiment.output_dir=runs/research_v1/eval_<axis>_<variant>_s42 \
  metadrive.config.start_seed=5000 \
  metadrive.config.num_scenarios=200 \
  eval.n_episodes=100
```

## 13. Analysis Template

```bash
python scripts/analyze_failures.py --run runs/research_v1/eval_<axis>_<variant>_s42
```

## 14. Aggregation Template

```bash
python scripts/aggregate_results.py --root runs/research_v1
python scripts/aggregate_results.py --root runs/research_v1 --output runs/research_v1/summary_main_results.csv
```

## 15. What Files Each Teammate Must Send Back

```text
runs/research_v1/<axis>_<variant>_s42/config.yaml
runs/research_v1/eval_<axis>_<variant>_s42/eval/heldout_random.csv
runs/research_v1/eval_<axis>_<variant>_s42/analysis/failure_by_mode.csv
runs/research_v1/eval_<axis>_<variant>_s42/analysis/paper_numbers.md
```
