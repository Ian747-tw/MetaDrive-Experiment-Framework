# Axis 1 Reproduction Commands

These commands reproduce the newest Axis 1 6/12 multiseed research protocol. They intentionally use the stable configs and seed overrides from the newest runs.

## Environment

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
make validate-research-v1-artifacts
```

## Training

```bash
for seed in 2000 3000 4000 6000 7000 8000; do
  for method in naive fasb; do
    cfg="configs/research_v1/axis1_${method}_stable_final.yaml"
    run="multiseed_axis1_${method}_stable_seed${seed}"
    CUDA_VISIBLE_DEVICES= python scripts/train.py \
      --config "$cfg" \
      experiment.name="$run" \
      experiment.output_dir="runs/research_v1/${run}" \
      experiment.seed="$seed" \
      metadrive.config.start_seed="$seed" \
      sampler.start_seed="$seed" \
      training.total_timesteps=300000
  done
done
```

## Dev Checkpoint Selection

```bash
for seed in 2000 3000 4000 6000 7000 8000; do
  for method in naive fasb; do
    run="multiseed_axis1_${method}_stable_seed${seed}"
    python scripts/select_best_checkpoint.py \
      --run-dir "runs/research_v1/${run}" \
      --eval-config configs/eval/heldout_random.yaml \
      --output-dir "runs/research_v1/multiseed_axis1_select/${run}" \
      --metric safety_efficiency_score \
      --eval-start-seed 4500 \
      --eval-num-scenarios 100 \
      --eval-episodes 100 \
      --horizon 500 \
      --traffic-density 0.1
  done
done
```

## Final Heldout Evaluation

```bash
for seed in 2000 3000 4000 6000 7000 8000; do
  for method in naive fasb; do
    run="multiseed_axis1_${method}_stable_seed${seed}"
    eval_run="eval_${run}_selected_finalheldout"
    python scripts/evaluate.py \
      --config configs/eval/heldout_random.yaml \
      --checkpoint "runs/research_v1/${run}/checkpoints/selected_dev_best.zip" \
      experiment.name="$eval_run" \
      experiment.output_dir="runs/research_v1/${eval_run}" \
      eval.n_episodes=100 \
      metadrive.config.start_seed=5000 \
      metadrive.config.num_scenarios=200 \
      metadrive.config.horizon=500 \
      metadrive.config.traffic_density=0.1
    python scripts/analyze_failures.py --run "runs/research_v1/${eval_run}"
  done
done
```

## Result Aggregation

The current summary CSVs are in:

```text
results/summary/multiseed_axis1_per_seed.csv
results/summary/multiseed_axis1_summary.csv
results/summary/multiseed_axis1_paired_deltas.csv
```

The paper conclusion should be based on the six-seed/twelve-train block, not the older single-run diagnostic result.
