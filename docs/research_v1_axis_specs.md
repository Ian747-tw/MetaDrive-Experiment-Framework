# Research V1 Axis Specs

## Shared Final Rules

All final comparisons use the same base checkpoint, canonical large failure buffer, 300k fine-tuning steps, train seed range, eval seed range, horizon, traffic density, metrics, and analysis process. Axis researchers may only change the variables listed for their axis.

Locked final paths:

```text
base checkpoint: runs/research_v1/base_pretrain_s42/checkpoints/final.zip
large buffer:    runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

## Axis 1 - Main Comparison

Goal: show FASB-PPO beats normal PPO fine-tuning and fixed-budget fine-tuning under the same base checkpoint, buffer, timesteps, train/eval seeds, horizon, optimizer regime, and checkpoint-selection policy.

Methods:

```text
Base checkpoint eval
Naive FT stable protocol 300k
Fixed-budget FT stable protocol 300k
Original FASB-PPO 300k
Stable FASB-PPO 300k
```

Allowed variables: `mode`, method config, sampler policy, and method-specific safety-budget behavior. Not allowed: base checkpoint, large buffer path, total timesteps, train/eval seeds, horizon, traffic density, optimizer regime, or checkpoint-selection policy.

The historical `configs/research_v1/axis1_fasb_final.yaml` is retained as the original collapsed default. The stabilized default is `configs/research_v1/axis1_fasb_stable_final.yaml`, selected after 300k dev calibration on `start_seed=4500`, `num_scenarios=100`, without using the final heldout range for selection. It uses gentler failure replay and penalty settings:

```text
sampler.failure_ratio=0.05
safety_budget.d_min=0.10
safety_budget.d_max=0.30
safety_budget.timeout_budget=0.30
penalty_scheduler.lambda_min=0.0
penalty_scheduler.lambda_max=0.25
algorithm.params.learning_rate=0.00003
```

Axis 2 and Axis 3 ablations should vary sampler and budget/penalty settings around this calibrated stable default while continuing to report the original collapse as a diagnostic result.

The fair Axis 1 comparison uses the same stable optimizer and dev checkpoint-selection protocol for all fine-tuned methods. Use:

```text
configs/research_v1/axis1_naive_stable_final.yaml
configs/research_v1/axis1_fixed_budget_stable_final.yaml
configs/research_v1/axis1_fasb_stable_final.yaml
```

Old configs with `linear:3.0e-4` are historical diagnostics, not the fair stable-protocol comparison.

Commands:

```bash
python scripts/train.py --config configs/research_v1/axis1_naive_stable_final.yaml
python scripts/train.py --config configs/research_v1/axis1_fixed_budget_stable_final.yaml
python scripts/train.py --config configs/research_v1/axis1_fasb_final.yaml
python scripts/train.py --config configs/research_v1/axis1_fasb_stable_final.yaml
```

Select dev-best checkpoints for each stable-protocol fine-tuned method before final heldout:

```bash
python scripts/select_best_checkpoint.py \
  --run-dir runs/research_v1/<method>_s42 \
  --eval-config configs/eval/heldout_random.yaml \
  --output-dir runs/research_v1/stabilization/select_<method>_s42 \
  --metric safety_efficiency_score \
  --eval-start-seed 4500 --eval-num-scenarios 100 \
  --eval-episodes 100 --horizon 500 --traffic-density 0.1
```

Evaluate selected checkpoints:

```bash
python scripts/evaluate.py --config configs/eval/heldout_random.yaml \
  --checkpoint runs/research_v1/<method>_s42/checkpoints/selected_dev_best.zip \
  experiment.name=eval_<method>_s42 \
  experiment.output_dir=runs/research_v1/eval_<method>_s42 \
  eval.n_episodes=100 metadrive.config.start_seed=5000 \
  metadrive.config.num_scenarios=200 metadrive.config.horizon=500
python scripts/analyze_failures.py --run runs/research_v1/eval_<method>_s42
python scripts/aggregate_results.py --root runs/research_v1
```

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

CSV path: `runs/research_v1/eval_<method>_s42/eval/heldout_random.csv`.

Success: FASB improves safety-efficiency tradeoff over normal fine-tuning by reducing collision/offroad/cost while preserving route completion and success. Failure: safety improves only by collapsing progress, or progress improves while safety worsens.

## Axis 2 - Sampler

Goal: study failure replay ratio.

Variants:

```text
Uniform
Mixed 0.3
Mixed 0.6
Mixed 0.9
Mixed 1.0
```

Allowed variables: `sampler._target_`, `sampler.failure_ratio`, `sampler.alpha`. Screening: 100k. Final: best 1-2 ratios at 300k. Not allowed: changing base checkpoint, final buffer, train/eval seeds, horizon, traffic density, or PPO hyperparameters for one variant only.

Claim: failure-aware replay improves specialization; too much replay may overfit or forget general driving.

Output naming: `runs/research_v1/axis2_sampler_<variant>_s42` and `runs/research_v1/eval_axis2_sampler_<variant>_s42`.

## Axis 3 - Budget/Penalty

Goal: study adaptive safety pressure.

Variants:

```text
fixed 0.03
fixed 0.05
fixed 0.10
adaptive default
adaptive strict
adaptive loose
```

Allowed variables: `safety_budget.*` and `penalty_scheduler.*`. Not allowed: sampler ratio, final buffer, base checkpoint, train/eval seeds, horizon, traffic density, or total timesteps.

Claim: adaptive budget better balances safety and progress than fixed global penalty.

Success: lower collision/offroad/cost with preserved route completion. Failure: strictness causes timeout/low-progress collapse.

## Axis 4 - Cost Function

Goal: study safety cost definition.

Variants:

```text
CrashOnlyCost
DefaultDrivingCost
NearMissHeavyCost
```

Allowed variables: `cost_function.*`. Not allowed: scorer, classifier, sampler, safety budget, train/eval seeds, base checkpoint, buffer, timesteps.

Claim: richer pre-crash safety cost may improve safety but can increase conservatism.

Interpretation: use `failure_by_mode.csv` to distinguish fewer crashes from more timeouts/low-progress behavior.

## Axis 5 - Failure Scorer/Generalization

Goal: study failure definition and distribution shift.

Variants:

```text
DefaultFailureScorer
NearFailureScorer
canonical buffer vs optional dense buffer
heldout random vs dense/easy traffic eval
```

Allowed variables: `failure_scorer.*`, `failure_classifier.*`, eval `traffic_density`, and axis-specific buffers only when explicitly studying buffer or distribution shift. Not allowed: changing total timesteps, base checkpoint, or train/eval seed ranges in the same comparison.

Claim: failure discovery quality and target-distribution shift affect specialization.

Success: scorer/generalization changes improve safety-efficiency under the intended eval distribution without hiding regressions in heldout random.

## Required Analysis For Every Axis

Produce:

```text
runs/research_v1/eval_<method>_s42/eval/heldout_random.csv
runs/research_v1/eval_<method>_s42/analysis/failure_summary.csv
runs/research_v1/eval_<method>_s42/analysis/failure_by_mode.csv
runs/research_v1/eval_<method>_s42/analysis/paper_numbers.md
```

Then run:

```bash
python scripts/aggregate_results.py --root runs/research_v1
```
