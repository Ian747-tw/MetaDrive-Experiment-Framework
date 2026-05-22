# Research V1 Axis Specs

## Shared Final Rules

All final comparisons use the same base checkpoint, canonical large failure buffer, 300k fine-tuning steps, train seed range, eval seed range, horizon, traffic density, metrics, and analysis process. Axis researchers may only change the variables listed for their axis.

Locked final paths:

```text
base checkpoint: runs/research_v1/base_pretrain_s42/checkpoints/final.zip
large buffer:    runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

## Axis 1 - Main Comparison

Goal: show FASB-PPO beats normal PPO fine-tuning and fixed-budget fine-tuning under the same base checkpoint, buffer, timesteps, train/eval seeds, and horizon.

Methods:

```text
Base checkpoint eval
Naive FT 300k
Fixed-budget FT 300k
Original FASB-PPO 300k
FASB-PPO v2 300k
```

Allowed variables: `mode`, method config, and method-specific safety-budget behavior. Not allowed: base checkpoint, large buffer path, total timesteps, train/eval seeds, horizon, traffic density, PPO hyperparameters unless changed for every compared method.

Commands:

```bash
python scripts/train.py --config configs/research_v1/axis1_naive_final.yaml
python scripts/train.py --config configs/research_v1/axis1_fixed_budget_final.yaml
python scripts/train.py --config configs/research_v1/axis1_fasb_final.yaml
python scripts/train.py --config configs/research_v1/axis1_fasb_v2_final.yaml
```

`axis1_fasb_final.yaml` is retained as the original historical default. Axis 1 found that this original default collapsed into timeout: it reduced collision/offroad/cost by nearly stopping, not by preserving useful driving progress. `axis1_fasb_v2_final.yaml` is the attempted v2 candidate selected on the 100k-step dev validation range (`start_seed=4500`, `num_scenarios=100`) before final heldout rerun. FASB v2 uses gentler failure replay and safety pressure: `sampler.failure_ratio=0.3`, `safety_budget.d_min=0.03`, `safety_budget.d_max=0.15`, `safety_budget.timeout_budget=0.20`, `penalty_scheduler.lambda_min=0.05`, and `penalty_scheduler.lambda_max=2.0`.

Follow-up result: the 100k dev-selected FASB v2 candidate did not survive the final 300k schedule. Its 300k checkpoint collapsed on both dev and final heldout. Keep `axis1_fasb_v2_final.yaml` as a reproducibility record of the failed candidate, not as an accepted default.

Axis 2 and Axis 3 should use this failure to guide ablations around training duration, replay ratio, and penalty strength. The original FASB config remains useful as a strict/conservative reference, but neither the original config nor the attempted v2 candidate should be presented as a proven balanced default.

Evaluate each method:

```bash
python scripts/evaluate.py --config configs/eval/heldout_random.yaml \
  --checkpoint runs/research_v1/<method>_s42/checkpoints/final.zip \
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

Allowed variables: `sampler._target_`, `sampler.failure_ratio`, `sampler.alpha`. Screening: 100k. Final: best 1-2 ratios at 300k. Not allowed: changing base checkpoint, final buffer, train/eval seeds, horizon, traffic density, or PPO hyperparameters for one variant only. Because the attempted v2 candidate passed 100k dev but collapsed at 300k, sampler ablations should include an intermediate dev check at the intended final training length before final heldout evaluation.

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

Allowed variables: `safety_budget.*` and `penalty_scheduler.*`. Not allowed: sampler ratio, final buffer, base checkpoint, train/eval seeds, horizon, traffic density, or total timesteps. Include timeout and route-completion hard screens at 300k dev before promoting any budget/penalty setting.

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
