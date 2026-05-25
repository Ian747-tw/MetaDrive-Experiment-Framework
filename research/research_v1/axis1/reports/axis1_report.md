# Axis 1 Research Report: Stable FASB-PPO vs Naive PPO

## Status

Axis 1 is complete for the current stable protocol. The main conclusion is mixed: stable FASB-PPO improves average safety-oriented metrics over naive PPO across six training seeds, but it does not reliably dominate naive PPO on paired seed comparisons.

FASB-PPO in this project means:

```text
SB3 PPO + failure-aware sampler + adaptive safety penalty
```

It is not PPO-Lagrangian and does not claim constrained-RL guarantees.

## Research Question

Axis 1 asks whether failure-aware safety fine-tuning improves over normal PPO fine-tuning when both methods share the same foundation checkpoint, optimizer regime, training budget, checkpoint-selection policy, evaluation range, and metrics.

The evaluated hypothesis is:

```text
Failure-aware scenario replay plus adaptive safety penalties improve safety-efficiency over normal PPO fine-tuning without causing timeout or low-progress collapse.
```

## Locked Protocol

All active final comparisons use:

| item | value |
| --- | --- |
| Base checkpoint | GitHub release artifact; local after download: `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Canonical failure buffer | GitHub release artifact; local after download: `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Fine-tuning steps | `300000` |
| Learning rate | `0.00003` |
| Vec env | `dummy`, `n_envs=1` |
| Device | `cpu` |
| Training horizon | `500` |
| Training traffic density | `0.1` |
| Dev checkpoint selection | `start_seed=4500`, `num_scenarios=100`, `eval.n_episodes=100` |
| Final heldout eval | `start_seed=5000`, `num_scenarios=200`, `eval.n_episodes=100` |
| Selection metric | `safety_efficiency_score` |
| Hard rejects | `success_rate < 0.20`, `route_completion_mean < 0.40`, `timeout_rate > 0.80` |

The multiseed training ranges used:

```text
training seeds: 2000, 3000, 4000, 6000, 7000, 8000
```

Seed `5000` was intentionally not used for training because final heldout evaluation starts at `5000`.

## Methods Compared

| method | description | active config |
| --- | --- | --- |
| Naive PPO stable | Normal PPO fine-tuning from the base checkpoint. | `configs/research_v1/axis1_naive_stable_final.yaml` |
| Stable FASB-PPO | PPO fine-tuning with failure-aware sampler and adaptive safety penalty. | `configs/research_v1/axis1_fasb_stable_final.yaml` |

Stable FASB settings:

```text
sampler.failure_ratio=0.05
safety_budget.d_min=0.10
safety_budget.d_max=0.30
safety_budget.timeout_budget=0.30
penalty_scheduler.lambda_min=0.0
penalty_scheduler.lambda_max=0.25
algorithm.params.learning_rate=0.00003
```

## Output Artifacts

Main multiseed result files:

```text
runs/research_v1/multiseed_axis1_summary/multiseed_axis1_per_seed.csv
runs/research_v1/multiseed_axis1_summary/multiseed_axis1_summary.csv
runs/research_v1/multiseed_axis1_summary/multiseed_axis1_paired_deltas.csv
```

Each run has:

```text
runs/research_v1/multiseed_axis1_<method>_stable_seed<seed>/checkpoints/selected_dev_best.zip
runs/research_v1/multiseed_axis1_select/multiseed_axis1_<method>_stable_seed<seed>/checkpoint_selection.csv
runs/research_v1/eval_multiseed_axis1_<method>_stable_seed<seed>_selected_finalheldout/eval/heldout_random.csv
runs/research_v1/eval_multiseed_axis1_<method>_stable_seed<seed>_selected_finalheldout/analysis/failure_by_mode.csv
```

## Six-Train Interim Result

The six-train result uses the first three paired seeds: `2000`, `3000`, and `4000`.

| method | success_rate | collision_rate | offroad_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive PPO stable | 0.430 +/- 0.044 | 0.267 +/- 0.015 | 0.507 +/- 0.025 | 0.570 +/- 0.044 | 0.7055 +/- 0.0392 | 261.16 +/- 20.87 | -0.628 +/- 0.097 |
| Stable FASB-PPO | 0.430 +/- 0.030 | 0.237 +/- 0.051 | 0.457 +/- 0.021 | 0.570 +/- 0.030 | 0.6784 +/- 0.0068 | 248.81 +/- 4.31 | -0.548 +/- 0.069 |

Paired FASB-minus-naive deltas:

| seed | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | -0.06 | +0.00 | -0.06 | +0.06 | -0.0511 | -5.72 | -0.030 |
| 3000 | -0.02 | +0.01 | -0.04 | +0.02 | -0.0561 | +7.07 | +0.000 |
| 4000 | +0.08 | -0.10 | -0.05 | -0.08 | +0.0259 | -38.39 | +0.270 |

Six-train interpretation:

Stable FASB-PPO improves average safety-efficiency, collision, offroad, and cost. Success and timeout are tied on average. Route completion is lower for FASB. Paired safety-efficiency results are `1 win / 1 tie / 1 loss`, so the six-train result suggests a safety benefit but is not conclusive.

## Twelve-Train Final Multiseed Result

The twelve-train result uses six paired seeds: `2000`, `3000`, `4000`, `6000`, `7000`, and `8000`.

| method | success_rate | collision_rate | offroad_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive PPO stable | 0.440 +/- 0.041 | 0.257 +/- 0.056 | 0.488 +/- 0.060 | 0.560 +/- 0.041 | 0.6981 +/- 0.0327 | 267.99 +/- 17.71 | -0.585 +/- 0.123 |
| Stable FASB-PPO | 0.435 +/- 0.034 | 0.230 +/- 0.042 | 0.467 +/- 0.045 | 0.565 +/- 0.034 | 0.6879 +/- 0.0189 | 252.10 +/- 6.00 | -0.544 +/- 0.060 |

Per-seed selected checkpoint and final heldout metrics:

| seed | method | selected checkpoint | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | naive | `final.zip` | 0.46 | 0.25 | 0.51 | 0.54 | 0.7264 | 251.69 | -0.570 |
| 2000 | FASB | `latest_300000_steps.zip` | 0.40 | 0.25 | 0.45 | 0.60 | 0.6753 | 245.97 | -0.600 |
| 3000 | naive | `latest_100000_steps.zip` | 0.45 | 0.27 | 0.48 | 0.55 | 0.7298 | 246.70 | -0.575 |
| 3000 | FASB | `latest_100000_steps.zip` | 0.43 | 0.28 | 0.44 | 0.57 | 0.6737 | 253.77 | -0.575 |
| 4000 | naive | `latest_200000_steps.zip` | 0.38 | 0.28 | 0.53 | 0.62 | 0.6603 | 285.09 | -0.740 |
| 4000 | FASB | `final.zip` | 0.46 | 0.18 | 0.48 | 0.54 | 0.6862 | 246.70 | -0.470 |
| 6000 | naive | `final.zip` | 0.40 | 0.23 | 0.57 | 0.60 | 0.6601 | 291.27 | -0.700 |
| 6000 | FASB | `latest_100000_steps.zip` | 0.47 | 0.26 | 0.41 | 0.53 | 0.7041 | 255.76 | -0.465 |
| 7000 | naive | `latest_300000_steps.zip` | 0.49 | 0.34 | 0.42 | 0.51 | 0.6895 | 268.94 | -0.525 |
| 7000 | FASB | `latest_200000_steps.zip` | 0.46 | 0.23 | 0.54 | 0.54 | 0.6704 | 261.45 | -0.580 |
| 8000 | naive | `latest_300000_steps.zip` | 0.46 | 0.17 | 0.42 | 0.54 | 0.7222 | 264.22 | -0.400 |
| 8000 | FASB | `latest_300000_steps.zip` | 0.39 | 0.18 | 0.48 | 0.61 | 0.7173 | 248.95 | -0.575 |

Paired FASB-minus-naive deltas:

| seed | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | -0.06 | +0.00 | -0.06 | +0.06 | -0.0511 | -5.72 | -0.030 |
| 3000 | -0.02 | +0.01 | -0.04 | +0.02 | -0.0561 | +7.07 | +0.000 |
| 4000 | +0.08 | -0.10 | -0.05 | -0.08 | +0.0259 | -38.39 | +0.270 |
| 6000 | +0.07 | +0.03 | -0.16 | -0.07 | +0.0440 | -35.51 | +0.235 |
| 7000 | -0.03 | -0.11 | +0.12 | +0.03 | -0.0191 | -7.49 | -0.055 |
| 8000 | -0.07 | +0.01 | +0.06 | +0.07 | -0.0049 | -15.27 | -0.175 |

Paired win counts for FASB-minus-naive:

| metric | mean delta | FASB wins | ties | naive wins | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| safety_efficiency_score | +0.0408 | 2 | 1 | 3 | FASB better on mean, not robust by paired wins. |
| success_rate | -0.0050 | 2 | 0 | 4 | Naive slightly better. |
| collision_rate | -0.0267 | 3 | 1 | 2 | FASB lower on mean. |
| offroad_rate | -0.0217 | 4 | 0 | 2 | FASB lower on mean. |
| timeout_rate | +0.0050 | 2 | 0 | 4 | Naive slightly better. |
| route_completion_mean | -0.0102 | 2 | 0 | 4 | Naive slightly better. |
| episode_cost_mean | -15.8850 | 5 | 0 | 1 | FASB lower on mean and paired wins. |

## Historical Diagnostics

The original Axis 1 single-run result looked stronger for stable FASB:

| method | success_rate | collision_rate | offroad_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive PPO stable single run | 0.4400 | 0.2200 | 0.3900 | 0.5600 | 0.6828 | 245.0600 | -0.4500 |
| Fixed-budget stable single run | 0.4800 | 0.2300 | 0.4100 | 0.5200 | 0.7288 | 241.4000 | -0.4200 |
| Stable FASB-PPO single run | 0.5300 | 0.3000 | 0.3700 | 0.4700 | 0.7330 | 230.2700 | -0.3750 |

However, fresh retraining showed high seed sensitivity. The multiseed result is therefore more authoritative than the earlier single-run result.

Collapsed historical variants remain diagnostic only:

| method | success_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original FASB collapsed | 0.0000 | 1.0000 | 0.0143 | 0.0000 | -0.5000 |
| Failed FASB v2 collapsed | 0.0000 | 1.0000 | 0.0121 | 596.3000 | -1.2700 |
| Fixed-budget old protocol collapsed | 0.0000 | 1.0000 | 0.0029 | 315.9700 | -1.4500 |

## Main Interpretation

Does stable FASB beat naive PPO?

No clean domination. Across six paired seeds, stable FASB has better mean safety-efficiency, collision, offroad, and cost. Naive PPO has slightly better mean success, timeout, and route completion. Paired safety-efficiency results are `2 FASB wins / 1 tie / 3 naive wins`.

Does stable FASB improve safety?

Yes on average. Stable FASB reduces mean collision from `0.257` to `0.230`, offroad from `0.488` to `0.467`, and cost from `267.99` to `252.10`. Cost is the most consistent improvement, with FASB lower on five of six paired seeds.

Does stable FASB preserve progress?

Mostly, but not perfectly. Stable FASB avoids the original timeout collapse, but mean success is slightly lower (`0.435` vs `0.440`), mean timeout is slightly higher (`0.565` vs `0.560`), and route completion is slightly lower (`0.6879` vs `0.6981`).

Did any stable method collapse into timeout?

No. Neither stable naive nor stable FASB collapses like the original FASB/v2/fixed-budget failures. Timeout remains high for both methods, but not total collapse.

## Bugs and Issues Encountered

- Original FASB collapsed into timeout under the first default.
- FASB v2 appeared acceptable at 100k dev but collapsed after 300k, showing that short calibration was misleading.
- A fairness issue was found in the first stable comparison: stable FASB used `learning_rate=0.00003`, while old naive/fixed baselines used `linear:3.0e-4`. The active protocol fixes this by using the stable optimizer and dev checkpoint-selection policy for all compared fine-tuned methods.
- The result is sensitive to retraining seed. Single-run claims are not reliable enough for the final paper.

## Recommendation for Axis 2-5

Proceed with Axes 2-5, but frame stable FASB as a safety-biased method with mixed performance rather than a method that strictly beats naive PPO. The strongest Axis 1 claim is:

```text
Stable FASB-PPO avoids timeout collapse and improves average safety/cost metrics over naive PPO, but does not robustly dominate naive PPO across training seeds.
```

Axes 2-5 should explain this tradeoff:

- Axis 2 should test whether failure replay ratio controls the safety/progress tradeoff.
- Axis 3 should test whether budget/penalty scheduling causes either useful safety pressure or conservative timeout behavior.
- Axis 4 should test whether the cost definition is driving the cost improvement or creating unintended conservatism.
- Axis 5 should test whether scorer quality and evaluation distribution determine when failure-aware fine-tuning generalizes.
