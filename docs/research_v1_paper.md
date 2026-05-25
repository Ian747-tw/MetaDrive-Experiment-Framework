# Failure-Aware Safety-Biased PPO Fine-Tuning for MetaDrive

## 1. Abstract

Reinforcement learning policies for autonomous driving must balance task progress with safety under rare but consequential failure modes. Standard PPO fine-tuning can improve driving performance, but it does not explicitly focus training on previously observed collisions, offroad events, or low-progress failures. This project studies a modular failure-aware fine-tuning framework for MetaDrive. The proposed method, FASB-PPO, keeps the core optimizer as Stable-Baselines3 PPO and adds three practical components: a failure-aware sampler, a configurable safety cost, and an adaptive safety penalty. The framework is designed as an experiment platform rather than a monolithic algorithm: it provides reproducible artifact validation, canonical base checkpoints, canonical failure buffers, locked evaluation ranges, dev-only checkpoint selection, failure-mode analysis, and five research axes for main comparison and ablation.

Axis 1 compares stable FASB-PPO with normal PPO fine-tuning under a shared base checkpoint, optimizer regime, fine-tuning budget, checkpoint-selection protocol, and final heldout evaluation. Across six paired training seeds, stable FASB-PPO reduces average collision rate from 0.257 to 0.230, offroad rate from 0.488 to 0.467, episode cost from 267.99 to 252.10, and improves mean safety-efficiency score from -0.585 to -0.544. However, naive PPO slightly outperforms FASB-PPO on mean success rate, timeout rate, and route completion, and FASB-PPO wins safety-efficiency on only two of six paired seeds, with one tie and three losses. The current conclusion is therefore measured: stable FASB-PPO avoids the earlier timeout collapse and improves average safety/cost metrics, but it does not robustly dominate naive PPO across training seeds. Axes 2-5 are designed to explain this tradeoff through sampler, penalty, cost, and generalization ablations.

## 2. Introduction

### 2.1 Motivation, Observation, and Contribution

Autonomous-driving reinforcement learning is difficult because the reward objective and the safety objective are not naturally aligned. A policy can complete routes while producing unsafe events, or it can avoid costs by becoming overly conservative and timing out. This tension appeared clearly in our first Axis 1 experiments: the original FASB default drove collision/offroad/cost to zero but also collapsed into `success_rate=0.00`, `timeout_rate=1.00`, and nearly zero route completion. A later v2 calibration looked acceptable at 100k timesteps but collapsed at 300k, showing that short calibration was not sufficient.

The central observation is that safety fine-tuning must be evaluated as a tradeoff, not as a single metric. Avoiding collisions is not enough if the agent stops making progress. Improving success is not enough if it increases unsafe behavior. The project therefore emphasizes a controlled research protocol: same base checkpoint, same training budget, same optimizer regime, same dev checkpoint-selection policy, same final heldout evaluation, and same failure-mode analysis.

The main contribution is a modular FASB-PPO research framework for MetaDrive. The method is intentionally pragmatic:

```text
FASB-PPO = SB3 PPO + failure-aware sampler + adaptive safety penalty
```

It does not rewrite PPO and does not claim PPO-Lagrangian or constrained-RL guarantees. Its novelty is in making failure-aware safety specialization experimentally fair, reproducible, and ablatable:

- A canonical foundation stage provides a shared base checkpoint and failure buffer.
- Failure-aware sampling replays previously discovered failure scenarios while preserving random scenario coverage.
- Adaptive safety penalties provide tunable pressure against unsafe behavior without requiring a full constrained policy optimizer.
- Dev-only checkpoint selection prevents selecting checkpoints on final heldout results.
- Failure-mode analysis separates collisions, offroad events, solved episodes, timeouts, and unknown failures.
- A five-axis research design separates the main effect from sampler, budget, cost, and generalization mechanisms.

### 2.2 Problem Statement

The problem addressed is:

```text
Can a failure-aware safety-biased PPO fine-tuning framework improve safety-efficiency over normal PPO fine-tuning in MetaDrive while preserving route progress and avoiding timeout collapse?
```

This problem is important because many safe-RL methods optimize aggregate cost or constraints, but in driving-style tasks the practical failure modes are structured. A collision, offroad event, timeout, and near-miss should not be treated only as undifferentiated scalar reward noise. At the same time, the method must remain comparable to common PPO baselines and should not require a complete rewrite of the learning algorithm.

## 3. Related Work

PPO is a standard policy-gradient method that improves training stability through clipped policy updates and has become a common baseline for continuous-control RL [1]. This project keeps PPO as the optimizer to make the comparison interpretable: changes in performance are attributed to failure-aware sampling and safety penalties, not to replacing the underlying RL algorithm.

Safe and constrained RL methods such as Constrained Policy Optimization study policy optimization under explicit cost constraints [2]. Safety Gym and related safe-exploration benchmarks emphasize the need to evaluate both reward and safety costs [3]. FASB-PPO is related in motivation but differs in implementation: it does not enforce formal constraints or solve a constrained optimization problem. Instead, it adds modular safety pressure and failure replay around SB3 PPO.

MetaDrive provides procedurally generated driving scenarios for generalizable RL research [4]. This project uses MetaDrive as the experimental environment and contributes a research workflow around reproducible fine-tuning, artifact validation, failure-buffer reuse, and axis-based ablations.

Failure replay and prioritized sampling ideas are broadly related to replaying informative experiences or emphasizing difficult examples. The difference here is that replay is organized at the scenario level: the framework stores failure scenarios and samples them during fine-tuning, allowing researchers to study whether revisiting failed driving situations improves safety without overfitting or freezing.

## 4. Methodology

### 4.1 Framework Overview

The framework has three stages:

1. Foundation: train or download a base PPO checkpoint and build a canonical large failure buffer.
2. Fine-tuning: compare naive PPO, fixed-budget safety fine-tuning, and FASB-PPO variants from the same base checkpoint.
3. Evaluation and analysis: select checkpoints on dev seeds, evaluate once on final heldout seeds, aggregate metrics, and analyze failure modes.

The stable active protocol uses:

| component | setting |
| --- | --- |
| Base checkpoint | `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Failure buffer | `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Fine-tuning timesteps | `300000` |
| Learning rate | `0.00003` |
| Training scenario seeds | method-specific train seed, `num_scenarios=500` |
| Dev checkpoint selection | `start_seed=4500`, `num_scenarios=100`, `eval.n_episodes=100` |
| Final heldout evaluation | `start_seed=5000`, `num_scenarios=200`, `eval.n_episodes=100` |
| Horizon | `500` |
| Traffic density | `0.1` |

### 4.2 FASB-PPO Components

Failure-aware sampler:

The sampler mixes random MetaDrive scenarios with scenarios from the canonical failure buffer. The stable default uses:

```text
sampler.failure_ratio=0.05
sampler.alpha=0.7
sampler.max_too_hard_ratio=0.15
```

The goal is to expose the policy to meaningful failures without letting the replay distribution dominate all training.

Adaptive safety budget and penalty:

The safety budget and penalty scheduler convert observed costs into a shaped training signal:

```text
safety_budget.d_min=0.10
safety_budget.d_max=0.30
safety_budget.timeout_budget=0.30
penalty_scheduler.lambda_min=0.0
penalty_scheduler.lambda_max=0.25
```

This is intentionally a bounded penalty approach. It is not a Lagrangian method and does not guarantee constraint satisfaction.

Cost and failure plugins:

The framework exposes plugin interfaces for:

- `cost_function`
- `failure_scorer`
- `failure_classifier`
- `sampler`
- `safety_budget`
- `penalty_scheduler`

This modular structure enables Axes 2-5 to change one research variable at a time.

### 4.3 Checkpoint Selection

Each fine-tuned run saves periodic checkpoints and `final.zip`. Checkpoints are evaluated only on the dev range:

```text
start_seed=4500
num_scenarios=100
eval.n_episodes=100
```

The selection metric is `safety_efficiency_score`, with hard rejection if:

```text
success_rate < 0.20
route_completion_mean < 0.40
timeout_rate > 0.80
```

The selected checkpoint is copied to:

```text
<run-dir>/checkpoints/selected_dev_best.zip
```

The final heldout range is not used for checkpoint selection.

### 4.4 Challenges and Fixes

Timeout collapse:

The original FASB default collapsed into timeouts. This exposed a failure mode where safety pressure can create a stationary or overly conservative policy. The stable default reduces failure replay and penalty strength.

Short calibration was misleading:

An attempted v2 looked acceptable at 100k dev evaluation but collapsed at 300k. The protocol now uses 300k dev calibration and checkpoint selection.

Optimizer confound:

The first stable comparison used a lower learning rate for FASB than for old naive/fixed baselines. The active Axis 1 comparison fixes this by using the same stable learning rate and checkpoint-selection protocol for naive PPO and FASB-PPO.

Seed sensitivity:

Fresh retraining changed the single-run conclusion. The final Axis 1 analysis therefore reports a six-seed paired comparison rather than relying on one run.

## 5. Experiments

### 5.1 Five-Axis Research Design

Axis 1: Main comparison.

Compares naive PPO stable fine-tuning and stable FASB-PPO under identical optimizer, training budget, checkpoint-selection, and final-evaluation protocol. Fixed-budget safety fine-tuning is retained as a diagnostic baseline from earlier runs and can be rerun under the multiseed protocol if compute allows.

Axis 2: Sampler ablation.

Will vary only the sampler target, failure replay ratio, and priority/alpha. This axis tests whether failure replay improves specialization or causes overfitting/conservatism.

Planned fill-in table:

| variant | failure_ratio | alpha | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mixed low | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mixed medium | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mixed high | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Axis 3: Budget and penalty ablation.

Will vary only `safety_budget.*` and `penalty_scheduler.*`. This axis tests whether adaptive safety pressure improves the safety/progress balance compared with fixed or stricter penalties.

Planned fill-in table:

| variant | d_min | d_max | timeout_budget | lambda_min | lambda_max | success | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed low | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed medium | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Adaptive default | 0.10 | 0.30 | 0.30 | 0.0 | 0.25 | TBD | TBD | TBD | TBD |
| Adaptive strict | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Axis 4: Cost-function ablation.

Will vary only `cost_function.*`. This axis tests whether crash-only, default driving cost, or near-miss-heavy cost better predicts useful safety improvement.

Planned fill-in table:

| cost function | success | collision | offroad | timeout | route | episode_cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CrashOnlyCost | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DefaultDrivingCost | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NearMissHeavyCost | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Axis 5: Failure scorer and generalization.

Will vary failure scorer/classifier or explicitly labeled eval distribution/buffer settings. This axis tests whether the quality of failure discovery and target distribution affects transfer.

Planned fill-in table:

| variant | scorer/classifier | eval distribution | buffer | success | route | cost | safety_eff |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Default scorer | TBD | heldout random | canonical | TBD | TBD | TBD | TBD |
| Near-failure scorer | TBD | heldout random | canonical | TBD | TBD | TBD | TBD |
| Distribution shift | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.2 Axis 1 Experimental Setup

Methods:

| method | description |
| --- | --- |
| Naive PPO stable | PPO fine-tuning from the base checkpoint using the stable optimizer protocol. |
| Stable FASB-PPO | PPO fine-tuning plus failure-aware sampling and adaptive safety penalty. |

Training seeds:

```text
2000, 3000, 4000, 6000, 7000, 8000
```

Seed `5000` is excluded because it overlaps the final heldout scenario range.

Metrics:

- `success_rate`
- `collision_rate`
- `offroad_rate`
- `timeout_rate`
- `route_completion_mean`
- `episode_cost_mean`
- `cost_violation_rate`
- `safety_efficiency_score`

### 5.3 Axis 1 Results

Six-train interim result:

| method | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive PPO | 0.430 +/- 0.044 | 0.267 +/- 0.015 | 0.507 +/- 0.025 | 0.570 +/- 0.044 | 0.7055 +/- 0.0392 | 261.16 +/- 20.87 | -0.628 +/- 0.097 |
| FASB-PPO | 0.430 +/- 0.030 | 0.237 +/- 0.051 | 0.457 +/- 0.021 | 0.570 +/- 0.030 | 0.6784 +/- 0.0068 | 248.81 +/- 4.31 | -0.548 +/- 0.069 |

Twelve-train final multiseed result:

| method | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive PPO | 0.440 +/- 0.041 | 0.257 +/- 0.056 | 0.488 +/- 0.060 | 0.560 +/- 0.041 | 0.6981 +/- 0.0327 | 267.99 +/- 17.71 | -0.585 +/- 0.123 |
| FASB-PPO | 0.435 +/- 0.034 | 0.230 +/- 0.042 | 0.467 +/- 0.045 | 0.565 +/- 0.034 | 0.6879 +/- 0.0189 | 252.10 +/- 6.00 | -0.544 +/- 0.060 |

Paired FASB-minus-naive summary over six seeds:

| metric | mean delta | FASB wins | ties | naive wins |
| --- | ---: | ---: | ---: | ---: |
| safety_efficiency_score | +0.0408 | 2 | 1 | 3 |
| success_rate | -0.0050 | 2 | 0 | 4 |
| collision_rate | -0.0267 | 3 | 1 | 2 |
| offroad_rate | -0.0217 | 4 | 0 | 2 |
| timeout_rate | +0.0050 | 2 | 0 | 4 |
| route_completion_mean | -0.0102 | 2 | 0 | 4 |
| episode_cost_mean | -15.8850 | 5 | 0 | 1 |

### 5.4 Axis 1 Analysis

Stable FASB-PPO is safer on average:

- Collision rate decreases from 0.257 to 0.230.
- Offroad rate decreases from 0.488 to 0.467.
- Episode cost decreases from 267.99 to 252.10.
- Safety-efficiency improves from -0.585 to -0.544.

Stable FASB-PPO does not clearly dominate task progress:

- Success rate slightly decreases from 0.440 to 0.435.
- Timeout rate slightly increases from 0.560 to 0.565.
- Route completion decreases from 0.6981 to 0.6879.

The result is therefore best described as a safety-progress tradeoff. FASB-PPO avoids the catastrophic timeout collapse observed in the original configuration, but its average safety gains are not strong enough to claim robust superiority over naive PPO across all seeds.

### 5.5 Discussion

The most important result is not that FASB-PPO strictly wins; it is that the framework exposes when a safety method appears good for the wrong reason. Original FASB reduced raw costs by freezing. Stable FASB avoids that collapse, but the multiseed result shows seed sensitivity and a nuanced tradeoff.

This is a useful contribution because many RL projects stop at one seed or one final checkpoint. This framework makes the comparison more scientifically defensible by:

- separating dev checkpoint selection from final heldout evaluation,
- using paired seed comparisons,
- preserving failed configurations as diagnostics,
- reporting safety, progress, and failure modes together,
- making every major design choice ablatable in Axes 2-5.

## 6. Conclusion

This project develops a reproducible MetaDrive research framework for failure-aware safety-biased PPO fine-tuning. The framework introduces modular failure replay, adaptive safety penalties, plugin-based costs and failure classifiers, validated artifacts, dev-only checkpoint selection, and axis-based ablations.

Axis 1 shows that stable FASB-PPO avoids the original timeout collapse and improves average safety/cost metrics compared with naive PPO. However, it does not robustly dominate naive PPO across six paired training seeds. The current evidence supports a cautious claim:

```text
Stable FASB-PPO improves average safety-efficiency and cost metrics, but the improvement is seed-sensitive and trades off slightly against success, timeout, and route completion.
```

Axes 2-5 should now determine which component causes the safety improvement and whether the progress tradeoff can be reduced.

## 7. Member Workload

Camera-ready version only. Fill this section with team member names and contributions.

| member | contribution |
| --- | --- |
| TBD | Framework implementation / training / analysis / writing |
| TBD | Axis 2 sampler experiments |
| TBD | Axis 3 budget experiments |
| TBD | Axis 4 cost-function experiments |
| TBD | Axis 5 generalization experiments |

## 8. References

[1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

[2] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. "Constrained Policy Optimization." arXiv:1705.10528, 2017. https://arxiv.org/abs/1705.10528

[3] Alex Ray, Joshua Achiam, and Dario Amodei. "Benchmarking Safe Exploration in Deep Reinforcement Learning." OpenAI Safety Gym, 2019. https://openai.com/index/benchmarking-safe-exploration-in-deep-reinforcement-learning

[4] Quanyi Li, Zhenghao Peng, Zhenghai Xue, Qihang Zhang, and Bolei Zhou. "MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning." arXiv:2109.12674, 2021. https://arxiv.org/abs/2109.12674

## 9. Appendices

### Appendix A: Main Artifact Paths

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
runs/research_v1/multiseed_axis1_summary/multiseed_axis1_per_seed.csv
runs/research_v1/multiseed_axis1_summary/multiseed_axis1_summary.csv
runs/research_v1/multiseed_axis1_summary/multiseed_axis1_paired_deltas.csv
```

### Appendix B: Active Axis 1 Configs

```text
configs/research_v1/axis1_naive_stable_final.yaml
configs/research_v1/axis1_fasb_stable_final.yaml
```

### Appendix C: Reporting Guidance for Axes 2-5

Each axis should report:

- locked settings,
- only the varied axis variable,
- selected checkpoint path,
- dev selection metrics,
- final heldout metrics,
- failure-mode table,
- interpretation of safety/progress tradeoff,
- whether the result changes the Axis 1 conclusion.
