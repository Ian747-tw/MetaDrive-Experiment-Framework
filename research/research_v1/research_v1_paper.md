# Failure-Aware Safety-Biased PPO Fine-Tuning for MetaDrive

## 1. Abstract

Reinforcement learning policies for autonomous driving must balance task progress with safety under rare but consequential failure modes. Standard PPO fine-tuning can improve driving performance, but it does not explicitly focus training on previously observed collisions, offroad events, or low-progress failures. This project studies a modular failure-aware fine-tuning framework for MetaDrive. The proposed method, FASB-PPO, keeps the core optimizer as Stable-Baselines3 PPO and adds three practical components: a failure-aware sampler, a configurable safety cost, and an adaptive safety penalty. The framework is designed as an experiment platform rather than a monolithic algorithm: it provides reproducible artifact validation, canonical base checkpoints, canonical failure buffers, locked evaluation ranges, dev-only checkpoint selection, failure-mode analysis, and five research axes for main comparison and ablation.

Axis 1 compares stable FASB-PPO with normal PPO fine-tuning under a shared base checkpoint, optimizer regime, fine-tuning budget, checkpoint-selection protocol, and final heldout evaluation. Across six paired training seeds, stable FASB-PPO reduces average collision rate from 0.257 to 0.230, offroad rate from 0.488 to 0.467, episode cost from 267.99 to 252.10, and improves mean safety-efficiency score from -0.585 to -0.544. However, naive PPO slightly outperforms FASB-PPO on mean success rate, timeout rate, and route completion, and FASB-PPO wins safety-efficiency on only two of six paired seeds, with one tie and three losses. Axes 2-5 explain this tradeoff: the Axis 2 screen suggests aggressive failure replay is harmful, fixed safety budgets outperform the current adaptive budget, the default driving cost is the best cost definition, and near-failure scoring is only marginally better than the default scorer. The current conclusion is measured: the framework is a useful and reproducible tool for studying safety-specialized PPO, but the current FASB default is a safety-biased tradeoff rather than a uniformly dominant algorithm.

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
- A reviewer package under `research/research_v1/` stores paper text, configs, result CSVs, checkpoint-selection outputs, and failure analyses for reproduction.

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

Important coverage caveat: the sampler filters failure-buffer entries to the active training scenario range. The canonical local buffer used for this paper contains 2,178 entries with scenario seeds from 1 to 2999. Runs whose training `start_seed` is 3000 or larger therefore have no eligible in-range failure-buffer entries and fall back to the random scenario component. This does not invalidate the optimizer/protocol comparison, but it limits claims specifically about replay exposure on high-start-seed replicates. Direct evidence about the replay ratio itself should be read primarily from Axis 2 and other runs whose training range overlaps the buffer.

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

Failure-buffer coverage:

The large failure buffer covers scenario seeds 1-2999. Some multiseed runs intentionally vary the training scenario range beyond that coverage. Those runs remain useful as robustness checks for the full stable training code path, but they are not pure tests of failure replay because the replay sampler has no eligible in-range failures for `start_seed >= 3000`.

## 5. Experiments

### 5.1 Experimental Design and Reading Guide

The experiment section follows the advice that a good paper should communicate one clear idea and then support it with progressively stronger evidence: first the main effect, then the mechanism, then the limitations. The central claim is not that FASB-PPO is universally better than PPO. The claim is that the framework makes safety-specialized PPO fine-tuning measurable and ablatable, and that the current stable default improves some safety metrics while exposing safety-progress tradeoffs.

All final axes use the same foundation assets unless the axis explicitly studies that variable:

| shared component | value |
| --- | --- |
| Base checkpoint | `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Canonical failure buffer | `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Optimizer | SB3 PPO, `learning_rate=0.00003`, CPU |
| Fine-tuning budget | `300000` timesteps |
| Dev checkpoint selection | `start_seed=4500`, 100 episodes |
| Final heldout evaluation | `start_seed=5000`, 100 episodes, 200 scenarios |
| Horizon and traffic | horizon `500`, traffic density `0.1` unless distribution shift is explicit |

The primary metric is `safety_efficiency_score`, interpreted together with success, timeout, route completion, collision, offroad, and episode cost. A method is not considered successful if safety improves only by freezing into timeout.

The reported win counts are direction-aware: higher is better for success, route completion, and safety-efficiency, while lower is better for collision, offroad, timeout, and episode cost.

### 5.2 Axis 1: Main Comparison

Axis 1 compares normal PPO fine-tuning against stable FASB-PPO. It asks whether failure-aware replay plus safety penalties improves safety-efficiency under the same optimizer, base checkpoint, checkpoint-selection policy, and final heldout evaluation.

![Axis 1 main comparison](charts/axis1_main_comparison.svg)

Twelve-train final multiseed result over six paired seeds:

| method | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive PPO | 0.440 +/- 0.041 | 0.257 +/- 0.056 | 0.488 +/- 0.060 | 0.560 +/- 0.041 | 0.6981 +/- 0.0327 | 267.99 +/- 17.71 | -0.585 +/- 0.123 |
| Stable FASB-PPO | 0.435 +/- 0.034 | 0.230 +/- 0.042 | 0.467 +/- 0.045 | 0.565 +/- 0.034 | 0.6879 +/- 0.0189 | 252.10 +/- 6.00 | -0.544 +/- 0.060 |

Paired FASB-minus-naive summary:

| metric | mean delta | FASB wins | ties | naive wins |
| --- | ---: | ---: | ---: | ---: |
| safety_efficiency_score | +0.0408 | 2 | 1 | 3 |
| success_rate | -0.0050 | 2 | 0 | 4 |
| collision_rate | -0.0267 | 3 | 1 | 2 |
| offroad_rate | -0.0217 | 4 | 0 | 2 |
| timeout_rate | +0.0050 | 2 | 0 | 4 |
| route_completion_mean | -0.0102 | 2 | 0 | 4 |
| episode_cost_mean | -15.8850 | 5 | 0 | 1 |

Axis 1 interpretation: stable FASB-PPO avoids the original timeout collapse and improves average collision, offroad, cost, and safety-efficiency. It does not robustly dominate naive PPO because success, timeout, and route completion are slightly worse on average, and paired safety-efficiency wins are mixed. The strongest supported claim is safety-bias, not universal superiority.

Because several Axis 1 training seeds are outside the canonical failure-buffer seed range, this result should be interpreted as the stable FASB protocol under multiseed retraining, not as evidence that every replicate received the same amount of failure replay. The cleanest replay-specific comparison is Axis 2, where the seed-2000 scenario range overlaps the buffer.

### 5.3 Axis 2: Failure Sampler Ratio

Axis 2 varies the failure replay ratio while keeping the stable optimizer, base checkpoint, budget, penalty, and cost fixed. It tests whether more failure replay improves specialization or causes overfitting/conservatism.

![Axis 2 sampler ablation](charts/axis2_sampler_ablation.svg)

Single-seed sampler screening result, seed 2000:

| variant | failure ratio | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed005 | 0.05 | 0.48 | 0.28 | 0.49 | 0.52 | 0.7067 | 260.50 | -0.550 |
| mixed030 | 0.30 | 0.47 | 0.28 | 0.44 | 0.53 | 0.6978 | 239.82 | -0.515 |
| mixed060 | 0.60 | 0.35 | 0.18 | 0.59 | 0.65 | 0.6298 | 277.59 | -0.745 |
| mixed090 | 0.90 | 0.42 | 0.25 | 0.48 | 0.58 | 0.7004 | 253.22 | -0.600 |

Axis 2 interpretation: moderate replay (`0.30`) gives the best safety-efficiency in this single-seed screen, mostly by lowering offroad and cost. High replay (`0.60`) is harmful: success and route completion fall, timeout rises, and safety-efficiency is worst. This supports the decision to keep the stable default replay ratio low (`0.05`) and suggests that replay is useful only when it remains a small part of the training distribution. Axis 2 should be treated as screening evidence until repeated over more seeds.

### 5.4 Axis 3: Budget and Penalty

Axis 3 varies safety budget and penalty scheduling. It directly tests the adaptive-safety part of FASB-PPO against fixed-budget alternatives.

![Axis 3 budget/penalty ablation](charts/axis3_budget_penalty_ablation.svg)

Ten-repetition result:

| variant | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed005 | 0.487 +/- 0.068 | 0.260 +/- 0.065 | 0.386 +/- 0.069 | 0.513 +/- 0.068 | 0.7067 +/- 0.0395 | 244.83 +/- 20.56 | -0.416 +/- 0.166 |
| fixed003 | 0.456 +/- 0.070 | 0.239 +/- 0.045 | 0.424 +/- 0.085 | 0.545 +/- 0.071 | 0.7083 +/- 0.0364 | 252.41 +/- 24.58 | -0.480 +/- 0.177 |
| fixed010 | 0.451 +/- 0.068 | 0.275 +/- 0.070 | 0.426 +/- 0.083 | 0.549 +/- 0.068 | 0.6950 +/- 0.0249 | 252.82 +/- 21.27 | -0.525 +/- 0.156 |
| adaptive_strict | 0.442 +/- 0.051 | 0.229 +/- 0.046 | 0.475 +/- 0.062 | 0.560 +/- 0.051 | 0.6818 +/- 0.0385 | 246.60 +/- 16.16 | -0.542 +/- 0.098 |
| adaptive_loose | 0.422 +/- 0.076 | 0.235 +/- 0.050 | 0.479 +/- 0.095 | 0.578 +/- 0.076 | 0.6689 +/- 0.0365 | 271.73 +/- 20.76 | -0.581 +/- 0.183 |
| adaptive_default | 0.373 +/- 0.122 | 0.173 +/- 0.056 | 0.534 +/- 0.154 | 0.627 +/- 0.122 | 0.6669 +/- 0.0342 | 275.03 +/- 48.84 | -0.648 +/- 0.295 |

Axis 3 interpretation: the adaptive-budget hypothesis is not supported. All fixed variants outperform all adaptive variants on mean safety-efficiency. `fixed005` is the best budget/penalty setting: it has the highest success, lowest offroad, lowest cost, and best safety-efficiency. `adaptive_default` has the lowest collision rate, but this is misleading because it also has the highest timeout rate and the worst safety-efficiency, indicating conservative hesitation rather than useful safety.

### 5.5 Axis 4: Cost Function

Axis 4 varies only the safety cost definition. It asks whether richer pre-crash costs improve safety or merely induce conservatism.

![Axis 4 cost-function ablation](charts/axis4_cost_function_ablation.svg)

Three-seed result:

| cost function | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DefaultDrivingCost | 0.507 +/- 0.042 | 0.267 +/- 0.015 | 0.410 +/- 0.062 | 0.493 +/- 0.042 | 0.7138 +/- 0.0314 | 251.22 +/- 13.95 | -0.417 +/- 0.055 |
| EventDrivingCost | 0.490 +/- 0.035 | 0.300 +/- 0.090 | 0.390 +/- 0.062 | 0.510 +/- 0.035 | 0.7018 +/- 0.0299 | 237.74 +/- 6.33 | -0.455 +/- 0.069 |
| NearMissHeavyCost | 0.440 +/- 0.053 | 0.213 +/- 0.040 | 0.430 +/- 0.010 | 0.560 +/- 0.053 | 0.6922 +/- 0.0286 | 266.02 +/- 11.18 | -0.483 +/- 0.085 |
| CrashOnlyCost | 0.373 +/- 0.042 | 0.237 +/- 0.064 | 0.493 +/- 0.055 | 0.627 +/- 0.042 | 0.6677 +/- 0.0314 | 267.50 +/- 9.43 | -0.670 +/- 0.053 |

Axis 4 interpretation: `DefaultDrivingCost` is the best balanced cost. `NearMissHeavyCost` lowers collisions but increases timeout and lowers success, confirming the expected conservatism tradeoff. `CrashOnlyCost` performs worst because sparse crash/offroad feedback is too weak for useful fine-tuning. `EventDrivingCost` has the lowest episode cost and is competitive, but it does not beat default on safety-efficiency.

### 5.6 Axis 5: Failure Scorer and Generalization

Axis 5 varies the failure scorer and adds a small traffic-density shift study. It asks whether a near-failure-oriented scorer discovers more useful failures than the default scorer.

![Axis 5 failure-scorer ablation](charts/axis5_failure_scorer_ablation.svg)

Four-seed canonical heldout result:

| scorer | success | collision | offroad | timeout | route | cost | safety_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NearFailureScorer | 0.453 +/- 0.070 | 0.260 +/- 0.039 | 0.445 +/- 0.058 | 0.550 +/- 0.067 | 0.6872 +/- 0.0311 | 252.78 +/- 13.44 | -0.528 +/- 0.134 |
| DefaultFailureScorer | 0.433 +/- 0.051 | 0.228 +/- 0.044 | 0.488 +/- 0.045 | 0.568 +/- 0.051 | 0.6813 +/- 0.0248 | 256.29 +/- 19.57 | -0.566 +/- 0.100 |

Paired near-failure-minus-default deltas:

| seed | delta safety_eff | delta success | delta collision | delta offroad | delta timeout | delta cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.130 | +0.08 | +0.11 | -0.12 | -0.08 | +3.63 |
| 2000 | -0.150 | -0.08 | +0.04 | -0.01 | +0.08 | -1.25 |
| 3000 | -0.015 | +0.01 | -0.01 | +0.04 | -0.01 | +11.03 |
| 4000 | +0.190 | +0.07 | -0.01 | -0.08 | -0.06 | -27.44 |

Axis 5 interpretation: `NearFailureScorer` is marginally better on mean safety-efficiency, success, offroad, timeout, route, and cost, but it is not robust by paired comparison: safety-efficiency is 2 wins and 2 losses. It also increases mean collision. The scorer is therefore a second-order knob, not a decisive improvement. Distribution-shift tests on seeds 42 and 2000 show no strong generalization advantage: both scorers degrade on easy traffic, while the default scorer is better under dense traffic.

Axis 5 also mixes replicate types: seeds 42 and 2000 use the fixed training scenario range that overlaps the failure buffer, while teammate seeds 3000 and 4000 vary the scenario range and are outside the current buffer coverage. The four-seed table is useful for robustness, but the scorer-specific replay claim is strongest for the overlapping-buffer seeds.

### 5.7 Cross-Axis Synthesis

The five axes clarify what the framework contributes and where the current method is weak.

| axis | main finding | implication |
| --- | --- | --- |
| Axis 1 | FASB improves average safety/cost but not paired dominance. | Main method is safety-biased, not uniformly superior. |
| Axis 2 | Too much failure replay hurts in the seed-2000 screen; moderate replay screens best. | Replay must be controlled and needs multiseed follow-up. |
| Axis 3 | Fixed budget `0.05` beats adaptive budget. | Current adaptive penalty is not the best safety mechanism. |
| Axis 4 | Default cost is best balanced; crash-only is worst. | Cost shaping is load-bearing. |
| Axis 5 | Near-failure scorer is marginal and not robust. | Scorer design matters less than budget/cost choices. |

The most important scientific result is that the framework prevents overclaiming. Original FASB appeared safe because it froze. The stable protocol, multiseed comparisons, and ablations reveal a more useful picture: safety-aware fine-tuning can reduce costs, but sampler ratio, budget schedule, and cost definition strongly determine whether the agent remains productive.

## 6. Conclusion

This project develops a reproducible MetaDrive research framework for failure-aware safety-biased PPO fine-tuning. The framework introduces modular failure replay, safety penalties, plugin-based costs and failure classifiers, validated artifacts, dev-only checkpoint selection, failure-mode analysis, and axis-based ablations.

Axis 1 shows that stable FASB-PPO avoids the original timeout collapse and improves average safety/cost metrics compared with naive PPO. However, it does not robustly dominate naive PPO across six paired training seeds. Axes 2-5 explain why: aggressive failure replay is risky in the current screen, the current adaptive budget is weaker than a fixed 0.05 budget, the default driving cost is the best balanced cost signal, and near-failure scoring is only marginally better than default scoring on the canonical distribution.

The current evidence supports a cautious claim:

```text
Stable FASB-PPO improves average safety-efficiency and cost metrics, but the improvement is seed-sensitive and trades off slightly against success, timeout, and route completion.
```

The contribution is therefore both methodological and empirical. Methodologically, the project provides a reusable, auditable framework for safety-specialized PPO experiments in MetaDrive. Empirically, it shows that the simplest safe-RL story is wrong: more replay, more adaptivity, and more near-miss shaping do not automatically improve driving behavior. The best current configuration is a calibrated, conservative system whose limitations are visible in the results rather than hidden by a single selected checkpoint.

The strongest remaining limitation is failure-buffer coverage. Future versions should regenerate or extend the canonical buffer so every multiseed training range has eligible failure replay examples, then repeat the paired Axis 1 and Axis 5 comparisons.

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
research/research_v1/
research/research_v1/charts/
research/research_v1/axis1/results/summary/
research/research_v1/axis2/results/summary/
research/research_v1/axis3/reports/
research/research_v1/axis4/reports/
research/research_v1/axis5/reports/
```

### Appendix B: Active Config Families

```text
configs/research_v1/axis1_naive_stable_final.yaml
configs/research_v1/axis1_fasb_stable_final.yaml
configs/research_v1/axis2_sampler_mixed060_final.yaml
configs/research_v1/axis3_budget_adaptive_default_final.yaml
configs/research_v1/axis3_budget_fixed_default_final.yaml
configs/research_v1/axis4_cost_default_final.yaml
configs/research_v1/axis5_default_scorer_final.yaml
configs/research_v1/axis5_near_failure_scorer_final.yaml
```

### Appendix C: Reviewer Reproduction

The reviewer package mirrors the paper structure:

```text
research/research_v1/axis*/configs/
research/research_v1/axis*/results/
research/research_v1/axis*/reports/
```

The raw model checkpoints are intentionally not stored in the paper package. They are reproduced from the canonical foundation release and the listed configs.
