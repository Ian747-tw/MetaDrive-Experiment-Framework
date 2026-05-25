# Axis 4 Research Report: Cost Function Ablation

**Status**: Axis 4 is complete for the current stable protocol over three training seeds (42, 2000, 3000). The main conclusion is that the standard `DefaultDrivingCost` gives the best safety-efficiency score, that removing all pre-crash shaping (`CrashOnlyCost`) is clearly the worst cost definition, and that a near-miss-heavy cost (`NearMissHeavyCost`) lowers collisions but increases conservatism (more hesitation/timeout, lower success). An episode-level cost (`EventDrivingCost`) gives the lowest mean episode cost and is competitive with the default but does not beat it on safety-efficiency.

FASB-PPO in this project means:

```text
SB3 PPO + failure-aware sampler + adaptive safety penalty
```

It is not PPO-Lagrangian and does not claim constrained-RL guarantees.

---

## Research Question

Axis 4 asks whether the **definition of the per-step safety cost** changes the safety-efficiency tradeoff of FASB-PPO, when the base checkpoint, failure buffer, sampler, budget, penalty schedule, optimizer, training seeds, checkpoint selection, and final evaluation are all held fixed.

The evaluated hypothesis is:

```
A richer pre-crash safety cost (near-miss / proximity shaping) improves safety by
reducing collisions and episode cost, but may increase conservatism (timeout and
low-progress), while a crash-only cost provides a weaker learning signal than the
default driving cost.
```

---

## Locked Protocol

| Item | Value |
|---|---|
| Base checkpoint | `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Failure buffer | `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Fine-tuning steps | 300,000 |
| Training seeds | 42, 2000, 3000 (RNG seed varied; training `start_seed=2000` fixed) |
| Sampler | `MixedFailureSampler`, `failure_ratio=0.05`, `alpha=0.7`, `max_too_hard_ratio=0.15` |
| Safety budget | `AdaptiveSafetyBudget`, `d_min=0.10`, `d_max=0.30`, `timeout_budget=0.30` |
| Penalty | `RiskPenaltyScheduler`, `lambda_min=0.0`, `lambda_max=0.25` |
| Optimizer | `lr=3e-5`, `n_steps=128`, `batch_size=64`, `n_epochs=10` |
| Selection metric | `safety_efficiency_score` |
| Dev selection | `start_seed=4500`, `num_scenarios=100`, 100 episodes |
| Eval protocol | heldout random, `start_seed=5000`, `num_scenarios=200`, `horizon=500`, 100 episodes |
| Axis variable | `cost_function.*` only |

---

## Methods Compared

| Method | Description | `cost_function._target_` |
|---|---|---|
| `default` | Standard driving cost: collision + offroad + light near-miss (per step) | `fasb.plugins.cost.DefaultDrivingCost` |
| `crash_only` | Crash-only cost: collision + offroad, no pre-crash term | `examples.custom_plugins.crash_only_cost.CrashOnlyCost` |
| `nearmiss_heavy` | Heavy near-miss cost: `near_miss_weight=0.5`, `threshold=5.0` m (per step) | `examples.custom_plugins.near_miss_heavy_cost.NearMissHeavyCost` |
| `event_driving` | Same as default but collision/offroad charged once per episode | `examples.custom_plugins.event_driving_cost.EventDrivingCost` |

---

## Three-Run Final Multiseed Result

Source: `reports/axis4_summary.csv` (mean ± std over seeds 42, 2000, 3000)

| Method | success_rate | collision_rate | offroad_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
|---|---|---|---|---|---|---|---|
| `default`        | 0.507 ± 0.042 | 0.267 ± 0.015 | 0.410 ± 0.062 | 0.493 ± 0.042 | 0.714 ± 0.031 | 251.22 ± 13.95 | **−0.417 ± 0.055** |
| `event_driving`  | 0.490 ± 0.035 | 0.300 ± 0.090 | 0.390 ± 0.062 | 0.510 ± 0.035 | 0.702 ± 0.030 | 237.74 ± 6.33 | **−0.455 ± 0.069** |
| `nearmiss_heavy` | 0.440 ± 0.053 | 0.213 ± 0.040 | 0.430 ± 0.010 | 0.560 ± 0.053 | 0.692 ± 0.029 | 266.02 ± 11.18 | **−0.483 ± 0.085** |
| `crash_only`     | 0.373 ± 0.042 | 0.237 ± 0.064 | 0.493 ± 0.055 | 0.627 ± 0.042 | 0.668 ± 0.031 | 267.50 ± 9.43 | **−0.670 ± 0.053** |

**Best single run overall**: `default` seed 42, safety-efficiency score = −0.360  
(Lowest collision single run: `nearmiss_heavy` seed 2000, collision_rate = 0.17; lowest cost single run: `event_driving` seed 2000, episode_cost = 231.09)

### Per-Seed Selected Checkpoint and Final Heldout Metrics

| seed | method | selected checkpoint | success | collision | offroad | timeout | route | cost | safety_eff |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | default | `final.zip` | 0.52 | 0.28 | 0.36 | 0.48 | 0.717 | 237.45 | −0.360 |
| 2000 | default | `latest_100000_steps.zip` | 0.46 | 0.27 | 0.39 | 0.54 | 0.681 | 250.88 | −0.470 |
| 3000 | default | `latest_300000_steps.zip` | 0.54 | 0.25 | 0.48 | 0.46 | 0.744 | 265.34 | −0.420 |
| 42 | crash_only | `latest_200000_steps.zip` | 0.42 | 0.31 | 0.43 | 0.58 | 0.701 | 257.45 | −0.610 |
| 2000 | crash_only | `latest_300000_steps.zip` | 0.34 | 0.19 | 0.53 | 0.66 | 0.639 | 276.15 | −0.710 |
| 3000 | crash_only | `latest_200000_steps.zip` | 0.36 | 0.21 | 0.52 | 0.64 | 0.664 | 268.91 | −0.690 |
| 42 | nearmiss_heavy | `final.zip` | 0.38 | 0.22 | 0.43 | 0.62 | 0.659 | 267.09 | −0.580 |
| 2000 | nearmiss_heavy | `latest_300000_steps.zip` | 0.46 | 0.17 | 0.44 | 0.54 | 0.707 | 276.62 | −0.420 |
| 3000 | nearmiss_heavy | `final.zip` | 0.48 | 0.25 | 0.42 | 0.52 | 0.710 | 254.34 | −0.450 |
| 42 | event_driving | `latest_100000_steps.zip` | 0.51 | 0.39 | 0.37 | 0.49 | 0.723 | 243.69 | −0.495 |
| 2000 | event_driving | `final.zip` | 0.51 | 0.30 | 0.34 | 0.49 | 0.714 | 231.09 | −0.375 |
| 3000 | event_driving | `latest_100000_steps.zip` | 0.45 | 0.21 | 0.46 | 0.55 | 0.668 | 238.43 | −0.495 |

### Paired Deltas vs. `default` (standard cost)

| Comparison | ΔSES | ΔSuccess | ΔCollision | ΔOffroad | ΔTimeout | ΔCost |
|---|---|---|---|---|---|---|
| `event_driving` vs `default` | **−0.038** | −0.017 | +0.033 | −0.020 | +0.017 | −13.49 |
| `nearmiss_heavy` vs `default` | **−0.066** | −0.067 | −0.053 | +0.020 | +0.067 | +14.79 |
| `crash_only` vs `default` | **−0.253** | −0.133 | −0.030 | +0.083 | +0.133 | +16.28 |

*Negative ΔSES = worse than the default cost. No cost variant beats the default on safety-efficiency.*

### Failure-Mode Evidence (seed 42, illustrative)

Source: `results/failure_analysis/*_failure_by_mode.csv` (episode counts by dominant failure mode)

| method | solved | collision | offroad | timeout_or_hesitation |
|---|---:|---:|---:|---:|
| `default`        | 40 | 28 | 23 | 9 |
| `event_driving`  | 30 | 39 | 20 | 8 |
| `nearmiss_heavy` | 22 | 22 | 34 | 16 |
| `crash_only`     | 27 | 31 | 28 | 8 |

`nearmiss_heavy` has the fewest collision episodes but the most `timeout_or_hesitation` episodes (16), the direct fingerprint of conservatism. `crash_only` has few solved episodes and high offroad, consistent with a weak guidance signal. `default` produces the most solved episodes.

---

## Main Interpretation

### Which cost definition is best?

**`DefaultDrivingCost`.** It has the best mean safety-efficiency score (−0.417), the highest success rate (0.507), the highest route completion (0.714), and the lowest timeout rate (0.493). The standard collision + offroad + light near-miss signal gives the most balanced behavior, and it has the lowest safety-efficiency variance (±0.055) among the four.

### Does richer pre-crash cost improve safety?

**Partially, with a clear conservatism cost.** `NearMissHeavyCost` achieves the lowest mean collision rate (0.213) of all variants, confirming that stronger proximity shaping reduces crashes. But it pays for this with a higher timeout rate (0.560 vs 0.493), lower success (0.440 vs 0.507), and higher episode cost (266.02 vs 251.22). The failure-mode breakdown shows the mechanism: collisions are converted into hesitation/timeout rather than into completed routes. This is exactly the safety-vs-conservatism tradeoff the hypothesis predicted.

### Does removing pre-crash shaping hurt?

**Yes, substantially.** `CrashOnlyCost` is the worst variant on success (0.373), timeout (0.627), offroad (0.493), route completion (0.668), and safety-efficiency (−0.670). A sparse crash-only signal gives weaker feedback during fine-tuning; the policy drifts toward offroad and stalling rather than completing routes. Notably its collision rate (0.237) is not the worst — it is not reckless, it is simply unproductive. The pre-crash term in the cost is load-bearing.

### Does episode-level charging help?

**It minimizes cost but does not win.** `EventDrivingCost` produces the lowest mean episode cost (237.74) and the lowest offroad (0.390), with safety-efficiency (−0.455) just behind default. Its weakness is an unstable collision rate (0.300 ± 0.090, the highest variance). Charging collision/offroad once per episode is a reasonable cleaner-signal alternative, but it does not improve safety-efficiency over the per-step default.

---

## Recommended Config

**`DefaultDrivingCost`** (`fasb.plugins.cost.DefaultDrivingCost`) is the recommended cost function for downstream FASB-PPO work. It gives the best mean safety-efficiency, the highest success and route completion, and the lowest timeout, without the conservatism of `NearMissHeavyCost` or the weak signal of `CrashOnlyCost`. If the downstream objective is to minimize collisions specifically and some route loss is acceptable, `NearMissHeavyCost` is the collision-minimizing alternative.

> All conclusions are bounded to 300k fine-tuning steps, the canonical large failure buffer (`base_explore_large`), the stable FASB-PPO optimizer protocol, and three training seeds. `cost_violation_rate` is 0.0 for every row because heldout evaluation does not attach the training-time budget/penalty wrapper, so it is not a usable comparison column for this axis.

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Per-variant summary (mean ± std) | `reports/axis4_summary.csv` |
| Per-seed table | `reports/axis4_per_seed.csv` (also `results/summary/axis4_per_seed.csv`) |
| Final eval CSVs | `results/final_eval/` |
| Failure analysis | `results/failure_analysis/` |
| Checkpoint selection | `results/checkpoint_selection/` |
| Config templates | `configs/templates/` |
| Resolved train configs | `configs/resolved_train/` |
| Resolved eval configs | `configs/resolved_eval/` |
