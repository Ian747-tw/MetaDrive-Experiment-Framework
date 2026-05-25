# Axis 5 Research Report: Failure Scorer and Generalization

**Status**: Axis 5 is complete for the current stable protocol over four training seeds (42, 2000, 3000, 4000). The main conclusion is that the `NearFailureScorer` is marginally better than the `DefaultFailureScorer` on mean safety-efficiency, success, offroad, and cost, while the default scorer is lower only on collision; the difference is within one standard deviation and is **not robust by paired seed comparison** (2 wins / 2 losses on safety-efficiency). A supplementary traffic-distribution-shift study (seeds 42, 2000) shows both scorers degrade on easy traffic and behave similarly under dense traffic.

FASB-PPO in this project means:

```text
SB3 PPO + failure-aware sampler + adaptive safety penalty
```

It is not PPO-Lagrangian and does not claim constrained-RL guarantees.

---

## Research Question

Axis 5 asks whether the **definition of the failure score** (which drives failure-buffer prioritization and the adaptive risk signal) changes the safety-efficiency tradeoff of FASB-PPO, and how the resulting policies behave under **traffic-distribution shift**, when all other components are held fixed.

The evaluated hypothesis is:

```
A near-failure-oriented scorer that up-weights near-miss and partial-progress
signals discovers more learnable failures than the default collision/offroad-heavy
scorer, improving safety-efficiency without hiding regressions, and the advantage
should hold under traffic-density shift.
```

---

## Locked Protocol

| Item | Value |
|---|---|
| Base checkpoint | GitHub release artifact; local after download: `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Failure buffer | GitHub release artifact; local after download: `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Fine-tuning steps | 300,000 |
| Training seeds | 42, 2000, 3000, 4000 |
| Cost function | `DefaultDrivingCost` |
| Sampler | `MixedFailureSampler`, `failure_ratio=0.05`, `alpha=0.7`, `max_too_hard_ratio=0.15` |
| Safety budget | `AdaptiveSafetyBudget`, `d_min=0.10`, `d_max=0.30`, `timeout_budget=0.30` |
| Penalty | `RiskPenaltyScheduler`, `lambda_min=0.0`, `lambda_max=0.25` |
| Optimizer | `lr=3e-5`, `n_steps=128`, `batch_size=64`, `n_epochs=10` |
| Selection metric | `safety_efficiency_score` |
| Dev selection | `start_seed=4500`, `num_scenarios=100`, 100 episodes |
| Eval protocol | heldout random, `start_seed=5000`, `num_scenarios=200`, `horizon=500`, 100 episodes |
| Axis variable | `failure_scorer.*` (and labeled traffic-density shift in the supplementary study) |

> **Seed protocol caveat.** Seeds 42 and 2000 were trained locally with the RNG seed varied and training `start_seed=2000` fixed (optimization-noise replicates). Seeds 3000 and 4000 were trained by teammates with `start_seed` following the seed (3000, 4000), so they also vary the training scenario set (data + optimization replicates). The 4-seed mean ± std below combines both replicate types as requested; treat the spread as conservative rather than a clean single-protocol variance.

---

## Methods Compared

| Method | Description | `failure_scorer._target_` |
|---|---|---|
| `default_scorer` | Collision/offroad-heavy scorer: `5·collision + 4·offroad + 2·timeout + 2·near_miss + (1−route)` | `fasb.plugins.failure_scorer.DefaultFailureScorer` |
| `near_failure_scorer` | Near-failure scorer: `3·near_miss + 2·partial_progress + 1.5·timeout + 1·collision + 1·offroad` | `examples.custom_plugins.near_failure_scorer.NearFailureScorer` |

---

## Four-Run Final Multiseed Result

Source: `reports/axis5_summary.csv` (mean ± std over seeds 42, 2000, 3000, 4000)

| Method | success_rate | collision_rate | offroad_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
|---|---|---|---|---|---|---|---|
| `near_failure_scorer` | 0.453 ± 0.070 | 0.260 ± 0.039 | 0.445 ± 0.058 | 0.550 ± 0.067 | 0.687 ± 0.031 | 252.78 ± 13.44 | **−0.528 ± 0.134** |
| `default_scorer`      | 0.433 ± 0.051 | 0.228 ± 0.044 | 0.488 ± 0.045 | 0.568 ± 0.051 | 0.681 ± 0.025 | 256.29 ± 19.57 | **−0.566 ± 0.100** |

**Best single run overall**: `near_failure_scorer` seed 4000, safety-efficiency score = −0.375  
**Lowest collision single run**: `default_scorer` seed 42, collision_rate = 0.19

### Per-Seed Selected Checkpoint and Final Heldout Metrics

| seed | method | selected checkpoint | success | collision | offroad | timeout | route | cost | safety_eff |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | default | `latest_200000_steps.zip` | 0.36 | 0.19 | 0.55 | 0.64 | 0.650 | 253.86 | −0.700 |
| 2000 | default | `latest_200000_steps.zip` | 0.44 | 0.21 | 0.49 | 0.56 | 0.674 | 270.86 | −0.540 |
| 3000 | default | `latest_300000_steps.zip` | 0.48 | 0.22 | 0.46 | 0.52 | 0.706 | 229.51 | −0.460 |
| 4000 | default | `final.zip` | 0.45 | 0.29 | 0.45 | 0.55 | 0.695 | 270.91 | −0.565 |
| 42 | near_failure | `latest_100000_steps.zip` | 0.44 | 0.30 | 0.43 | 0.56 | 0.661 | 257.49 | −0.570 |
| 2000 | near_failure | `latest_100000_steps.zip` | 0.36 | 0.25 | 0.48 | 0.64 | 0.660 | 269.61 | −0.690 |
| 3000 | near_failure | `latest_200000_steps.zip` | 0.49 | 0.21 | 0.50 | 0.51 | 0.708 | 240.54 | −0.475 |
| 4000 | near_failure | `latest_100000_steps.zip` | 0.52 | 0.28 | 0.37 | 0.49 | 0.720 | 243.47 | −0.375 |

### Paired Deltas (near_failure − default) per Seed

| seed | ΔSES | ΔSuccess | ΔCollision | ΔOffroad | ΔTimeout | ΔCost |
|---|---:|---:|---:|---:|---:|---:|
| 42 | **+0.130** | +0.08 | +0.11 | −0.12 | −0.08 | +3.63 |
| 2000 | **−0.150** | −0.08 | +0.04 | −0.01 | +0.08 | −1.25 |
| 3000 | **−0.015** | +0.01 | −0.01 | +0.04 | −0.01 | +11.03 |
| 4000 | **+0.190** | +0.07 | −0.01 | −0.08 | −0.06 | −27.44 |

Paired safety-efficiency: **2 near_failure wins / 0 ties / 2 losses**, mean ΔSES = +0.039. The near-failure scorer is better on the mean but not robust across seeds.

### Failure-Mode Evidence (seed 42, illustrative)

Source: `results/failure_analysis/*_failure_by_mode.csv` (episode counts by dominant failure mode)

| method | solved | collision | offroad | timeout_or_hesitation |
|---|---:|---:|---:|---:|
| `default_scorer`      | 27 | 19 | 39 | 9 |
| `near_failure_scorer` | 25 | 30 | 31 | 11 |

The default scorer produces more offroad-dominated failures (39 vs 31), consistent with its higher mean offroad rate (0.488 vs 0.445). The near-failure scorer shifts failures away from offroad but toward collision in this seed, which is why its collision mean is slightly higher.

---

## Supplementary: Traffic-Distribution Shift

The same selected checkpoints were evaluated under shifted traffic density. This study uses only the local seeds **42 and 2000** (n = 2), because the teammate seeds 3000/4000 do not have shift evaluations. Numbers are means over those two seeds; the canonical column here therefore differs from the 4-seed table above.

Eval ranges: easy `start_seed=8000`, `traffic_density=0.05`; canonical `start_seed=5000`, `0.1`; dense `start_seed=7000`, `0.2`. All `num_scenarios=200`, 100 episodes.

| method | condition | success | collision | offroad | timeout | safety_eff |
|---|---|---:|---:|---:|---:|---:|
| `default_scorer` | easy (0.05) | 0.355 | 0.185 | 0.560 | 0.645 | −0.713 |
| `default_scorer` | canonical (0.1) | 0.400 | 0.200 | 0.520 | 0.600 | −0.620 |
| `default_scorer` | dense (0.2) | 0.460 | 0.310 | 0.430 | 0.540 | −0.550 |
| `near_failure_scorer` | easy (0.05) | 0.365 | 0.250 | 0.515 | 0.635 | −0.718 |
| `near_failure_scorer` | canonical (0.1) | 0.400 | 0.275 | 0.455 | 0.600 | −0.630 |
| `near_failure_scorer` | dense (0.2) | 0.425 | 0.365 | 0.420 | 0.580 | −0.650 |

Both scorers **degrade on easy traffic** (lower success, higher offroad and timeout) rather than improve — with few surrounding vehicles, failures are dominated by offroad/route errors that the safety-oriented training does not target, so safety-efficiency is worst at density 0.05. Under dense traffic, `default_scorer` actually improves (more cars give clearer collision-avoidance signal, SES −0.550), while `near_failure_scorer` worsens (collision rises to 0.365). The near-failure scorer's advantage on the canonical 4-seed table does **not** carry over to the shifted distributions on these two seeds.

---

## Main Interpretation

### Does the near-failure scorer beat the default scorer?

**On the mean, marginally; not robustly.** Over four seeds, `near_failure_scorer` has better mean safety-efficiency (−0.528 vs −0.566), success (0.453 vs 0.433), offroad (0.445 vs 0.488), route completion (0.687 vs 0.681), and cost (252.78 vs 256.29). But every gap is within one standard deviation, the paired safety-efficiency record is 2–2, and the near-failure scorer has higher variance (±0.134 vs ±0.100). The hypothesis is weakly supported at best.

### Does it improve safety?

**Mixed.** The near-failure scorer lowers offroad (0.445 vs 0.488) but raises collision (0.260 vs 0.228). The default scorer is the collision-minimizing choice; the near-failure scorer is the offroad-minimizing choice. Neither dominates both safety dimensions. Episode cost is essentially tied (252.78 vs 256.29).

### Does it preserve progress?

**Yes, slightly better.** The near-failure scorer has marginally higher success and route completion and lower timeout, consistent with its partial-progress credit term steering the policy toward completing routes rather than stalling. Neither scorer collapses into timeout.

### Does the result generalize under distribution shift?

**No clear generalization.** On seeds 42/2000, both scorers are worst on easy traffic and the near-failure scorer's canonical edge disappears under dense traffic. Safety-aware fine-tuning helps most where collision risk is present (dense), not where the dominant failure is offroad/route error (easy). This is the most reviewer-relevant caveat: the scorer choice does not produce a distribution-robust safety win.

---

## Recommended Config

For the safety-efficiency objective, `NearFailureScorer` is the marginally better default on the canonical heldout set and is the recommended scorer when offroad and route completion matter most. If the objective is minimizing collisions specifically, `DefaultFailureScorer` is preferable. Given the within-noise gap and the lack of distribution-shift robustness, the honest framing is that **scorer choice is a second-order knob**, not a decisive lever, under the current stable protocol.

> All conclusions are bounded to 300k fine-tuning steps, the canonical large failure buffer, the stable FASB-PPO optimizer protocol, and the four training seeds described above (two RNG-only replicates plus two data+RNG replicates). The shift study covers only seeds 42 and 2000. `cost_violation_rate` is 0.0 for every row because heldout evaluation does not attach the training-time budget/penalty wrapper, so it is not a usable comparison column.

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Per-variant summary (mean ± std) | `reports/axis5_summary.csv` |
| Per-seed table (4 seeds) | `reports/axis5_per_seed.csv` (also `results/summary/axis5_per_seed_4seeds.csv`) |
| Earlier teammate partial (seeds 3000/4000) | `results/summary/axis5_partial_seed3000_4000_per_seed.csv` |
| Final eval CSVs (canonical + dense/easy shift) | `results/final_eval/` |
| Failure analysis | `results/failure_analysis/` |
| Checkpoint selection | `results/checkpoint_selection/` |
| Config templates | `configs/templates/` |
| Resolved train configs | `configs/resolved_train/` |
| Resolved eval configs | `configs/resolved_eval/` |
