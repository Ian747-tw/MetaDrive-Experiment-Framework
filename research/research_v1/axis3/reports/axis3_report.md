# Axis 3 Research Report: Budget/Penalty Scaling Analysis

**Status**: Axis 3 is complete for the current stable protocol. The main conclusion is that fixed safety budgets (especially `fixed005`) consistently outperform adaptive budget variants on safety-efficiency score, with fixed budgets producing higher success rates, lower offroad rates, and lower variance across seeds. Adaptive budget variants show instability and occasional collapse, particularly under default adaptive settings.

---

## Research Question

Axis 3 asks whether **adaptive safety budget and penalty scheduling** improves safety-efficiency compared to **fixed global penalty baselines**.

The evaluated hypothesis is:

```
Dynamic adjustment of the safety budget (AdaptiveSafetyBudget) leads to better
safety-efficiency balance across diverse training seeds than static fixed-budget
configurations (FixedSafetyBudget), by applying tighter pressure when the agent
is over-budget and relaxing it when the agent is within budget.
```

---

## Locked Protocol

| Item | Value |
|---|---|
| Base checkpoint | GitHub release artifact; local after download: `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Failure buffer | GitHub release artifact; local after download: `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Fine-tuning steps | 300,000 |
| Training seed | 42 (per-rep identical seed, varied via rep index) |
| Repetitions per variant | 10 |
| Selection metric | `safety_efficiency_score` |
| Sampler | `MixedFailureSampler`, `failure_ratio=0.05`, `alpha=0.7` |
| Optimizer | `lr=3e-5`, `n_steps=128`, `batch_size=64`, `n_epochs=10` |
| Eval protocol | heldout random, `start_seed=5000`, `num_scenarios=200`, `horizon=500`, 100 episodes |
| Axis variable | `safety_budget.*`, `penalty_scheduler.*` only |

---

## Methods Compared

| Method | Description | Key Config |
|---|---|---|
| `adaptive_default` | Adaptive budget, moderate bounds | `d_min=0.10`, `d_max=0.30`, `timeout_budget=0.30`, `λ_max=0.25` |
| `adaptive_loose` | Adaptive budget, relaxed bounds | `d_min=0.15`, `d_max=0.40`, `timeout_budget=0.40`, `λ_max=0.10` |
| `adaptive_strict` | Adaptive budget, tight bounds | `d_min=0.02`, `d_max=0.10`, `timeout_budget=0.07`, `λ_max=0.50` |
| `fixed003` | Fixed budget = 0.03 | `FixedSafetyBudget`, `budget=0.03`, `λ_max=0.10` |
| `fixed005` | Fixed budget = 0.05 | `FixedSafetyBudget`, `budget=0.05`, `λ_max=0.10` |
| `fixed010` | Fixed budget = 0.10 | `FixedSafetyBudget`, `budget=0.10`, `λ_max=0.10` |

---

## Ten-Run Final Multiseed Result

Source: `reports/axis3_results_summary.csv` (mean ± std over 10 repetitions)

| Method | success_rate | collision_rate | offroad_rate | timeout_rate | route_completion_mean | episode_cost_mean | safety_efficiency_score |
|---|---|---|---|---|---|---|---|
| `fixed005`        | 0.487 ± 0.068 | 0.260 ± 0.065 | 0.386 ± 0.069 | 0.513 ± 0.068 | 0.707 ± 0.040 | 244.83 ± 20.56 | **−0.416 ± 0.166** |
| `fixed003`        | 0.456 ± 0.070 | 0.239 ± 0.045 | 0.424 ± 0.085 | 0.545 ± 0.071 | 0.708 ± 0.036 | 252.41 ± 24.58 | **−0.480 ± 0.177** |
| `fixed010`        | 0.451 ± 0.068 | 0.275 ± 0.070 | 0.426 ± 0.083 | 0.549 ± 0.068 | 0.695 ± 0.025 | 252.82 ± 21.27 | **−0.525 ± 0.156** |
| `adaptive_strict` | 0.442 ± 0.051 | 0.229 ± 0.046 | 0.475 ± 0.062 | 0.560 ± 0.051 | 0.682 ± 0.039 | 246.60 ± 16.16 | **−0.542 ± 0.098** |
| `adaptive_loose`  | 0.422 ± 0.076 | 0.235 ± 0.050 | 0.479 ± 0.095 | 0.578 ± 0.076 | 0.669 ± 0.037 | 271.73 ± 20.76 | **−0.581 ± 0.183** |
| `adaptive_default`| 0.373 ± 0.122 | 0.173 ± 0.056 | 0.534 ± 0.154 | 0.627 ± 0.122 | 0.667 ± 0.034 | 275.03 ± 48.84 | **−0.648 ± 0.295** |

**Best single run overall**: `fixed005` rep08, safety-efficiency score = −0.125  
(Source: `reports/axis3_suite_best_overall.txt`)

### Paired Deltas vs. `adaptive_default` (baseline adaptive)

| Comparison | ΔSES | ΔSuccess | ΔOffroad | ΔTimeout | ΔCost |
|---|---|---|---|---|---|
| `fixed005` vs `adaptive_default` | **+0.232** | +0.114 | −0.148 | −0.114 | −30.20 |
| `fixed003` vs `adaptive_default` | **+0.168** | +0.083 | −0.110 | −0.082 | −22.62 |
| `fixed010` vs `adaptive_default` | **+0.123** | +0.078 | −0.108 | −0.078 | −22.21 |
| `adaptive_strict` vs `adaptive_default` | **+0.106** | +0.069 | −0.059 | −0.067 | −28.43 |
| `adaptive_loose` vs `adaptive_default` | **+0.067** | +0.049 | −0.055 | −0.049 | −3.30 |

*Positive ΔSES = better (less negative). All fixed variants outperform all adaptive variants.*

---

## Main Interpretation

### Does it beat the baseline?

**No — the adaptive budget does not outperform fixed budget in this setting.**  
The original hypothesis (adaptive > fixed) is **not supported**. All three fixed-budget variants rank above all three adaptive-budget variants by mean safety-efficiency score. `fixed005` leads with a mean SES of −0.416, versus −0.648 for `adaptive_default`. There is no clean domination within the fixed group either: `fixed010` performs worse than `fixed005`, showing that a higher budget tolerance is not always better.

### Does it improve safety?

**Partially — but not cleanly.**

- **Collision rate**: `adaptive_default` has the *lowest* collision rate (0.173), but this is a false signal — it is accompanied by the highest timeout rate (0.627), meaning the agent avoids collisions by refusing to drive forward.
- **Offroad rate**: Fixed variants consistently produce lower offroad rates (0.386–0.426) than adaptive variants (0.475–0.534). This is a genuine safety improvement.
- **Episode cost**: `fixed005` achieves the lowest mean episode cost (244.83) among all variants, below even `adaptive_strict` (246.60).
- **Adaptive-default instability**: rep08 collapsed (SES = −1.405, timeout rate = 0.91), inflating mean cost and dragging down the group average. The adaptive mechanism can overcorrect, causing the agent to time out rather than engage.

### Does it preserve progress?

**Fixed budget preserves progress better.**

- `fixed005` achieves the highest success rate (0.487) and ties for the highest route completion (0.707).
- All adaptive variants have lower success rates (0.373–0.442) and lower route completion (0.667–0.682) than the top fixed variants.
- The stricter the adaptive budget, the more the agent hesitates: `adaptive_strict` has a 0.560 timeout rate vs 0.513 for `fixed005`.
- `adaptive_default` shows the highest timeout rate (0.627) of all variants, confirming that overly dynamic penalty feedback can suppress progress without meaningfully reducing collisions.

---

## Recommended Config

**`fixed005`** (`FixedSafetyBudget`, `budget=0.05`, `λ_max=0.10`) is the recommended budget/penalty configuration for downstream axes that build on Axis 3 findings. It achieves the best mean safety-efficiency score, the highest success rate, and the lowest offroad rate, with competitive episode cost and moderate variance.

> All conclusions are bounded to 300k fine-tuning steps, the canonical large failure buffer (`base_explore_large`), and the stable FASB-PPO optimizer protocol. Results may differ under different buffer sizes, timestep budgets, or base checkpoints.

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Suite results (all reps) | `reports/axis3_suite_results_s42_10reps.csv` |
| Per-variant summary (mean ± std) | `reports/axis3_results_summary.csv` |
| Best overall run | `reports/axis3_suite_best_overall.txt` |
| Final eval CSVs | `results/final_eval/` |
| Failure analysis | `results/failure_analysis/` |
| Resolved train configs | `configs/resolved_train/` |
| Resolved eval configs | `configs/resolved_eval/` |
