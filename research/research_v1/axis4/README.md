# Axis 4 Package: Cost Function Ablation

Axis 4 studies how the **definition of the safety cost function** affects FASB-PPO under the stable multiseed protocol. All other components (sampler, budget, penalty, optimizer, base checkpoint, failure buffer, seeds, selection, evaluation) are held fixed.

## Main Question

Does the safety cost definition change the safety-efficiency tradeoff, and is richer pre-crash (near-miss) cost worth the added conservatism?

## Allowed Research Variable

```text
cost_function.*
```

Keep base checkpoint, failure buffer, sampler, budget, penalty schedule, optimizer, train seeds, dev selection, and final heldout evaluation fixed.

## Methods

| method | config template |
| --- | --- |
| Default driving cost | `configs/templates/axis4_cost_default_final.yaml` |
| Crash-only cost | `configs/templates/axis4_cost_crash_only_final.yaml` |
| Near-miss-heavy cost | `configs/templates/axis4_cost_nearmiss_heavy_final.yaml` |
| Event-driving cost | `configs/templates/axis4_cost_event_driving_final.yaml` |

## Protocol

Training seeds:

```text
42, 2000, 3000
```

Seeds vary the RNG only; training `start_seed=2000` is fixed across seeds.

For each seed and method:

1. Train for `300000` timesteps from the same base checkpoint.
2. Select checkpoint on dev seeds only: `start_seed=4500`, `num_scenarios=100`, `eval.n_episodes=100`.
3. Evaluate selected checkpoint once on final heldout: `start_seed=5000`, `num_scenarios=200`, `eval.n_episodes=100`.

## Subfolders

```text
reports/
  axis4_report.md
  axis4_summary.csv
  axis4_per_seed.csv

configs/templates/
  active config templates, one per cost variant

configs/resolved_train/
  resolved train configs for every variant/seed run

configs/resolved_eval/
  resolved final heldout eval configs for every variant/seed run

results/summary/
  per-seed aggregate CSV

results/checkpoint_selection/
  dev checkpoint-selection CSVs and reports

results/final_eval/
  final heldout eval CSVs

results/failure_analysis/
  failure summaries, failure-by-mode CSVs, and paper-number snippets
```

## Main Result

Across three training seeds, `DefaultDrivingCost` gives the best mean safety-efficiency score (−0.417), highest success, and lowest timeout. `NearMissHeavyCost` reaches the lowest collision rate (0.213) but trades it for higher timeout and lower success (conservatism). `CrashOnlyCost` is clearly worst, showing the pre-crash term carries real learning signal. `EventDrivingCost` minimizes episode cost but does not beat the default on safety-efficiency.

Use `reports/axis4_report.md` for the full interpretation.
