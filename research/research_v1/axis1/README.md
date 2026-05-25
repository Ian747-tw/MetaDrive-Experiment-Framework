# Axis 1 Package

Axis 1 compares stable naive PPO fine-tuning with stable FASB-PPO under the newest multiseed protocol.

## Main Question

Does failure-aware scenario replay plus adaptive safety penalty improve safety-efficiency over normal PPO fine-tuning without causing timeout or low-progress collapse?

## Methods

| method | config template |
| --- | --- |
| Naive PPO stable | `configs/templates/axis1_naive_stable_final.yaml` |
| Stable FASB-PPO | `configs/templates/axis1_fasb_stable_final.yaml` |

Stable FASB-PPO means:

```text
SB3 PPO + failure-aware sampler + adaptive safety penalty
```

It is not PPO-Lagrangian.

## Protocol

Training seeds:

```text
2000, 3000, 4000, 6000, 7000, 8000
```

For each seed and method:

1. Train for `300000` timesteps from the same base checkpoint.
2. Select checkpoint on dev seeds only: `start_seed=4500`, `num_scenarios=100`, `eval.n_episodes=100`.
3. Evaluate selected checkpoint once on final heldout: `start_seed=5000`, `num_scenarios=200`, `eval.n_episodes=100`.

## Subfolders

```text
reports/
  axis1_report.md
  axis1_summary.csv

configs/templates/
  active config templates used for the newest multiseed research

configs/resolved_train/
  resolved train configs for every method/seed run

configs/resolved_eval/
  resolved final heldout eval configs for every method/seed run

results/summary/
  multiseed aggregate CSVs

results/checkpoint_selection/
  dev checkpoint-selection CSVs and reports

results/final_eval/
  final heldout aggregate eval CSVs

results/failure_analysis/
  failure summaries, failure-by-mode CSVs, and paper-number snippets

others/
  reproduction commands and reviewer notes
```

## Main Result

Across six paired training seeds, stable FASB-PPO improves average collision, offroad, episode cost, and safety-efficiency, but naive PPO is slightly better on average success, timeout, and route completion. The result is mixed and should not be stated as a robust FASB win.

Use `reports/axis1_report.md` for the full interpretation.
