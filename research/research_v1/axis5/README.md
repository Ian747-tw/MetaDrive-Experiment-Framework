# Axis 5 Package: Failure Scorer and Generalization

Axis 5 studies how the **failure score definition** affects FASB-PPO and how the resulting policies behave under **traffic-distribution shift**, under the stable multiseed protocol with all other components fixed.

## Main Question

Does a near-failure-oriented scorer discover more learnable failures and improve safety-efficiency over the default collision/offroad-heavy scorer, and does any advantage hold under traffic-density shift?

## Allowed Research Variables

```text
failure_scorer.*
failure_classifier.*
explicitly labeled eval distribution changes
explicitly labeled buffer/generalization changes
```

Keep the stable optimizer, sampler, budget, penalty, and total training budget fixed unless the variant is explicitly labeled as a distribution-shift or buffer experiment.

## Methods

| method | config template |
| --- | --- |
| Default failure scorer | `configs/templates/axis5_default_scorer_final.yaml` |
| Near-failure scorer | `configs/templates/axis5_near_failure_scorer_final.yaml` |

## Protocol

Training seeds:

```text
42, 2000, 3000, 4000
```

Seed protocol note: seeds 42 and 2000 vary the RNG only with training `start_seed=2000` fixed (optimization-noise replicates); seeds 3000 and 4000 (teammate runs) vary `start_seed` with the seed (data + optimization replicates). The combined 4-seed mean ± std mixes both replicate types; see the report caveat.

For each seed and method:

1. Train for `300000` timesteps from the same base checkpoint.
2. Select checkpoint on dev seeds only: `start_seed=4500`, `num_scenarios=100`, `eval.n_episodes=100`.
3. Evaluate selected checkpoint once on final heldout: `start_seed=5000`, `num_scenarios=200`, `eval.n_episodes=100`.

Distribution-shift study (supplementary, local seeds 42/2000 only): the selected checkpoints are also evaluated on easy (`start_seed=8000`, `traffic_density=0.05`) and dense (`start_seed=7000`, `traffic_density=0.2`) traffic.

## Subfolders

```text
reports/
  axis5_report.md
  axis5_summary.csv
  axis5_per_seed.csv

configs/templates/
  active config templates, one per scorer variant

configs/resolved_train/
  resolved train configs for every variant/seed run

configs/resolved_eval/
  resolved final heldout and shift eval configs

results/summary/
  axis5_per_seed_4seeds.csv (combined)
  axis5_partial_seed3000_4000_per_seed.csv (earlier teammate partial)

results/checkpoint_selection/
  dev checkpoint-selection CSVs and reports

results/final_eval/
  final heldout eval CSVs plus dense/easy shift eval CSVs

results/failure_analysis/
  failure summaries, failure-by-mode CSVs, and paper-number snippets
```

## Main Result

Across four training seeds, `NearFailureScorer` is marginally better than `DefaultFailureScorer` on mean safety-efficiency (−0.528 vs −0.566), success, offroad, route completion, and cost, while `DefaultFailureScorer` is lower on collision. The gap is within one standard deviation and not robust by paired comparison (2 wins / 2 losses on safety-efficiency). The distribution-shift study (seeds 42/2000) shows both scorers degrade on easy traffic and the near-failure advantage does not carry over under shift. Scorer choice is best framed as a second-order knob.

Use `reports/axis5_report.md` for the full interpretation.
