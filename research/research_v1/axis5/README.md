# Axis 5 Package: Failure Scorer Partial Results

Allowed research variables:

```text
failure_scorer.*
failure_classifier.*
explicitly labeled eval distribution changes
explicitly labeled buffer/generalization changes
```

Keep the stable optimizer, sampler, budget, penalty, and total training budget fixed unless the report explicitly labels the variant as a distribution-shift or buffer experiment.

## Current Status

This folder currently contains partial raw results for teammate coordination, not a final Axis 5 report.

Completed seeds:

```text
3000, 4000
```

Completed variants:

```text
axis5_default_scorer_final.yaml
axis5_near_failure_scorer_final.yaml
```

Do not compute final mean +/- std or write the final Axis 5 conclusion from only these two seeds if more teammate seeds are still pending.

## Contents

```text
reports/
configs/templates/
configs/resolved_train/
configs/resolved_eval/
results/checkpoint_selection/
results/final_eval/
results/failure_analysis/
results/summary/axis5_partial_seed3000_4000_per_seed.csv
others/
```

The partial per-seed summary is only for checking and handoff:

```text
results/summary/axis5_partial_seed3000_4000_per_seed.csv
```
