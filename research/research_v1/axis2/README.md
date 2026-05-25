# Axis 2 Package: Sampler Ablation

Axis 2 studies the failure-aware sampler by varying the failure replay ratio while keeping the stable FASB optimizer, base checkpoint, large failure buffer, cost, safety budget, and penalty schedule fixed.

## Current Status

Axis 2 currently contains final-heldout point estimates over replay ratios:

```text
seed: 2000
variants: mixed005, mixed030, mixed060, mixed090
```

The result is useful as an ablation of replay-ratio behavior, but the conclusion should remain bounded to the available runs.

## Contents

```text
configs/templates/
configs/resolved_train/
configs/resolved_eval/
results/final_eval/
results/summary/
reports/
```

## Main Result

| variant | failure ratio | success | collision | offroad | timeout | route | cost | safety-eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed005 | 0.05 | 0.48 | 0.28 | 0.49 | 0.52 | 0.7067 | 260.50 | -0.550 |
| mixed030 | 0.30 | 0.47 | 0.28 | 0.44 | 0.53 | 0.6978 | 239.82 | -0.515 |
| mixed060 | 0.60 | 0.35 | 0.18 | 0.59 | 0.65 | 0.6298 | 277.59 | -0.745 |
| mixed090 | 0.90 | 0.42 | 0.25 | 0.48 | 0.58 | 0.7004 | 253.22 | -0.600 |

Interpretation: moderate replay (`0.30`) is best in the available Axis 2 runs by safety-efficiency and cost. High replay (`0.60`) hurts progress and increases timeout/offroad, showing that failure replay can become counterproductive if it dominates training.
