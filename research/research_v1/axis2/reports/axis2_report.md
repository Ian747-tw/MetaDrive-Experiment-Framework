# Axis 2 Research Report: Failure Sampler Ratio

## Research Question

Axis 2 asks whether increasing the failure-buffer replay ratio improves safety-specialized fine-tuning, or whether too much replay causes overfitting, conservatism, and lower route progress.

## Locked Protocol

| item | value |
| --- | --- |
| Base checkpoint | GitHub release artifact; local after download: `runs/research_v1/base_pretrain_s42/checkpoints/final.zip` |
| Failure buffer | GitHub release artifact; local after download: `runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl` |
| Training steps | `300000` |
| Optimizer | SB3 PPO, `learning_rate=0.00003` |
| Cost | `DefaultDrivingCost` |
| Safety budget | `d_min=0.10`, `d_max=0.30`, `timeout_budget=0.30` |
| Penalty | `lambda_min=0.0`, `lambda_max=0.25` |
| Final eval | heldout random, `start_seed=5000`, 100 episodes |
| Axis variable | `sampler.failure_ratio` |

## Result Chart

![Axis 2 sampler ablation](axis2_sampler_ablation.svg)

## Final Eval Results

Source: `results/final_eval/*_heldout_random.csv` and `results/summary/multiseed_axis2_summary.csv`.

| variant | failure ratio | success | collision | offroad | timeout | route | cost | safety-eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed005 | 0.05 | 0.48 | 0.28 | 0.49 | 0.52 | 0.7067 | 260.50 | -0.550 |
| mixed030 | 0.30 | 0.47 | 0.28 | 0.44 | 0.53 | 0.6978 | 239.82 | -0.515 |
| mixed060 | 0.60 | 0.35 | 0.18 | 0.59 | 0.65 | 0.6298 | 277.59 | -0.745 |
| mixed090 | 0.90 | 0.42 | 0.25 | 0.48 | 0.58 | 0.7004 | 253.22 | -0.600 |

## Analysis

`mixed030` is the best available Axis 2 result by safety-efficiency and cost. The stable default `mixed005` has the highest success and route completion, but it has higher offroad and higher cost than `mixed030`. High replay is clearly unsafe as a final default: `mixed060` has the lowest success, highest timeout, highest offroad, lowest route completion, and worst safety-efficiency.

The sampler result supports a bounded replay design. Failure-buffer examples can help, but replay should not dominate the training distribution. This also explains why the stable default uses `failure_ratio=0.05`: it is conservative and less likely to overfit. The `0.30` result is promising but should be repeated over more seeds before replacing the default.

## Recommendation

Keep `failure_ratio=0.05` as the conservative default until `mixed030` is repeated over more seeds. For future work, test `0.10`, `0.20`, and `0.30` with the fixed005 budget from Axis 3.
