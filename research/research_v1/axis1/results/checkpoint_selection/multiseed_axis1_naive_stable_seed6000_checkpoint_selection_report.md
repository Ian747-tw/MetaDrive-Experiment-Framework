# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_naive_stable_seed6000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6384 | 0.1900 | 0.4400 | 248.1900 | -0.4850 |
| runs/research_v1/multiseed_axis1_naive_stable_seed6000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4600 | 0.5400 | 0.6803 | 0.2600 | 0.4200 | 253.0300 | -0.4900 |
| runs/research_v1/multiseed_axis1_naive_stable_seed6000/checkpoints/latest_300000_steps.zip | False | False |  | 0.5200 | 0.4800 | 0.7253 | 0.2800 | 0.5200 | 245.3900 | -0.5200 |
| runs/research_v1/multiseed_axis1_naive_stable_seed6000/checkpoints/final.zip | True | False |  | 0.5700 | 0.4300 | 0.7320 | 0.2700 | 0.5000 | 244.3500 | -0.4150 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_naive_stable_seed6000/checkpoints/final.zip`
