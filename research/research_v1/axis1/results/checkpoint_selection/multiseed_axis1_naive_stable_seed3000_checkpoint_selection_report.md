# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_naive_stable_seed3000/checkpoints/latest_100000_steps.zip | True | False |  | 0.4700 | 0.5300 | 0.7289 | 0.3400 | 0.3200 | 248.2500 | -0.4550 |
| runs/research_v1/multiseed_axis1_naive_stable_seed3000/checkpoints/latest_200000_steps.zip | False | False |  | 0.5000 | 0.5000 | 0.7363 | 0.2900 | 0.4300 | 232.5200 | -0.4700 |
| runs/research_v1/multiseed_axis1_naive_stable_seed3000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4600 | 0.5400 | 0.6486 | 0.2100 | 0.4900 | 258.3900 | -0.5100 |
| runs/research_v1/multiseed_axis1_naive_stable_seed3000/checkpoints/final.zip | False | False |  | 0.4800 | 0.5200 | 0.6851 | 0.2400 | 0.5300 | 245.6000 | -0.5500 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_naive_stable_seed3000/checkpoints/latest_100000_steps.zip`
