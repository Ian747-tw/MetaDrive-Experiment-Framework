# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_naive_stable_seed8000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4600 | 0.5400 | 0.6774 | 0.3300 | 0.4300 | 251.3200 | -0.5700 |
| runs/research_v1/multiseed_axis1_naive_stable_seed8000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4500 | 0.5500 | 0.6730 | 0.2200 | 0.4300 | 235.2800 | -0.4750 |
| runs/research_v1/multiseed_axis1_naive_stable_seed8000/checkpoints/latest_300000_steps.zip | True | False |  | 0.5400 | 0.4600 | 0.7228 | 0.2400 | 0.4500 | 250.5500 | -0.3800 |
| runs/research_v1/multiseed_axis1_naive_stable_seed8000/checkpoints/final.zip | False | False |  | 0.4500 | 0.5500 | 0.6712 | 0.2300 | 0.4800 | 273.2500 | -0.5350 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_naive_stable_seed8000/checkpoints/latest_300000_steps.zip`
