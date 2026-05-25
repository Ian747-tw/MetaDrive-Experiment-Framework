# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_naive_stable_seed4000/checkpoints/latest_100000_steps.zip | False | False |  | 0.3800 | 0.6200 | 0.6264 | 0.2400 | 0.5000 | 289.1600 | -0.6700 |
| runs/research_v1/multiseed_axis1_naive_stable_seed4000/checkpoints/latest_200000_steps.zip | True | False |  | 0.4800 | 0.5200 | 0.6973 | 0.2700 | 0.4400 | 242.7500 | -0.4900 |
| runs/research_v1/multiseed_axis1_naive_stable_seed4000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4000 | 0.6100 | 0.6402 | 0.1900 | 0.5500 | 266.6600 | -0.6450 |
| runs/research_v1/multiseed_axis1_naive_stable_seed4000/checkpoints/final.zip | False | False |  | 0.4900 | 0.5100 | 0.7144 | 0.2200 | 0.5300 | 230.5000 | -0.5150 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_naive_stable_seed4000/checkpoints/latest_200000_steps.zip`
