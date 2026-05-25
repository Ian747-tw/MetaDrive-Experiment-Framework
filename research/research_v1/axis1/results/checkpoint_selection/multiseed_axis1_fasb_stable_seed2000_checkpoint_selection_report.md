# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_fasb_stable_seed2000/checkpoints/latest_100000_steps.zip | False | True | success_rate < 0.20; timeout_rate > 0.80 | 0.0500 | 0.9500 | 0.6532 | 0.1600 | 0.9000 | 400.5600 | -1.4850 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed2000/checkpoints/latest_200000_steps.zip | False | False |  | 0.3400 | 0.6600 | 0.6183 | 0.1200 | 0.6300 | 282.8000 | -0.7400 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed2000/checkpoints/latest_300000_steps.zip | True | False |  | 0.4400 | 0.5600 | 0.6417 | 0.2400 | 0.4100 | 250.9300 | -0.4900 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed2000/checkpoints/final.zip | False | False |  | 0.4500 | 0.5500 | 0.6708 | 0.2500 | 0.4600 | 252.9300 | -0.5350 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_fasb_stable_seed2000/checkpoints/latest_300000_steps.zip`
