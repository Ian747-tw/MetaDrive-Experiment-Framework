# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_fasb_stable_seed7000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4200 | 0.5800 | 0.6495 | 0.2700 | 0.5000 | 264.4900 | -0.6400 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed7000/checkpoints/latest_200000_steps.zip | True | False |  | 0.5100 | 0.4900 | 0.7310 | 0.2700 | 0.3700 | 232.6500 | -0.3750 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed7000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.7008 | 0.2500 | 0.4600 | 260.4800 | -0.5650 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed7000/checkpoints/final.zip | False | False |  | 0.4700 | 0.5300 | 0.7099 | 0.2200 | 0.4300 | 241.3700 | -0.4450 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_fasb_stable_seed7000/checkpoints/latest_200000_steps.zip`
