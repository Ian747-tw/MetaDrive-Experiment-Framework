# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_fasb_stable_seed8000/checkpoints/latest_100000_steps.zip | False | False |  | 0.3800 | 0.6200 | 0.6348 | 0.3500 | 0.6000 | 293.5200 | -0.8800 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed8000/checkpoints/latest_200000_steps.zip | False | False |  | 0.3500 | 0.6500 | 0.6337 | 0.2200 | 0.5700 | 276.8200 | -0.7650 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed8000/checkpoints/latest_300000_steps.zip | True | False |  | 0.5500 | 0.4500 | 0.7328 | 0.2300 | 0.3800 | 227.0700 | -0.2850 |
| runs/research_v1/multiseed_axis1_fasb_stable_seed8000/checkpoints/final.zip | False | False |  | 0.5100 | 0.4900 | 0.7236 | 0.2400 | 0.4500 | 232.2400 | -0.4250 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_fasb_stable_seed8000/checkpoints/latest_300000_steps.zip`
