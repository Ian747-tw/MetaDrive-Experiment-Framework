# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/multiseed_axis1_naive_stable_seed7000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4700 | 0.5300 | 0.7132 | 0.3200 | 0.3700 | 248.4700 | -0.4850 |
| runs/research_v1/multiseed_axis1_naive_stable_seed7000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4100 | 0.5900 | 0.6618 | 0.2700 | 0.5100 | 262.3300 | -0.6650 |
| runs/research_v1/multiseed_axis1_naive_stable_seed7000/checkpoints/latest_300000_steps.zip | True | False |  | 0.4700 | 0.5300 | 0.7274 | 0.2700 | 0.3600 | 227.4200 | -0.4250 |
| runs/research_v1/multiseed_axis1_naive_stable_seed7000/checkpoints/final.zip | False | False |  | 0.4800 | 0.5200 | 0.6643 | 0.3200 | 0.3700 | 258.2700 | -0.4700 |

## Selected checkpoint

`runs/research_v1/multiseed_axis1_naive_stable_seed7000/checkpoints/latest_300000_steps.zip`
