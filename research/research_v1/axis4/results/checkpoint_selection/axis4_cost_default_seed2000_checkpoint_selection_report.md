# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_default_final_s2000/checkpoints/latest_100000_steps.zip | True | False |  | 0.4700 | 0.5300 | 0.6997 | 0.2300 | 0.3400 | 245.9700 | -0.3650 |
| runs/research_v1/axis4_cost_default_final_s2000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4500 | 0.5500 | 0.6883 | 0.3200 | 0.4000 | 250.1100 | -0.5450 |
| runs/research_v1/axis4_cost_default_final_s2000/checkpoints/latest_300000_steps.zip | False | True | success_rate < 0.20; timeout_rate > 0.80 | 0.0200 | 0.9800 | 0.6222 | 0.1500 | 0.8700 | 407.7900 | -1.4900 |
| runs/research_v1/axis4_cost_default_final_s2000/checkpoints/final.zip | False | True | success_rate < 0.20; timeout_rate > 0.80 | 0.0200 | 0.9800 | 0.6377 | 0.1400 | 0.8900 | 409.3400 | -1.5000 |

## Selected checkpoint

`runs/research_v1/axis4_cost_default_final_s2000/checkpoints/latest_100000_steps.zip`
