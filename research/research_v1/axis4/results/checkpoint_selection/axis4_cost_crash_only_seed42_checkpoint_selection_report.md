# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_crash_only_final_s42/checkpoints/latest_100000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6598 | 0.2800 | 0.4400 | 262.5200 | -0.5750 |
| runs/research_v1/axis4_cost_crash_only_final_s42/checkpoints/latest_200000_steps.zip | True | False |  | 0.5300 | 0.4800 | 0.6932 | 0.2500 | 0.4300 | 229.2900 | -0.3900 |
| runs/research_v1/axis4_cost_crash_only_final_s42/checkpoints/latest_300000_steps.zip | False | False |  | 0.3900 | 0.6100 | 0.6282 | 0.1800 | 0.4600 | 260.8900 | -0.5550 |
| runs/research_v1/axis4_cost_crash_only_final_s42/checkpoints/final.zip | False | False |  | 0.3600 | 0.6400 | 0.5852 | 0.1500 | 0.5400 | 281.2300 | -0.6500 |

## Selected checkpoint

`runs/research_v1/axis4_cost_crash_only_final_s42/checkpoints/latest_200000_steps.zip`
