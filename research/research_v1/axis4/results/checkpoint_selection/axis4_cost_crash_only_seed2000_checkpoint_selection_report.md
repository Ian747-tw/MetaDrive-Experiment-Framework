# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_crash_only_final_s2000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6537 | 0.3100 | 0.4900 | 278.2800 | -0.6550 |
| runs/research_v1/axis4_cost_crash_only_final_s2000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4500 | 0.5500 | 0.7111 | 0.1800 | 0.5100 | 244.7600 | -0.5150 |
| runs/research_v1/axis4_cost_crash_only_final_s2000/checkpoints/latest_300000_steps.zip | True | False |  | 0.5000 | 0.5000 | 0.6962 | 0.1700 | 0.4100 | 237.9800 | -0.3300 |
| runs/research_v1/axis4_cost_crash_only_final_s2000/checkpoints/final.zip | False | False |  | 0.4500 | 0.5500 | 0.6652 | 0.1900 | 0.4400 | 247.3100 | -0.4550 |

## Selected checkpoint

`runs/research_v1/axis4_cost_crash_only_final_s2000/checkpoints/latest_300000_steps.zip`
