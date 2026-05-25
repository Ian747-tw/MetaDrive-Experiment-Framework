# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_default_final_s3000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6756 | 0.2700 | 0.4400 | 230.7900 | -0.5650 |
| runs/research_v1/axis4_cost_default_final_s3000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4800 | 0.5200 | 0.6992 | 0.3700 | 0.5300 | 274.3600 | -0.6800 |
| runs/research_v1/axis4_cost_default_final_s3000/checkpoints/latest_300000_steps.zip | True | False |  | 0.5300 | 0.4700 | 0.7266 | 0.2900 | 0.4900 | 294.8500 | -0.4850 |
| runs/research_v1/axis4_cost_default_final_s3000/checkpoints/final.zip | False | False |  | 0.4600 | 0.5400 | 0.7041 | 0.3000 | 0.5700 | 292.3000 | -0.6800 |

## Selected checkpoint

`runs/research_v1/axis4_cost_default_final_s3000/checkpoints/latest_300000_steps.zip`
