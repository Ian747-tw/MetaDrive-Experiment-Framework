# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_crash_only_final_s3000/checkpoints/latest_100000_steps.zip | False | False |  | 0.5100 | 0.4900 | 0.6945 | 0.2500 | 0.3900 | 238.6700 | -0.3750 |
| runs/research_v1/axis4_cost_crash_only_final_s3000/checkpoints/latest_200000_steps.zip | True | False |  | 0.5100 | 0.4900 | 0.7064 | 0.1700 | 0.4000 | 233.2400 | -0.3050 |
| runs/research_v1/axis4_cost_crash_only_final_s3000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4800 | 0.5200 | 0.6460 | 0.2500 | 0.4300 | 253.3700 | -0.4600 |
| runs/research_v1/axis4_cost_crash_only_final_s3000/checkpoints/final.zip | False | False |  | 0.4900 | 0.5200 | 0.6871 | 0.2300 | 0.4600 | 241.8200 | -0.4600 |

## Selected checkpoint

`runs/research_v1/axis4_cost_crash_only_final_s3000/checkpoints/latest_200000_steps.zip`
