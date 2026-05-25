# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_event_driving_final_s3000/checkpoints/latest_100000_steps.zip | True | False |  | 0.4600 | 0.5400 | 0.6851 | 0.1700 | 0.4100 | 232.7800 | -0.3900 |
| runs/research_v1/axis4_cost_event_driving_final_s3000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4700 | 0.5300 | 0.6655 | 0.2200 | 0.4600 | 230.8800 | -0.4750 |
| runs/research_v1/axis4_cost_event_driving_final_s3000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4500 | 0.5500 | 0.7171 | 0.2000 | 0.4900 | 243.9300 | -0.5150 |
| runs/research_v1/axis4_cost_event_driving_final_s3000/checkpoints/final.zip | False | False |  | 0.4600 | 0.5400 | 0.6917 | 0.2000 | 0.4500 | 230.8400 | -0.4600 |

## Selected checkpoint

`runs/research_v1/axis4_cost_event_driving_final_s3000/checkpoints/latest_100000_steps.zip`
