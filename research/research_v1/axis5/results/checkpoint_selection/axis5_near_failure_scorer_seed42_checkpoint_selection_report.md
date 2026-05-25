# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_near_failure_scorer_final_s42/checkpoints/latest_100000_steps.zip | True | False |  | 0.5100 | 0.4900 | 0.7096 | 0.3000 | 0.3100 | 238.4100 | -0.3450 |
| runs/research_v1/axis5_near_failure_scorer_final_s42/checkpoints/latest_200000_steps.zip | False | False |  | 0.4700 | 0.5300 | 0.6795 | 0.2300 | 0.5400 | 278.6000 | -0.5650 |
| runs/research_v1/axis5_near_failure_scorer_final_s42/checkpoints/latest_300000_steps.zip | False | False |  | 0.4700 | 0.5300 | 0.6900 | 0.2500 | 0.5100 | 270.0900 | -0.5550 |
| runs/research_v1/axis5_near_failure_scorer_final_s42/checkpoints/final.zip | False | False |  | 0.4500 | 0.5500 | 0.6356 | 0.1800 | 0.4900 | 266.9100 | -0.4950 |

## Selected checkpoint

`runs/research_v1/axis5_near_failure_scorer_final_s42/checkpoints/latest_100000_steps.zip`
