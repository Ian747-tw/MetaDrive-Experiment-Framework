# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_default_scorer_final_s42/checkpoints/latest_100000_steps.zip | False | False |  | 0.4100 | 0.5900 | 0.6402 | 0.2500 | 0.5000 | 266.0300 | -0.6350 |
| runs/research_v1/axis5_default_scorer_final_s42/checkpoints/latest_200000_steps.zip | True | False |  | 0.4300 | 0.5700 | 0.6604 | 0.1700 | 0.5000 | 257.6900 | -0.5250 |
| runs/research_v1/axis5_default_scorer_final_s42/checkpoints/latest_300000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6121 | 0.2100 | 0.4600 | 261.4000 | -0.5250 |
| runs/research_v1/axis5_default_scorer_final_s42/checkpoints/final.zip | False | False |  | 0.4000 | 0.6000 | 0.6533 | 0.2300 | 0.5400 | 262.9300 | -0.6700 |

## Selected checkpoint

`runs/research_v1/axis5_default_scorer_final_s42/checkpoints/latest_200000_steps.zip`
