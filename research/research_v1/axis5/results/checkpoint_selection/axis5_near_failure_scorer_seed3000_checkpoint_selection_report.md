# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_near_failure_scorer_seed3000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4600 | 0.5400 | 0.7205 | 0.2600 | 0.4500 | 241.1600 | -0.5200 |
| runs/research_v1/axis5_near_failure_scorer_seed3000/checkpoints/latest_200000_steps.zip | True | False |  | 0.3800 | 0.6200 | 0.6253 | 0.1500 | 0.4900 | 275.6500 | -0.5700 |
| runs/research_v1/axis5_near_failure_scorer_seed3000/checkpoints/latest_300000_steps.zip | False | False |  | 0.5600 | 0.4400 | 0.7017 | 0.2800 | 0.5200 | 256.2200 | -0.4600 |
| runs/research_v1/axis5_near_failure_scorer_seed3000/checkpoints/final.zip | False | False |  | 0.4600 | 0.5400 | 0.6870 | 0.1700 | 0.4900 | 244.9700 | -0.4700 |

## Selected checkpoint

`runs/research_v1/axis5_near_failure_scorer_seed3000/checkpoints/latest_200000_steps.zip`
