# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_default_scorer_seed3000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4000 | 0.6000 | 0.6672 | 0.2700 | 0.4500 | 253.3100 | -0.6200 |
| runs/research_v1/axis5_default_scorer_seed3000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4400 | 0.5600 | 0.6310 | 0.2300 | 0.4900 | 262.3300 | -0.5600 |
| runs/research_v1/axis5_default_scorer_seed3000/checkpoints/latest_300000_steps.zip | True | False |  | 0.4900 | 0.5100 | 0.6931 | 0.2800 | 0.4600 | 239.9000 | -0.5050 |
| runs/research_v1/axis5_default_scorer_seed3000/checkpoints/final.zip | False | False |  | 0.4100 | 0.5900 | 0.6942 | 0.1800 | 0.4600 | 233.3300 | -0.5250 |

## Selected checkpoint

`runs/research_v1/axis5_default_scorer_seed3000/checkpoints/latest_300000_steps.zip`
