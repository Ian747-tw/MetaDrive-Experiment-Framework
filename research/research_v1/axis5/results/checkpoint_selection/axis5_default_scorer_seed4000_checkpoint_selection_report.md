# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_default_scorer_seed4000/checkpoints/latest_100000_steps.zip | False | True | success_rate < 0.20; timeout_rate > 0.80 | 0.1500 | 0.8500 | 0.5382 | 0.1800 | 0.8200 | 393.8600 | -1.2750 |
| runs/research_v1/axis5_default_scorer_seed4000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6984 | 0.2700 | 0.4600 | 250.7000 | -0.5850 |
| runs/research_v1/axis5_default_scorer_seed4000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4200 | 0.5800 | 0.7255 | 0.3200 | 0.3700 | 273.7700 | -0.5600 |
| runs/research_v1/axis5_default_scorer_seed4000/checkpoints/final.zip | True | False |  | 0.4600 | 0.5400 | 0.7034 | 0.2100 | 0.4500 | 270.9100 | -0.4700 |

## Selected checkpoint

`runs/research_v1/axis5_default_scorer_seed4000/checkpoints/final.zip`
