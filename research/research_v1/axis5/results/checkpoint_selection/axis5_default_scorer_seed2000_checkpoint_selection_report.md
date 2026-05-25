# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_default_scorer_final_s2000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4100 | 0.5900 | 0.6125 | 0.2300 | 0.5300 | 268.9300 | -0.6450 |
| runs/research_v1/axis5_default_scorer_final_s2000/checkpoints/latest_200000_steps.zip | True | False |  | 0.5000 | 0.5000 | 0.7017 | 0.2400 | 0.4300 | 265.9300 | -0.4200 |
| runs/research_v1/axis5_default_scorer_final_s2000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4600 | 0.5400 | 0.7103 | 0.2200 | 0.4900 | 253.2900 | -0.5200 |
| runs/research_v1/axis5_default_scorer_final_s2000/checkpoints/final.zip | False | False |  | 0.4500 | 0.5600 | 0.6525 | 0.2700 | 0.5800 | 281.7400 | -0.6800 |

## Selected checkpoint

`runs/research_v1/axis5_default_scorer_final_s2000/checkpoints/latest_200000_steps.zip`
