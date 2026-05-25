# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_near_failure_scorer_final_s2000/checkpoints/latest_100000_steps.zip | True | False |  | 0.4700 | 0.5300 | 0.7242 | 0.1000 | 0.4800 | 239.9100 | -0.3750 |
| runs/research_v1/axis5_near_failure_scorer_final_s2000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4300 | 0.5700 | 0.6793 | 0.3100 | 0.4300 | 260.2900 | -0.5950 |
| runs/research_v1/axis5_near_failure_scorer_final_s2000/checkpoints/latest_300000_steps.zip | False | False |  | 0.4600 | 0.5400 | 0.6819 | 0.2700 | 0.5300 | 277.8800 | -0.6100 |
| runs/research_v1/axis5_near_failure_scorer_final_s2000/checkpoints/final.zip | False | False |  | 0.5300 | 0.4700 | 0.6969 | 0.2700 | 0.4400 | 267.2300 | -0.4150 |

## Selected checkpoint

`runs/research_v1/axis5_near_failure_scorer_final_s2000/checkpoints/latest_100000_steps.zip`
