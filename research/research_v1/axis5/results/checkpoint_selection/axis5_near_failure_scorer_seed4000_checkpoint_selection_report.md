# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis5_near_failure_scorer_seed4000/checkpoints/latest_100000_steps.zip | True | False |  | 0.5500 | 0.4500 | 0.7155 | 0.2900 | 0.3200 | 236.3100 | -0.2850 |
| runs/research_v1/axis5_near_failure_scorer_seed4000/checkpoints/latest_200000_steps.zip | False | False |  | 0.5100 | 0.4900 | 0.6877 | 0.1900 | 0.4000 | 246.9000 | -0.3250 |
| runs/research_v1/axis5_near_failure_scorer_seed4000/checkpoints/latest_300000_steps.zip | False | False |  | 0.5200 | 0.4800 | 0.7016 | 0.1900 | 0.4400 | 228.6600 | -0.3500 |
| runs/research_v1/axis5_near_failure_scorer_seed4000/checkpoints/final.zip | False | False |  | 0.4600 | 0.5400 | 0.6653 | 0.2100 | 0.4100 | 244.8100 | -0.4300 |

## Selected checkpoint

`runs/research_v1/axis5_near_failure_scorer_seed4000/checkpoints/latest_100000_steps.zip`
