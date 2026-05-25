# Checkpoint Selection Report

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis4_cost_nearmiss_heavy_final_s2000/checkpoints/latest_100000_steps.zip | False | False |  | 0.4200 | 0.5900 | 0.6489 | 0.2600 | 0.4900 | 263.9200 | -0.6250 |
| runs/research_v1/axis4_cost_nearmiss_heavy_final_s2000/checkpoints/latest_200000_steps.zip | False | False |  | 0.4800 | 0.5200 | 0.6507 | 0.2800 | 0.5800 | 295.1500 | -0.6400 |
| runs/research_v1/axis4_cost_nearmiss_heavy_final_s2000/checkpoints/latest_300000_steps.zip | True | False |  | 0.5800 | 0.4300 | 0.7418 | 0.3100 | 0.4400 | 266.0200 | -0.3850 |
| runs/research_v1/axis4_cost_nearmiss_heavy_final_s2000/checkpoints/final.zip | False | False |  | 0.4600 | 0.5400 | 0.7088 | 0.3100 | 0.4700 | 287.8100 | -0.5900 |

## Selected checkpoint

`runs/research_v1/axis4_cost_nearmiss_heavy_final_s2000/checkpoints/latest_300000_steps.zip`
