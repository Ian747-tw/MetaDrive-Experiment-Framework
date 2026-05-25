# Checkpoint Selection Report (axis3_fixed005_s42_rep08)

Selection metric: `safety_efficiency_score`

| checkpoint | selected | hard_reject | reject_reasons | success_rate | timeout_rate | route_completion_mean | collision_rate | offroad_rate | episode_cost_mean | safety_efficiency_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/research_v1/axis3_fixed005_s42_rep08/checkpoints/latest_100000_steps.zip | False | True | success_rate < 0.20; timeout_rate > 0.80 | 0.0600 | 0.9200 | 0.6125 | 0.1500 | 0.8500 | 380.4500 | -1.3950 |
| runs/research_v1/axis3_fixed005_s42_rep08/checkpoints/latest_200000_steps.zip | False | False | | 0.4800 | 0.5000 | 0.7254 | 0.2500 | 0.3500 | 235.1200 | -0.3850 |
| runs/research_v1/axis3_fixed005_s42_rep08/checkpoints/latest_300000_steps.zip | True | False | | 0.6100 | 0.3900 | 0.7629 | 0.2800 | 0.2600 | 216.7200 | -0.1250 |
| runs/research_v1/axis3_fixed005_s42_rep08/checkpoints/final.zip | False | False | | 0.5800 | 0.4100 | 0.7451 | 0.2900 | 0.2800 | 222.1500 | -0.1850 |

## Selected checkpoint

`runs/research_v1/axis3_fixed005_s42_rep08/checkpoints/latest_300000_steps.zip`
