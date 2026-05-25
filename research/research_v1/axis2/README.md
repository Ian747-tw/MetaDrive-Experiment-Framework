# Axis 2 Package Skeleton: Sampler Ablation

Fill this folder after Axis 2 runs.

Allowed research variable:

```text
sampler._target_
sampler.failure_ratio
sampler.alpha
```

Keep the stable protocol fixed unless the Axis 2 report explicitly justifies a screening run:

```text
learning_rate=0.00003
training.total_timesteps=300000 for final configs
base checkpoint=runs/research_v1/base_pretrain_s42/checkpoints/final.zip
failure buffer=runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
safety_budget.d_min=0.10
safety_budget.d_max=0.30
safety_budget.timeout_budget=0.30
penalty_scheduler.lambda_min=0.0
penalty_scheduler.lambda_max=0.25
```

Expected subfolders:

```text
reports/
configs/
results/
others/
```
