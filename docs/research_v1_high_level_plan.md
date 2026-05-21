# Research V1 High-Level Plan

## Central Hypothesis

Failure-aware scenario replay plus adaptive safety penalties improves safety-specialized fine-tuning over normal PPO fine-tuning, while preserving enough general driving performance.

## Five Axes

Axis 1 proves the main effect: naive fine-tuning vs fixed-budget fine-tuning vs FASB-PPO.

Axis 2 explains the sampler and replay contribution by varying failure replay ratio and priority.

Axis 3 explains the adaptive safety contribution by varying fixed/adaptive budgets and penalty schedules.

Axis 4 studies safety cost definition: crash-only, default driving cost, and near-miss-heavy cost.

Axis 5 studies failure definition and generalization under scorer/classifier and traffic-distribution changes.

## Integration

Axis 1 is the main comparison table. Axes 2-5 explain why the main method works, where it fails, and which design choices matter. All axes use the same base checkpoint, canonical large failure buffer, final timesteps, train seeds, eval seeds, horizon, and traffic density unless the axis explicitly studies one of those variables.

## Screening Vs Final Phase

Screening runs may use 100k timesteps to discard weak variants. Final runs use 300k timesteps for every compared method in the same table. Never compare a 300k FASB run to a 100k naive run.

## Compute Allocation

Spend compute first on the shared base checkpoint and large canonical failure buffer. Then run Axis 1 final comparisons. Use remaining compute for the most informative Axis 2-5 variants, with screening before finalizing expensive variants.

## Final Paper Table Plan

Report a main table with success, route completion, collision, offroad, timeout, episode cost, cost violation, and safety-efficiency score. Add a failure-mode table from `failure_by_mode.csv`. Include ablation tables for sampler, budget/penalty, cost, and failure scorer/generalization.

## Limitations

This framework is SB3 PPO plus failure-aware sampler and adaptive safety penalty. It is not full PPO-Lagrangian. Results depend on MetaDrive scenario seeds, chosen failure definitions, and the quality/diversity of the generated failure buffer.

## Do Not Overclaim

Do not claim full constrained-RL guarantees, PPO-Lagrangian, real-world driving safety, or broad generalization beyond the evaluated seed/traffic distributions. Claim only what the locked comparisons support.

## Shared Artifact Distribution

Required release:

```text
research-v1-foundation-v1
```

Assets:

```text
research_v1_foundation_artifacts.tar.gz
research_v1_artifact_manifest.json
```

Extracted paths:

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
runs/research_v1/eval_base_pretrain/eval/heldout_random.csv
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

Verification command:

```bash
make validate-research-v1-artifacts
```

Axis 1-4 must use the release-provided checkpoint and canonical large buffer. Axis 5 may use axis-specific buffers only when the research question explicitly studies buffer or distribution shift.
