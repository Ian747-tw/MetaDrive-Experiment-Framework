# Research V1 Results

## Evaluate One Method

```bash
python scripts/evaluate.py \
  --config configs/eval/heldout_random.yaml \
  --checkpoint runs/research_v1/<method>/checkpoints/final.zip \
  experiment.name=eval_<method> \
  experiment.output_dir=runs/research_v1/eval_<method> \
  eval.n_episodes=100 \
  metadrive.config.start_seed=5000 \
  metadrive.config.num_scenarios=200 \
  metadrive.config.horizon=500
```

## Analyze

```bash
python scripts/analyze_failures.py --run runs/research_v1/eval_<method>
```

## Output Files

```text
runs/research_v1/eval_<method>/eval/heldout_random.csv
runs/research_v1/eval_<method>/analysis/failure_summary.csv
runs/research_v1/eval_<method>/analysis/failure_by_mode.csv
runs/research_v1/eval_<method>/analysis/paper_numbers.md
```

## Aggregate

```bash
python scripts/aggregate_results.py --root runs/research_v1
```

Outputs:

```text
runs/research_v1/summary_main_results.csv
runs/research_v1/summary_failure_by_mode.csv
```

## Table Interpretation

FASB beats normal fine-tuning only if it uses the same compute, base checkpoint, train seed range, eval seed range, horizon, traffic density, and evaluation protocol. Do not cherry-pick a safety metric if success or route completion collapses. Report `safety_efficiency_score`, success/completion, raw cost, collision/offroad/timeout rates, and failure-mode breakdown together.

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
