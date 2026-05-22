# Research V1 Final Runbook

## Step 1 - Activate Environment

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
```

## Step 2 - Verify Base Checkpoint

```bash
CUDA_VISIBLE_DEVICES= python scripts/evaluate.py \
  --config configs/research_v1/base_eval.yaml \
  --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip

python scripts/check_base_checkpoint_quality.py \
  --eval-csv runs/research_v1/eval_base_pretrain/eval/heldout_random.csv \
  --min-episodes 100 \
  --min-success-rate 0.10 \
  --min-route-completion 0.35 \
  --max-timeout-rate 0.95
```

## Step 3 - Improve Base Checkpoint If Needed

If quality fails or you need a stronger base:

```bash
cp runs/research_v1/base_pretrain_s42/checkpoints/final.zip \
  runs/research_v1/base_pretrain_s42/checkpoints/final_before_reinforce.zip

CUDA_VISIBLE_DEVICES= python scripts/train.py \
  --config configs/research_v1/base_pretrain.yaml \
  algorithm.checkpoint_path=runs/research_v1/base_pretrain_s42/checkpoints/final.zip \
  training.total_timesteps=500000 \
  algorithm.params.device=cpu
```

Then re-evaluate and rerun the quality checker. Do not call the checkpoint improved unless success and route completion are equal or better and timeout does not materially regress.

## Step 4 - Build Large Canonical Failure Buffer

```bash
CUDA_VISIBLE_DEVICES= python scripts/explore_failures.py \
  --config configs/research_v1/base_explore_large.yaml
```

The config target is 3000 exploration episodes. Use a CLI override only for screening.

## Step 5 - Validate Large Buffer

```bash
python scripts/check_failure_buffer_quality.py \
  --buffer runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl \
  --min-records 1000 \
  --max-unknown-fraction 0.25 \
  --require-multiple-modes
```

Preferred target: 2000-3000 records if exploration is cheap enough.

## Step 6 - Final Readiness

```bash
python scripts/check_research_v1_ready.py \
  --root runs/research_v1 \
  --checkpoint runs/research_v1/base_pretrain_s42/checkpoints/final.zip \
  --eval-csv runs/research_v1/eval_base_pretrain/eval/heldout_random.csv \
  --buffer runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl \
  --min-failures 1000 \
  --min-episodes 100 \
  --min-success-rate 0.10 \
  --min-route-completion 0.35 \
  --max-timeout-rate 0.95
```

## Step 7 - Run Axes

Every final-axis run must use:

```text
failure_buffer.path=runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
training.total_timesteps=300000
```

Keep base checkpoint, final buffer, train/eval seeds, horizon, traffic density, and timesteps locked unless the axis explicitly studies that variable.

Axis 1 includes both the original FASB-PPO config and an attempted FASB-PPO v2 candidate config:

```bash
CUDA_VISIBLE_DEVICES= python scripts/train.py \
  --config configs/research_v1/axis1_fasb_final.yaml

CUDA_VISIBLE_DEVICES= python scripts/train.py \
  --config configs/research_v1/axis1_fasb_v2_final.yaml
```

The original FASB default collapsed into timeout in the first Axis 1 run: it achieved zero collision/offroad/cost by nearly freezing. Keep that result visible as a historical reference. The attempted FASB v2 candidate used gentler failure replay and safety penalty settings selected on the dev validation range (`start_seed=4500`, `num_scenarios=100`), not on the final heldout range. However, the 300k v2 checkpoint also collapsed on dev and final heldout, so it is not an accepted default. Axis 2 and Axis 3 should explicitly test whether candidate settings remain valid at the intended final training duration before using final heldout.

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

Download and verify:

```bash
python scripts/download_research_v1_artifacts.py --release research-v1-foundation-v1
make validate-research-v1-artifacts
```

Axis 1-4 must use the release-provided checkpoint and canonical large buffer. Axis 5 may use axis-specific buffers only when the research question explicitly studies buffer or distribution shift.
