# Research V1 Reviewer Package

This folder is the reviewer-facing research package for the MetaDrive FASB-PPO project. It is organized so paper reviewers and teammates can inspect the paper draft, reports, configs, and lightweight result artifacts without digging through generated training directories.

No model checkpoints, failure buffers, or large generated artifacts are stored here. Reproduction uses the canonical artifact release and the configs in this repository.

## Layout

```text
research/research_v1/
  research_v1_paper.md
  axis1/
    reports/
    configs/
      templates/
      resolved_train/
      resolved_eval/
    results/
      summary/
      checkpoint_selection/
      final_eval/
      failure_analysis/
    others/
  axis2/
  axis3/
  axis4/
  axis5/
```

## Axis Status

| axis | status | contents |
| --- | --- | --- |
| Axis 1 | filled | Main multiseed stable FASB vs naive PPO result. |
| Axis 2 | skeleton | Sampler ablations to be filled by teammates. |
| Axis 3 | skeleton | Budget/penalty ablations to be filled by teammates. |
| Axis 4 | skeleton | Cost-function ablations to be filled by teammates. |
| Axis 5 | skeleton | Failure scorer/generalization ablations to be filled by teammates. |

## Axis 1 Authority

Axis 1 uses the newest 6/12 multiseed research runs, not the older single-run result. The authoritative training seeds are:

```text
2000, 3000, 4000, 6000, 7000, 8000
```

Seed `5000` is excluded because the final heldout range starts at `5000`.

Authoritative summary files:

```text
axis1/reports/axis1_report.md
axis1/reports/axis1_summary.csv
axis1/results/summary/multiseed_axis1_per_seed.csv
axis1/results/summary/multiseed_axis1_summary.csv
axis1/results/summary/multiseed_axis1_paired_deltas.csv
```

The older single-run stable FASB result is retained only as a diagnostic in the report. It is not the main Axis 1 claim.

## Reproduction Requirements

From the repository root:

```bash
cd ~/metadrive
source .venv/bin/activate
cd ~/projects/MetaDrive-Experiment-Framework
pip install -e . --no-deps
python scripts/check_env.py --require-metadrive
make validate-research-v1-artifacts
```

Required canonical artifacts:

```text
runs/research_v1/base_pretrain_s42/checkpoints/final.zip
runs/research_v1/base_explore_large/buffers/failure_buffer.jsonl
```

If the artifacts are missing, run:

```bash
python scripts/download_research_v1_artifacts.py --release research-v1-foundation-v1
make validate-research-v1-artifacts
```

## Notes For Teammates

Each future axis should follow the same folder convention:

```text
axisN/
  reports/   final markdown report and compact CSV summary
  configs/   active configs and resolved configs
  results/   result CSVs, eval CSVs, checkpoint-selection CSVs, failure analysis
  others/    commands, notes, diagnostics, or reviewer instructions
```

Do not place checkpoints, model zip files, failure buffers, or large generated run directories in this research package.
