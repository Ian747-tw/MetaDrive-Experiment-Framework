# Axis3 Notes

- This folder is a curated view of Axis 3 experiments under runs/research_v1/.
- Large artifacts (checkpoints/*.zip, full logs) stay in the original run directories; this axis3/ tree stores configs + CSV summaries.

## Source Runs
- Training runs: runs/research_v1/axis3_{fixed|adaptive}_*_repXX
- Eval runs:     runs/research_v1/eval_axis3_{fixed|adaptive}_*_repXX

## Where To Look
- Final eval CSVs: results/final_eval/
- Failure analysis: results/failure_analysis/
- Resolved configs: configs/resolved_train/ and configs/resolved_eval/
- Suite summary (10 reps): reports/axis3_suite_results_s42_10reps.csv

