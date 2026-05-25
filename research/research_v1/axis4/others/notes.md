# Axis4 Notes

- This folder is a curated view of Axis 4 experiments under runs/research_v1/.
- Large artifacts (checkpoints/*.zip, full logs) stay in the original run directories; this axis4/ tree stores configs + CSV summaries + reports.

## Source Runs
- Training runs: runs/research_v1/axis4_cost_{default|crash_only|nearmiss_heavy|event_driving}_final_s{42|2000|3000}
- Checkpoint selection: runs/research_v1/stabilization/select_axis4_cost_*_final_s{seed}
- Eval runs: runs/research_v1/eval_axis4_cost_*_final_s{seed}

## Collector Script
`collect_axis45.py` copies the run artifacts into this package layout and recomputes the per-seed and mean ± std summary CSVs. It handles both Axis 4 and Axis 5. Run from the repo root:

```bash
python research/research_v1/axis4/others/collect_axis45.py
```

It is idempotent: re-running overwrites the copied configs/results and regenerates `reports/axis4_summary.csv`, `reports/axis4_per_seed.csv`, and `results/summary/axis4_per_seed.csv`. Add new seeds by adding them to `ax4_seeds` in the script.

## Where To Look
- Final eval CSVs: results/final_eval/
- Failure analysis: results/failure_analysis/
- Resolved configs: configs/resolved_train/ and configs/resolved_eval/
- Per-variant summary (mean ± std): reports/axis4_summary.csv
- Full interpretation: reports/axis4_report.md
