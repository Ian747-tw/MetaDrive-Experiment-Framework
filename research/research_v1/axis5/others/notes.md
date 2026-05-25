# Axis5 Notes

- This folder is a curated view of Axis 5 experiments under runs/research_v1/.
- Large artifacts (checkpoints/*.zip, full logs) stay in the original run directories; this axis5/ tree stores configs + CSV summaries + reports.

## Source Runs
- Local training runs (seeds 42, 2000): runs/research_v1/axis5_{default_scorer|near_failure_scorer}_final_s{42|2000}
- Checkpoint selection: runs/research_v1/stabilization/select_axis5_*_final_s{seed}
- Canonical eval: runs/research_v1/eval_axis5_*_final_s{seed}
- Shift evals (seeds 42, 2000): runs/research_v1/eval_axis5_*_final_{dense|easy}_s{seed}
- Teammate seeds 3000, 4000 were already present in this package (results/summary/axis5_partial_seed3000_4000_per_seed.csv) and use start_seed = seed, not the fixed start_seed=2000 used by the local seeds.

## Collector Script
`collect_axis45.py` copies the local run artifacts (seeds 42, 2000) into this package layout, then merges them with the teammate seeds 3000/4000 from the partial CSV to recompute the combined 4-seed summaries. It handles both Axis 4 and Axis 5. Run from the repo root:

```bash
python research/research_v1/axis5/others/collect_axis45.py
```

It is idempotent and does not overwrite the teammate seed3000/seed4000 artifacts already in the package. It regenerates `reports/axis5_summary.csv`, `reports/axis5_per_seed.csv`, and `results/summary/axis5_per_seed_4seeds.csv`. Add new local seeds via `ax5_local_seeds` in the script.

## Where To Look
- Final eval CSVs (canonical + dense/easy shift): results/final_eval/
- Failure analysis: results/failure_analysis/
- Resolved configs: configs/resolved_train/ and configs/resolved_eval/
- Combined per-variant summary (mean ± std, 4 seeds): reports/axis5_summary.csv
- Full interpretation: reports/axis5_report.md
