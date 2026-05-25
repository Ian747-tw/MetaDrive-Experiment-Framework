"""One-off collector: copy local axis-4/axis-5 run artifacts into the
research/research_v1 package layout and compute summary CSVs.

Axis 4: 4 cost variants x seeds {42,2000,3000} (start_seed fixed at 2000).
Axis 5: 2 scorer variants x local seeds {42,2000}; combined with teammate
        seeds {3000,4000} already in the package for the 4-seed summary.
"""
from __future__ import annotations

import csv
import shutil
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "research_v1"
PKG = REPO / "research" / "research_v1"

METRICS = [
    "success_rate", "collision_rate", "offroad_rate", "timeout_rate",
    "route_completion_mean", "episode_cost_mean", "cost_violation_rate",
    "safety_efficiency_score",
]


def read_metrics_csv(path: Path) -> dict[str, float]:
    with path.open() as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items() if v not in ("", None)}


def selected_dev_row(path: Path) -> dict[str, float]:
    with path.open() as f:
        for row in csv.DictReader(f):
            if str(row.get("selected")).lower() == "true":
                return {k: (float(v) if _isnum(v) else v) for k, v in row.items()}
    return {}


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  MISS {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def collect_variant(axis_dir: Path, run_name: str, pkg_stem: str, seed: int,
                    shift: str | None = None) -> dict | None:
    """Copy one run's artifacts; return a per-seed summary record (canonical only)."""
    eval_suffix = "" if shift is None else f"_{shift}"
    eval_run = f"eval_{run_name}{eval_suffix}_s{seed}" if shift else f"eval_{run_name}_s{seed}"
    eval_dir = RUNS / eval_run
    final_csv = eval_dir / "eval" / "heldout_random.csv"
    if not final_csv.exists():
        print(f"  MISS final eval {final_csv}")
        return None

    tag = f"{pkg_stem}_seed{seed}" + (f"_{shift}" if shift else "")
    eval_tag = f"eval_{pkg_stem}_seed{seed}" + (f"_{shift}" if shift else "_selected_finalheldout")

    # final eval csv
    copy(final_csv, axis_dir / "results" / "final_eval" / f"{eval_tag}_heldout_random.csv")
    # failure analysis
    for fname in ("failure_summary.csv", "failure_by_mode.csv", "paper_numbers.md"):
        copy(eval_dir / "analysis" / fname,
             axis_dir / "results" / "failure_analysis" / f"{eval_tag}_{fname}")
    # eval resolved config
    copy(eval_dir / "config_resolved.yaml",
         axis_dir / "configs" / "resolved_eval" / f"{eval_tag}.yaml")

    if shift:
        return None  # shift evals are supplementary; not part of per-seed canonical summary

    # train resolved config
    copy(RUNS / f"{run_name}_s{seed}" / "config_resolved.yaml",
         axis_dir / "configs" / "resolved_train" / f"{pkg_stem}_seed{seed}.yaml")
    # checkpoint selection
    sel_csv = RUNS / "stabilization" / f"select_{run_name}_s{seed}" / "checkpoint_selection.csv"
    copy(sel_csv, axis_dir / "results" / "checkpoint_selection" / f"{pkg_stem}_seed{seed}_checkpoint_selection.csv")
    copy(RUNS / "stabilization" / f"select_{run_name}_s{seed}" / "checkpoint_selection_report.md",
         axis_dir / "results" / "checkpoint_selection" / f"{pkg_stem}_seed{seed}_checkpoint_selection_report.md")

    final = read_metrics_csv(final_csv)
    dev = selected_dev_row(sel_csv) if sel_csv.exists() else {}
    rec = {
        "seed": seed,
        "variant": pkg_stem,
        "run": f"{pkg_stem}_seed{seed}",
        "selected_checkpoint": Path(str(dev.get("checkpoint", ""))).name,
    }
    for m in ["success_rate", "collision_rate", "offroad_rate", "timeout_rate",
              "route_completion_mean", "episode_cost_mean", "safety_efficiency_score"]:
        rec[f"dev_{m}"] = dev.get(m, "")
    for m in METRICS:
        rec[m] = final.get(m, "")
    return rec


def write_per_seed(path: Path, records: list[dict]) -> None:
    cols = (["seed", "variant", "run", "selected_checkpoint"]
            + [f"dev_{m}" for m in ["success_rate", "collision_rate", "offroad_rate",
                                    "timeout_rate", "route_completion_mean",
                                    "episode_cost_mean", "safety_efficiency_score"]]
            + METRICS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["variant"], x["seed"])):
            w.writerow(r)


def write_meanstd(path: Path, records: list[dict], variants: list[str]) -> None:
    cols = ["variant", "n_seeds", "seeds"]
    for m in METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for v in variants:
            rows = [r for r in records if r["variant"] == v and r.get("success_rate") != ""]
            if not rows:
                continue
            out = {"variant": v, "n_seeds": len(rows),
                   "seeds": "|".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"]))}
            for m in METRICS:
                vals = [float(r[m]) for r in rows]
                out[f"{m}_mean"] = round(statistics.mean(vals), 4)
                out[f"{m}_std"] = round(statistics.pstdev(vals) if len(vals) == 1 else statistics.stdev(vals), 4)
            w.writerow(out)


# ---------------- Axis 4 ----------------
print("=== AXIS 4 ===")
ax4 = PKG / "axis4"
ax4_variants = {
    "axis4_cost_default_final": "axis4_cost_default",
    "axis4_cost_crash_only_final": "axis4_cost_crash_only",
    "axis4_cost_nearmiss_heavy_final": "axis4_cost_nearmiss_heavy",
    "axis4_cost_event_driving_final": "axis4_cost_event_driving",
}
ax4_seeds = [42, 2000, 3000]
ax4_records = []
for run_name, stem in ax4_variants.items():
    # template config
    copy(REPO / "configs" / "research_v1" / f"{run_name}.yaml",
         ax4 / "configs" / "templates" / f"{run_name}.yaml")
    for s in ax4_seeds:
        print(f"[axis4] {stem} seed{s}")
        rec = collect_variant(ax4, run_name, stem, s)
        if rec:
            ax4_records.append(rec)
write_per_seed(ax4 / "results" / "summary" / "axis4_per_seed.csv", ax4_records)
write_per_seed(ax4 / "reports" / "axis4_per_seed.csv", ax4_records)
write_meanstd(ax4 / "reports" / "axis4_summary.csv", ax4_records, list(ax4_variants.values()))
print(f"axis4: {len(ax4_records)} per-seed records")

# ---------------- Axis 5 ----------------
print("=== AXIS 5 ===")
ax5 = PKG / "axis5"
ax5_variants = {
    "axis5_default_scorer_final": "axis5_default_scorer",
    "axis5_near_failure_scorer_final": "axis5_near_failure_scorer",
}
ax5_local_seeds = [42, 2000]
ax5_local_records = []
for run_name, stem in ax5_variants.items():
    copy(REPO / "configs" / "research_v1" / f"{run_name}.yaml",
         ax5 / "configs" / "templates" / f"{run_name}.yaml")
    for s in ax5_local_seeds:
        print(f"[axis5] {stem} seed{s}")
        rec = collect_variant(ax5, run_name, stem, s)
        if rec:
            ax5_local_records.append(rec)
        # shift evals (supplementary; only local seeds have them)
        for shift in ("dense", "easy"):
            collect_variant(ax5, run_name, stem, s, shift=shift)

# Pull existing teammate seeds 3000/4000 from the package partial CSV
partial = ax5 / "results" / "summary" / "axis5_partial_seed3000_4000_per_seed.csv"
ax5_records = list(ax5_local_records)
if partial.exists():
    with partial.open() as f:
        for row in csv.DictReader(f):
            rec = {"seed": int(row["seed"]),
                   "variant": f"axis5_{row['variant']}",
                   "run": row["run"],
                   "selected_checkpoint": row["selected_checkpoint"]}
            for m in ["success_rate", "collision_rate", "offroad_rate", "timeout_rate",
                      "route_completion_mean", "episode_cost_mean", "safety_efficiency_score"]:
                rec[f"dev_{m}"] = row.get(f"dev_{m}", "")
            for m in METRICS:
                rec[m] = row.get(m, "")
            ax5_records.append(rec)

write_per_seed(ax5 / "results" / "summary" / "axis5_per_seed_4seeds.csv", ax5_records)
write_per_seed(ax5 / "reports" / "axis5_per_seed.csv", ax5_records)
write_meanstd(ax5 / "reports" / "axis5_summary.csv", ax5_records, list(ax5_variants.values()))
print(f"axis5: {len(ax5_records)} per-seed records (local {len(ax5_local_records)} + teammate)")
print("DONE")
