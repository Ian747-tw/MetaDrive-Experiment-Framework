import pandas as pd
from pathlib import Path
import re

def main():
    csv_path = Path(r"C:\Users\white\Downloads\summary_main_results.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Standard headers for individual eval CSV files
    eval_cols = [
        "n_episodes", "success_rate", "collision_rate", "offroad_rate", 
        "timeout_rate", "route_completion_mean", "episode_reward_mean", 
        "episode_modified_reward_mean", "episode_cost_mean", 
        "cost_violation_rate", "avg_episode_length", "safety_efficiency_score"
    ]

    # Create base paths
    base_dir = Path(r"c:\Users\white\Desktop\hw\DRL\hw5\MetaDrive-Experiment-Framework\research\research_v1")
    
    # Process each row
    axis2_rows = []

    for _, row in df.iterrows():
        method_name = str(row["method"])
        eval_run = str(row["eval_run"])
        
        # Determine Axis (Only Axis 2 is processed on this branch)
        axis = None
        if "axis2" in method_name or "axis2" in eval_run:
            axis = 2
        else:
            continue

        # Extract seed
        seed_match = re.search(r'_s(\d+)$', eval_run)
        if seed_match:
            seed = int(seed_match.group(1))
        elif "eval_base_pretrain" in eval_run:
            seed = 42
        else:
            seed = 42

        # Create output directories
        axis_dir = base_dir / f"axis{axis}"
        final_eval_dir = axis_dir / "results" / "final_eval"
        final_eval_dir.mkdir(parents=True, exist_ok=True)

        # Write individual CSV
        eval_file = final_eval_dir / f"{eval_run}_heldout_random.csv"
        # Select matching columns
        row_dict = row.to_dict()
        eval_row = {col: row_dict.get(col, 0.0) for col in eval_cols}
        pd.DataFrame([eval_row])[eval_cols].to_csv(eval_file, index=False)
        # print(f"Wrote: {eval_file}")

        # Store for aggregation
        row_dict["seed"] = seed
        row_dict["axis"] = axis
        if axis == 2:
            axis2_rows.append(row_dict)

    print("Finished writing individual eval CSVs.")

    # --- AGGREGATE AXIS 2 ---
    if axis2_rows:
        axis2_df = pd.DataFrame(axis2_rows)
        summary_dir = base_dir / "axis2" / "results" / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = base_dir / "axis2" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        others_dir = base_dir / "axis2" / "others"
        others_dir.mkdir(parents=True, exist_ok=True)

        # 1. Multi-seed Variance Analysis (seeds 2000-7000, methods mixed005, mixed030, mixed060, mixed090)
        ms_df = axis2_df[axis2_df["seed"].isin([2000, 3000, 4000, 5000, 6000, 7000]) & axis2_df["method"].str.startswith("axis2_mixed")].copy()
        
        # We need a clean method name (e.g. mixed005, mixed030)
        def clean_method(m):
            m = m.replace("axis2_", "")
            m = re.sub(r'_s\d+$', '', m)
            return m
        ms_df["method_clean"] = ms_df["method"].apply(clean_method)

        # Sort values for consistency
        ms_df = ms_df.sort_values(by=["method_clean", "seed"])

        # Write multiseed_axis2_per_seed.csv
        per_seed_cols = ["seed", "method", "eval_run", "success_rate", "collision_rate", "offroad_rate", "timeout_rate", "route_completion_mean", "episode_cost_mean", "safety_efficiency_score"]
        # Use clean method name in per-seed output
        per_seed_df = ms_df.copy()
        per_seed_df["method"] = per_seed_df["method_clean"]
        per_seed_df[per_seed_cols].to_csv(summary_dir / "multiseed_axis2_per_seed.csv", index=False)
        print(f"Wrote: {summary_dir / 'multiseed_axis2_per_seed.csv'}")

        # Compute summary stats (3 seeds vs 6 seeds)
        summary_rows = []
        metrics = ["success_rate", "collision_rate", "offroad_rate", "timeout_rate", "route_completion_mean", "episode_cost_mean", "safety_efficiency_score"]

        # 6_train_first3 (seeds 2000, 3000, 4000)
        first3_df = ms_df[ms_df["seed"].isin([2000, 3000, 4000])]
        for method_name, group in first3_df.groupby("method_clean"):
            row = {
                "block": "6_train_first3",
                "block_trains": 6,
                "method": method_name,
                "n_seeds": len(group)
            }
            for m in metrics:
                row[f"{m}_mean"] = group[m].mean()
                row[f"{m}_std"] = group[m].std()
            summary_rows.append(row)

        # 12_train_all6 (seeds 2000, 3000, 4000, 5000, 6000, 7000)
        for method_name, group in ms_df.groupby("method_clean"):
            row = {
                "block": "12_train_all6",
                "block_trains": 12,
                "method": method_name,
                "n_seeds": len(group)
            }
            for m in metrics:
                row[f"{m}_mean"] = group[m].mean()
                row[f"{m}_std"] = group[m].std()
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        # Order columns to match Axis 1
        summary_cols = ["block", "block_trains", "method", "n_seeds"]
        for m in metrics:
            summary_cols.extend([f"{m}_mean", f"{m}_std"])
        summary_df[summary_cols].to_csv(summary_dir / "multiseed_axis2_summary.csv", index=False)
        print(f"Wrote: {summary_dir / 'multiseed_axis2_summary.csv'}")

        # 2. Single-seed Sweep (seed 42, methods starts with axis2_sampler_)
        ss_df = axis2_df[axis2_df["seed"] == 42].copy()
        
        def extract_ratio(m):
            if "uniform" in m:
                return "uniform"
            match = re.search(r'mixed(\d+)', m)
            if match:
                val = int(match.group(1))
                if val == 95:
                    return 0.95
                if val == 99:
                    return 0.99
                return val / 100.0
            return "unknown"

        ss_df["failure_ratio"] = ss_df["method"].apply(extract_ratio)
        ss_df = ss_df.sort_values(by="failure_ratio", key=lambda x: x.map(lambda v: v if isinstance(v, float) else (1.01 if v == "uniform" else -1)))

        sweep_cols = ["failure_ratio", "method", "success_rate", "collision_rate", "offroad_rate", "timeout_rate", "route_completion_mean", "episode_cost_mean", "safety_efficiency_score"]
        ss_df[sweep_cols].to_csv(summary_dir / "single_seed_axis2_sweep.csv", index=False)
        print(f"Wrote: {summary_dir / 'single_seed_axis2_sweep.csv'}")


    print("All data structured and aggregated successfully!")

if __name__ == "__main__":
    main()
