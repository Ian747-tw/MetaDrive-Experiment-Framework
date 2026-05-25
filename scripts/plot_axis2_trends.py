import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import shutil

def main():
    csv_path = Path(r"C:\Users\white\Downloads\summary_main_results.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    # Extract seed and ratio columns dynamically
    def get_seed(eval_run):
        match = re.search(r'_s(\d+)$', str(eval_run))
        if match:
            return int(match.group(1))
        return None
        
    df["seed"] = df["eval_run"].apply(get_seed)
    
    def get_ratio(m):
        match = re.search(r'mixed(\d+)', str(m))
        if match:
            return int(match.group(1)) / 100.0
        return None
        
    df["ratio"] = df["method"].apply(get_ratio)

    # Filter for Axis 2 multi-seed runs
    # Filter for Axis 2 multi-seed runs
    # Only keep seeds < 3000 because others fall outside the failure buffer
    df_ms = df[
        (df["seed"] < 3000) & 
        df["method"].str.startswith("axis2_mixed")
    ].copy()
    
    df_ms = df_ms.dropna(subset=["ratio", "seed"])
    df_ms = df_ms.sort_values(by=["seed", "ratio"])

    # Output paths
    base_dir = Path(r"c:\Users\white\Desktop\hw\DRL\hw5\MetaDrive-Experiment-Framework")
    reports_dir = base_dir / "research" / "research_v1" / "axis2" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts_dir = Path(r"C:\Users\white\.gemini\antigravity\brain\d93b5628-02f5-44cc-9c3e-b3bce99d0dff")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {
        2000: "#1f77b4", # blue
        3000: "#ff7f0e", # orange
        4000: "#2ca02c", # green
        5000: "#d62728", # red
        6000: "#9467bd", # purple
        7000: "#8c564b"  # brown
    }

    metrics = {
        "success_rate": ("Success Rate", "Success Rate", [0, 1.0]),
        "collision_rate": ("Collision Rate", "Collision Rate", [0, 1.0]),
        "timeout_rate": ("Timeout Rate", "Timeout Rate", [0, 1.0]),
        "safety_efficiency_score": ("Safety-Efficiency Score", "Safety-Efficiency Score", [-2.0, 0.5])
    }

    for metric_col, (title, ylabel, ylim) in metrics.items():
        plt.figure(figsize=(10, 6), dpi=150)
        
        # Plot each seed
        seeds = sorted(df_ms["seed"].unique())
        ratios = sorted(df_ms["ratio"].unique())
        
        for seed in seeds:
            seed_df = df_ms[df_ms["seed"] == seed].sort_values(by="ratio")
            plt.plot(
                seed_df["ratio"], 
                seed_df[metric_col], 
                marker='o', 
                linewidth=2, 
                color=colors.get(seed, "#7f7f7f"), 
                label=f"Seed {seed}",
                alpha=0.85
            )

        # Compute Mean and Standard Deviation across seeds
        mean_vals = []
        std_vals = []
        for r in ratios:
            r_vals = df_ms[df_ms["ratio"] == r][metric_col].values
            mean_vals.append(np.mean(r_vals))
            std_vals.append(np.std(r_vals))
        
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)

        # Plot Mean Trend Line (thick black dashed line)
        plt.plot(
            ratios, 
            mean_vals, 
            color="black", 
            linestyle="--", 
            linewidth=3.5, 
            label="Overall Mean",
            zorder=10
        )
        
        # Shade the Standard Deviation band
        plt.fill_between(
            ratios, 
            mean_vals - std_vals, 
            mean_vals + std_vals, 
            color="black", 
            alpha=0.1, 
            label="1 Std Dev Band",
            zorder=1
        )

        # Formatting
        plt.title(f"Axis 2 Sampler Ablation: {title} vs Failure Replay Ratio", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Failure Replay Ratio", fontsize=12, labelpad=10)
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.xticks(ratios, [f"{r:.2f}" for r in ratios])
        plt.ylim(ylim)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        # Save to reports and artifacts
        dest_filename = f"axis2_{metric_col}_trends.png"
        dest_path = reports_dir / dest_filename
        plt.savefig(dest_path)
        plt.close()
        
        # Copy to artifacts directory
        artifact_path = artifacts_dir / dest_filename
        shutil.copy2(dest_path, artifact_path)
        print(f"Generated and copied: {dest_filename}")

    print("All line charts generated successfully!")

if __name__ == "__main__":
    main()
