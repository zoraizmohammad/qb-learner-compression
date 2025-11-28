"""
Aggregate λ-sweep experiment results and create summary visualizations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Define the lambda values and their corresponding run directories
lambda_sweep_runs = [
    (0.0, "compressed_2025-11-28_15-50-44"),
    (0.001, "compressed_2025-11-28_15-55-12"),
    (0.01, "compressed_2025-11-28_15-59-36"),
    (0.05, "compressed_2025-11-28_16-03-55"),
    (0.1, "compressed_2025-11-28_16-08-17"),
    (0.2, "compressed_2025-11-28_16-12-38"),
]

results_dir = Path("results/runs")
output_dir = Path("results/lambda_sweep_summary")
output_dir.mkdir(parents=True, exist_ok=True)

# Collect all results
results = []

for lam, run_name in lambda_sweep_runs:
    metrics_path = results_dir / run_name / "final_metrics.json"
    config_path = results_dir / run_name / "config.json"
    
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found, skipping...")
        continue
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    result = {
        "lambda": lam,
        "experiment_name": f"compressed_pothos_large_lam{lam}",
        "final_loss": metrics.get("final_loss", np.nan),
        "final_ce_loss": metrics.get("final_ce_loss", np.nan),
        "final_accuracy": metrics.get("final_accuracy", np.nan),
        "two_qubit_count": metrics.get("final_two_qubit_count", np.nan),
        "mask_sparsity": metrics.get("final_mask_sparsity", np.nan),
        "active_entanglers": metrics.get("final_active_entanglers", np.nan),
        "total_entanglers": metrics.get("total_entanglers", np.nan),
        "n_iterations": metrics.get("n_iterations", 150),
        "n_qubits": metrics.get("n_qubits", 2),
        "depth": metrics.get("depth", 3),
    }
    results.append(result)

# Create DataFrame
df = pd.DataFrame(results)

# Sort by lambda
df = df.sort_values("lambda")

# Save CSV summary
csv_path = output_dir / "lambda_sweep_summary.csv"
df.to_csv(csv_path, index=False)
print(f"Saved summary CSV to {csv_path}")

# Print summary table
print("\n" + "="*80)
print("λ-Sweep Results Summary")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. CE Loss vs Lambda
ax = axes[0, 0]
ax.plot(df["lambda"], df["final_ce_loss"], marker='o', linewidth=2, markersize=8)
ax.set_xlabel("λ (Regularization Strength)", fontsize=12)
ax.set_ylabel("Final CE Loss", fontsize=12)
ax.set_title("Cross-Entropy Loss vs λ", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 2. Total Loss vs Lambda (includes regularization term)
ax = axes[0, 1]
ax.plot(df["lambda"], df["final_loss"], marker='s', linewidth=2, markersize=8, color='orange')
ax.set_xlabel("λ (Regularization Strength)", fontsize=12)
ax.set_ylabel("Final Total Loss", fontsize=12)
ax.set_title("Total Loss (CE + λ·Cost) vs λ", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 3. Gate Cost vs Lambda
ax = axes[1, 0]
ax.plot(df["lambda"], df["two_qubit_count"], marker='^', linewidth=2, markersize=8, color='green')
ax.set_xlabel("λ (Regularization Strength)", fontsize=12)
ax.set_ylabel("2-Qubit Gate Count", fontsize=12)
ax.set_title("Gate Cost vs λ", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 4. Pareto Front: CE Loss vs Gate Cost
ax = axes[1, 1]
scatter = ax.scatter(df["two_qubit_count"], df["final_ce_loss"], 
                     c=df["lambda"], s=150, cmap='viridis', 
                     edgecolors='black', linewidths=1.5, alpha=0.7)
ax.set_xlabel("2-Qubit Gate Count", fontsize=12)
ax.set_ylabel("Final CE Loss", fontsize=12)
ax.set_title("Pareto Front: CE Loss vs Gate Cost", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("λ (Regularization Strength)", fontsize=11)

# Annotate points with lambda values
for idx, row in df.iterrows():
    ax.annotate(f"λ={row['lambda']}", 
                (row['two_qubit_count'], row['final_ce_loss']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plot_path = output_dir / "lambda_sweep_analysis.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved analysis plot to {plot_path}")
plt.close()

# Create a detailed Pareto front plot
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Plot all points
scatter = ax.scatter(df["two_qubit_count"], df["final_ce_loss"], 
                     c=df["lambda"], s=200, cmap='viridis', 
                     edgecolors='black', linewidths=2, alpha=0.8, zorder=3)

# Annotate with lambda values
for idx, row in df.iterrows():
    ax.annotate(f"λ={row['lambda']}", 
                (row['two_qubit_count'], row['final_ce_loss']),
                xytext=(8, 8), textcoords='offset points', 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel("2-Qubit Gate Count", fontsize=14, fontweight='bold')
ax.set_ylabel("Cross-Entropy Loss", fontsize=14, fontweight='bold')
ax.set_title("Pareto Front Analysis: λ-Sweep Results", fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("λ (Regularization Strength)", fontsize=12, fontweight='bold')

plt.tight_layout()
pareto_path = output_dir / "pareto_front_lambda_sweep.png"
plt.savefig(pareto_path, dpi=150, bbox_inches='tight')
print(f"Saved Pareto front plot to {pareto_path}")
plt.close()

print("\n✓ λ-sweep results aggregated successfully!")
print(f"  Summary CSV: {csv_path}")
print(f"  Analysis plots: {output_dir}")

