"""
plots.py

Plotting utilities for visualizing training results in the quantum-Bayesian-learner project.

This module provides comprehensive visualization tools for:
- Learning curves (loss vs epochs)
- Accuracy curves (accuracy vs epochs)
- Two-qubit gate counts over training
- Pareto front between accuracy (or loss) vs 2-qubit gate cost
- Mask visualization (active vs pruned entanglers)
- Comparison plots of baseline vs compressed runs
- Histograms or bar charts of gate costs per variant
- Heatmaps of pruning patterns across depths / layers
- Density-matrix fidelity comparisons
- Scatter plots of predicted vs true probabilities
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

# Try to import seaborn for nicer aesthetics (optional)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # Set matplotlib style manually
    plt.style.use("default")


def ensure_dir() -> None:
    """Ensure the results directory exists."""
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")


def plot_loss_curve(loss_history: List[float], title: str, fname: str) -> None:
    """
    Plot loss curve over training iterations.
    
    Parameters
    ----------
    loss_history : List[float]
        List of loss values for each iteration/epoch.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(8, 6))
    iterations = range(len(loss_history))
    plt.plot(iterations, loss_history, linewidth=2, label="Loss")
    plt.xlabel("Iteration / Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_accuracy_curve(acc_history: List[float], title: str, fname: str) -> None:
    """
    Plot accuracy curve over training iterations.
    
    Parameters
    ----------
    acc_history : List[float]
        List of accuracy values for each iteration/epoch.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(8, 6))
    iterations = range(len(acc_history))
    plt.plot(iterations, acc_history, linewidth=2, label="Accuracy", color="green")
    plt.xlabel("Iteration / Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_gate_cost_curve(gate_history: List[int], title: str, fname: str) -> None:
    """
    Plot two-qubit gate count over training iterations.
    
    Parameters
    ----------
    gate_history : List[int]
        List of two-qubit gate counts for each iteration/epoch.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(8, 6))
    iterations = range(len(gate_history))
    plt.plot(iterations, gate_history, linewidth=2, label="Two-Qubit Gate Count", color="red", marker="o", markersize=4)
    plt.xlabel("Iteration / Epoch", fontsize=12)
    plt.ylabel("Two-Qubit Gate Count", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_pareto_front(costs: List[int], accuracies: List[float], title: str, fname: str) -> None:
    """
    Plot Pareto front showing trade-off between gate cost and accuracy.
    
    Parameters
    ----------
    costs : List[int]
        List of two-qubit gate counts.
    accuracies : List[float]
        List of accuracy values corresponding to each cost.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(costs, accuracies, alpha=0.6, s=100, c=range(len(costs)), cmap="viridis", edgecolors="black", linewidths=0.5)
    plt.colorbar(label="Iteration")
    plt.xlabel("Two-Qubit Gate Count (Cost)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    # Highlight Pareto-optimal points (lower cost, higher accuracy is better)
    if len(costs) > 0:
        costs_arr = np.array(costs)
        acc_arr = np.array(accuracies)
        
        # Find Pareto-optimal points (points that are not dominated)
        pareto_mask = np.ones(len(costs), dtype=bool)
        for i in range(len(costs)):
            for j in range(len(costs)):
                if i != j:
                    # Point j dominates point i if it has lower cost AND higher accuracy
                    if costs_arr[j] <= costs_arr[i] and acc_arr[j] >= acc_arr[i]:
                        if costs_arr[j] < costs_arr[i] or acc_arr[j] > acc_arr[i]:
                            pareto_mask[i] = False
                            break
        
        pareto_costs = costs_arr[pareto_mask]
        pareto_accs = acc_arr[pareto_mask]
        
        if len(pareto_costs) > 0:
            # Sort for plotting
            sort_idx = np.argsort(pareto_costs)
            pareto_costs = pareto_costs[sort_idx]
            pareto_accs = pareto_accs[sort_idx]
            plt.plot(pareto_costs, pareto_accs, 'r--', linewidth=2, alpha=0.7, label="Pareto Front")
            plt.scatter(pareto_costs, pareto_accs, s=200, c="red", marker="*", edgecolors="black", linewidths=1.5, label="Pareto-Optimal", zorder=5)
    
    plt.legend()
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_mask_heatmap(mask: np.ndarray, title: str, fname: str) -> None:
    """
    Plot heatmap of mask showing active vs inactive entangling blocks.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask array with shape (depth, num_edges).
        Values of 1 indicate active entanglers, 0 indicates pruned.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(10, 6))
    
    # Create heatmap using seaborn
    depth, num_edges = mask.shape
    
    # Create labels for axes
    depth_labels = [f"Depth {d}" for d in range(depth)]
    edge_labels = [f"Edge {e}" for e in range(num_edges)]
    
    # Plot heatmap - use seaborn if available, otherwise use matplotlib
    if HAS_SEABORN:
        sns.heatmap(
            mask,
            annot=True,
            fmt="d",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=edge_labels,
            yticklabels=depth_labels,
            cbar_kws={"label": "Active (1) / Inactive (0)"},
            linewidths=0.5,
            linecolor="black",
        )
    else:
        # Fallback to matplotlib imshow
        im = plt.imshow(mask, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, label="Active (1) / Inactive (0)")
        plt.xticks(range(num_edges), edge_labels)
        plt.yticks(range(depth), depth_labels)
        
        # Add annotations
        for i in range(depth):
            for j in range(num_edges):
                plt.text(j, i, int(mask[i, j]), ha="center", va="center", color="black", fontweight="bold")
    
    plt.xlabel("Edge Index", fontsize=12)
    plt.ylabel("Depth / Layer", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def compare_methods_bar(method_results: Dict[str, Dict[str, float]], fname: str) -> None:
    """
    Produce a bar chart comparing different methods.
    
    Parameters
    ----------
    method_results : Dict[str, Dict[str, float]]
        Dictionary mapping method names to their results.
        Each method should have "acc" (accuracy) and "cost" (gate count) keys.
        Example:
        {
            "baseline": {"acc": 0.85, "cost": 12},
            "compiled": {"acc": 0.84, "cost": 10},
            "naive_prune": {"acc": 0.82, "cost": 8},
            "compressed": {"acc": 0.83, "cost": 6}
        }
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    methods = list(method_results.keys())
    accuracies = [method_results[m]["acc"] for m in methods]
    costs = [method_results[m]["cost"] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot cost comparison
    bars2 = ax2.bar(methods, costs, color="coral", alpha=0.7, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Two-Qubit Gate Count", fontsize=12)
    ax2.set_title("Gate Cost Comparison", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.suptitle("Method Comparison: Baseline vs Compressed", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_density_fidelity(fidelities: List[float], title: str, fname: str) -> None:
    """
    Plot density matrix fidelity over training iterations.
    
    Useful for debugging belief-state behavior and tracking how well
    the quantum state matches expected distributions.
    
    Parameters
    ----------
    fidelities : List[float]
        List of fidelity values (between 0 and 1) for each iteration.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(8, 6))
    iterations = range(len(fidelities))
    plt.plot(iterations, fidelities, linewidth=2, label="Fidelity", color="purple", marker="o", markersize=4)
    plt.xlabel("Iteration / Epoch", fontsize=12)
    plt.ylabel("Density Matrix Fidelity", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_pred_vs_true(preds: np.ndarray, labels: np.ndarray, title: str, fname: str) -> None:
    """
    Plot scatter plot of predicted probabilities vs true labels.
    
    Parameters
    ----------
    preds : np.ndarray
        Array of predicted probabilities (shape: (n_samples,)).
    labels : np.ndarray
        Array of true labels (shape: (n_samples,)). Should be 0 or 1.
    title : str
        Plot title.
    fname : str
        Filename (without extension) to save the plot.
    """
    ensure_dir()
    
    plt.figure(figsize=(8, 6))
    
    # Separate predictions by true label
    preds_class0 = preds[labels == 0]
    preds_class1 = preds[labels == 1]
    
    # Plot scatter with different colors for each class
    plt.scatter(preds_class0, np.zeros_like(preds_class0), alpha=0.6, s=100, label="Class 0 (True)", color="blue", marker="o")
    plt.scatter(preds_class1, np.ones_like(preds_class1), alpha=0.6, s=100, label="Class 1 (True)", color="red", marker="s")
    
    # Add diagonal reference line (perfect predictions)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label="Perfect Prediction")
    
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    
    save_path = f"results/{fname}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


# Additional utility functions for compatibility with existing code

def plot_loss_curves(
    history,
    save_path: Optional[str] = None,
    title: str = "Training Loss Curves",
) -> None:
    """
    Plot loss curves from pandas DataFrame (for compatibility with existing code).
    
    Parameters
    ----------
    history : pd.DataFrame
        Training history with columns: iteration, loss, ce_loss
    save_path : str, optional
        Path to save the figure. If None, uses default naming.
    title : str
        Plot title.
    """
    ensure_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot total loss
    axes[0].plot(history["iteration"], history["loss"], label="Total Loss", linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss (CE + λ * CNOT)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot CE loss
    if "ce_loss" in history.columns:
        axes[1].plot(history["iteration"], history["ce_loss"], label="CE Loss", linewidth=2, color="orange")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Cross-Entropy Loss")
        axes[1].set_title("Classification Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        save_path = "results/figures/loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_mask_sparsity(
    history,
    save_path: Optional[str] = None,
    title: str = "Mask Sparsity Over Training",
) -> None:
    """
    Plot mask sparsity from pandas DataFrame (for compatibility with existing code).
    
    Parameters
    ----------
    history : pd.DataFrame
        Training history with columns: iteration, mask_sparsity
    save_path : str, optional
        Path to save the figure. If None, uses default naming.
    title : str
        Plot title.
    """
    ensure_dir()
    
    if "mask_sparsity" not in history.columns:
        print("Warning: mask_sparsity column not found in history")
        return
    
    plt.figure(figsize=(8, 5))
    plt.plot(history["iteration"], history["mask_sparsity"], linewidth=2, color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Mask Sparsity (fraction of active entanglers)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    if save_path:
        from pathlib import Path
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        save_path = "results/figures/mask_sparsity.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.close()
