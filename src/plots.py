"""
plots.py

Plotting utilities for visualizing training results.

Provides functions for plotting loss curves, CNOT vs accuracy scatter plots,
and Pareto frontier plots for comparing baseline and compressed models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


def plot_loss_curves(
    history: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Training Loss Curves",
) -> None:
    """
    Plot loss curves over training iterations.
    
    Parameters
    ----------
    history : pd.DataFrame
        Training history with columns: iteration, loss, ce_loss
    save_path : Path, optional
        Path to save the figure. If None, displays the plot.
    title : str
        Plot title.
    """
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cnot_vs_accuracy(
    history: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "CNOT Count vs Accuracy",
) -> None:
    """
    Plot CNOT count vs accuracy (1 - CE loss) scatter plot.
    
    Parameters
    ----------
    history : pd.DataFrame
        Training history with columns: iteration, two_qubit_count, ce_loss
    save_path : Path, optional
        Path to save the figure. If None, displays the plot.
    title : str
        Plot title.
    """
    if "two_qubit_count" not in history.columns or "ce_loss" not in history.columns:
        print("Warning: Missing required columns for CNOT vs accuracy plot")
        return
    
    # Compute accuracy (1 - normalized CE loss, or use a different metric)
    # For binary classification, we can use 1 - ce_loss as a proxy
    accuracy = 1.0 - history["ce_loss"] / history["ce_loss"].max()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(history["two_qubit_count"], accuracy, alpha=0.6, s=50, c=history["iteration"], cmap="viridis")
    plt.colorbar(label="Iteration")
    plt.xlabel("Two-Qubit Gate Count")
    plt.ylabel("Accuracy (1 - normalized CE loss)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def pareto_plot(
    baseline_history: pd.DataFrame,
    compressed_history: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Pareto Frontier: Baseline vs Compressed",
) -> None:
    """
    Plot Pareto frontier comparing baseline and compressed models.
    
    Shows the trade-off between CNOT count and accuracy for both models.
    
    Parameters
    ----------
    baseline_history : pd.DataFrame
        Baseline training history.
    compressed_history : pd.DataFrame
        Compressed training history.
    save_path : Path, optional
        Path to save the figure. If None, displays the plot.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute accuracy for both
    baseline_ce = baseline_history["ce_loss"]
    compressed_ce = compressed_history["ce_loss"]
    
    # Normalize CE loss to get accuracy proxy
    all_ce = pd.concat([baseline_ce, compressed_ce])
    max_ce = all_ce.max()
    
    baseline_acc = 1.0 - baseline_ce / max_ce
    compressed_acc = 1.0 - compressed_ce / max_ce
    
    # Plot baseline
    ax.scatter(
        baseline_history["two_qubit_count"],
        baseline_acc,
        label="Baseline",
        alpha=0.6,
        s=50,
        color="blue",
    )
    
    # Plot compressed
    ax.scatter(
        compressed_history["two_qubit_count"],
        compressed_acc,
        label="Compressed",
        alpha=0.6,
        s=50,
        color="red",
    )
    
    # Plot final points
    ax.scatter(
        baseline_history["two_qubit_count"].iloc[-1],
        baseline_acc.iloc[-1],
        label="Baseline (final)",
        s=200,
        color="blue",
        marker="*",
        edgecolors="black",
        linewidths=2,
    )
    
    ax.scatter(
        compressed_history["two_qubit_count"].iloc[-1],
        compressed_acc.iloc[-1],
        label="Compressed (final)",
        s=200,
        color="red",
        marker="*",
        edgecolors="black",
        linewidths=2,
    )
    
    ax.set_xlabel("Two-Qubit Gate Count")
    ax.set_ylabel("Accuracy (1 - normalized CE loss)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mask_sparsity(
    history: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Mask Sparsity Over Training",
) -> None:
    """
    Plot mask sparsity (fraction of active entanglers) over training.
    
    Parameters
    ----------
    history : pd.DataFrame
        Training history with columns: iteration, mask_sparsity
    save_path : Path, optional
        Path to save the figure. If None, displays the plot.
    title : str
        Plot title.
    """
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

