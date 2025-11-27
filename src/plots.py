"""
plots.py

Plotting utilities for visualizing training results in the quantum-Bayesian-learner project.

This module provides comprehensive visualization tools for quantum Bayesian learner experiments,
including learning curves, gate cost dynamics, Pareto trade-offs, mask/pruning patterns, and
method comparison plots suitable for publication.

Expected Usage:
---------------
Training scripts (train_baseline.py, train_compressed.py) should maintain histories as
dictionaries with keys like:
    - "iteration": List[int] or np.ndarray
    - "loss": List[float] or np.ndarray
    - "ce_loss": List[float] or np.ndarray (optional)
    - "acc": List[float] or np.ndarray (optional)
    - "gate_count" or "two_qubit_count": List[int] or np.ndarray (optional)
    - "mask_sparsity": List[float] or np.ndarray (optional)

At the end of training, call high-level convenience functions:
    - plot_all_curves_from_history(history, prefix="baseline")
    - plot_baseline_vs_compressed_histories(baseline_history, compressed_history)

Or use individual plotting functions for custom visualizations.

All functions save plots to a specified output directory (default: "results/figures").
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Try to import seaborn for nicer aesthetics (optional)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use("default")


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : str
        Directory path to ensure exists.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def log(message: str, verbose: bool = True) -> None:
    """
    Log a message (can be muted or redirected).
    
    Parameters
    ----------
    message : str
        Message to log.
    verbose : bool, optional
        If False, suppress output (default: True).
    """
    if verbose:
        print(message)


def _validate_array_lengths(*arrays: Union[List, np.ndarray], name: str = "arrays") -> None:
    """
    Validate that all arrays have the same length.
    
    Parameters
    ----------
    *arrays : Union[List, np.ndarray]
        Arrays to validate.
    name : str, optional
        Name for error message (default: "arrays").
    
    Raises
    ------
    ValueError
        If arrays have different lengths.
    """
    if len(arrays) < 2:
        return
    
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"{name} must have the same length, got lengths: {lengths}"
        )


def _to_numpy_array(data: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert input to numpy array, handling empty inputs.
    
    Parameters
    ----------
    data : Union[List, np.ndarray]
        Input data to convert.
    
    Returns
    -------
    np.ndarray
        Numpy array.
    
    Raises
    ------
    ValueError
        If data is empty.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("Input data cannot be empty")
    return arr


# ============================================================================
# Core Plotting Functions (Individual)
# ============================================================================

def plot_loss_curve(
    loss_history: Union[List[float], np.ndarray],
    title: str = "Training Loss",
    fname: str = "loss_curve",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot loss curve over training iterations.
    
    Parameters
    ----------
    loss_history : Union[List[float], np.ndarray]
        List or array of loss values for each iteration/epoch.
    title : str, optional
        Plot title (default: "Training Loss").
    fname : str, optional
        Filename (without extension) to save the plot (default: "loss_curve").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_loss_curve
    >>> history = {"loss": [0.5, 0.4, 0.3, 0.25]}
    >>> plot_loss_curve(history["loss"], title="Baseline Loss", fname="baseline_loss")
    """
    try:
        loss_arr = _to_numpy_array(loss_history)
    except ValueError:
        warnings.warn("Empty loss history provided, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 6))
    iterations = np.arange(len(loss_arr))
    plt.plot(iterations, loss_arr, linewidth=2.5, label="Loss", color="#1f77b4")
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_accuracy_curve(
    acc_history: Union[List[float], np.ndarray],
    title: str = "Training Accuracy",
    fname: str = "accuracy_curve",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot accuracy curve over training iterations.
    
    Parameters
    ----------
    acc_history : Union[List[float], np.ndarray]
        List or array of accuracy values for each iteration/epoch.
    title : str, optional
        Plot title (default: "Training Accuracy").
    fname : str, optional
        Filename (without extension) to save the plot (default: "accuracy_curve").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_accuracy_curve
    >>> history = {"acc": [0.6, 0.7, 0.8, 0.85]}
    >>> plot_accuracy_curve(history["acc"], title="Baseline Accuracy")
    """
    try:
        acc_arr = _to_numpy_array(acc_history)
    except ValueError:
        warnings.warn("Empty accuracy history provided, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 6))
    iterations = np.arange(len(acc_arr))
    plt.plot(iterations, acc_arr, linewidth=2.5, label="Accuracy", color="#2ca02c", marker="o", markersize=4)
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_gate_cost_curve(
    gate_history: Union[List[int], np.ndarray],
    title: str = "Two-Qubit Gate Count",
    fname: str = "gate_cost_curve",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot two-qubit gate count over training iterations.
    
    Parameters
    ----------
    gate_history : Union[List[int], np.ndarray]
        List or array of two-qubit gate counts for each iteration/epoch.
    title : str, optional
        Plot title (default: "Two-Qubit Gate Count").
    fname : str, optional
        Filename (without extension) to save the plot (default: "gate_cost_curve").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_gate_cost_curve
    >>> history = {"gate_count": [12, 10, 8, 6]}
    >>> plot_gate_cost_curve(history["gate_count"], title="Gate Cost Over Training")
    """
    try:
        gate_arr = _to_numpy_array(gate_history)
    except ValueError:
        warnings.warn("Empty gate history provided, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 6))
    iterations = np.arange(len(gate_arr))
    plt.plot(iterations, gate_arr, linewidth=2.5, label="Two-Qubit Gate Count", 
             color="#d62728", marker="s", markersize=5)
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Two-Qubit Gate Count", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_pareto_front(
    costs: Union[List[int], np.ndarray],
    accuracies: Union[List[float], np.ndarray],
    title: str = "Pareto Front: Accuracy vs Gate Cost",
    fname: str = "pareto_front",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot Pareto front showing trade-off between gate cost and accuracy.
    
    Parameters
    ----------
    costs : Union[List[int], np.ndarray]
        List or array of two-qubit gate counts.
    accuracies : Union[List[float], np.ndarray]
        List or array of accuracy values corresponding to each cost.
    title : str, optional
        Plot title (default: "Pareto Front: Accuracy vs Gate Cost").
    fname : str, optional
        Filename (without extension) to save the plot (default: "pareto_front").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_pareto_front
    >>> costs = [20, 15, 10, 8, 6]
    >>> accs = [0.85, 0.84, 0.83, 0.82, 0.80]
    >>> plot_pareto_front(costs, accs, title="Compression Trade-off")
    """
    try:
        costs_arr = _to_numpy_array(costs)
        acc_arr = _to_numpy_array(accuracies)
        _validate_array_lengths(costs_arr, acc_arr, name="costs and accuracies")
    except ValueError as e:
        warnings.warn(f"Invalid input for Pareto plot: {e}, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with iteration coloring
    scatter = plt.scatter(costs_arr, acc_arr, alpha=0.7, s=120, 
                         c=np.arange(len(costs_arr)), cmap="viridis",
                         edgecolors="black", linewidths=0.8, zorder=3)
    plt.colorbar(scatter, label="Iteration", fontsize=11)
    
    # Find and highlight Pareto-optimal points
    if len(costs_arr) > 0:
        pareto_mask = np.ones(len(costs_arr), dtype=bool)
        for i in range(len(costs_arr)):
            for j in range(len(costs_arr)):
                if i != j:
                    # Point j dominates point i if lower cost AND higher accuracy
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
            plt.plot(pareto_costs, pareto_accs, 'r--', linewidth=2.5, 
                    alpha=0.8, label="Pareto Front", zorder=2)
            plt.scatter(pareto_costs, pareto_accs, s=250, c="red", marker="*",
                       edgecolors="black", linewidths=1.5, label="Pareto-Optimal", zorder=4)
    
    plt.xlabel("Two-Qubit Gate Count (Cost)", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_mask_heatmap(
    mask: np.ndarray,
    title: str = "Mask Heatmap",
    fname: str = "mask_heatmap",
    output_dir: str = "results/figures",
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    verbose: bool = True
) -> None:
    """
    Plot heatmap of mask showing active vs inactive entangling blocks.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask array with shape (depth, num_edges).
        Values of 1 indicate active entanglers, 0 indicates pruned.
    title : str, optional
        Plot title (default: "Mask Heatmap").
    fname : str, optional
        Filename (without extension) to save the plot (default: "mask_heatmap").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    x_labels : Optional[List[str]], optional
        Custom labels for x-axis (edges). If None, auto-generates "Edge 0", "Edge 1", etc.
    y_labels : Optional[List[str]], optional
        Custom labels for y-axis (depths). If None, auto-generates "Depth 0", "Depth 1", etc.
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_mask_heatmap
    >>> import numpy as np
    >>> mask = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])  # (depth=3, edges=3)
    >>> plot_mask_heatmap(mask, title="Final Pruned Mask")
    """
    if mask.size == 0:
        warnings.warn("Empty mask provided, skipping plot")
        return
    
    mask = np.asarray(mask, dtype=int)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D array, got shape {mask.shape}")
    
    ensure_dir(output_dir)
    
    depth, num_edges = mask.shape
    
    # Generate labels if not provided
    if x_labels is None:
        x_labels = [f"Edge {e}" for e in range(num_edges)]
    if y_labels is None:
        y_labels = [f"Depth {d}" for d in range(depth)]
    
    plt.figure(figsize=(max(10, num_edges * 1.2), max(6, depth * 0.8)))
    
    # Plot heatmap
    if HAS_SEABORN:
        sns.heatmap(
            mask,
            annot=True,
            fmt="d",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar_kws={"label": "Active (1) / Inactive (0)"},
            linewidths=0.8,
            linecolor="black",
            square=False,
        )
    else:
        # Fallback to matplotlib
        im = plt.imshow(mask, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, label="Active (1) / Inactive (0)")
        plt.xticks(range(num_edges), x_labels, rotation=0)
        plt.yticks(range(depth), y_labels)
        
        # Add annotations
        for i in range(depth):
            for j in range(num_edges):
                plt.text(j, i, int(mask[i, j]), ha="center", va="center",
                        color="black", fontweight="bold", fontsize=10)
    
    plt.xlabel("Edge Index", fontsize=13)
    plt.ylabel("Depth / Layer", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_mask_before_after(
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    fname: str = "mask_before_after",
    output_dir: str = "results/figures",
    title_before: str = "Mask Before Pruning",
    title_after: str = "Mask After Pruning",
    verbose: bool = True
) -> None:
    """
    Plot side-by-side comparison of masks before and after pruning.
    
    Parameters
    ----------
    mask_before : np.ndarray
        Binary mask before pruning, shape (depth, num_edges).
    mask_after : np.ndarray
        Binary mask after pruning, shape (depth, num_edges).
    fname : str, optional
        Filename (without extension) to save the plot (default: "mask_before_after").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    title_before : str, optional
        Title for before plot (default: "Mask Before Pruning").
    title_after : str, optional
        Title for after plot (default: "Mask After Pruning").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_mask_before_after
    >>> import numpy as np
    >>> mask_init = np.ones((3, 4), dtype=int)  # All active
    >>> mask_final = np.array([[1,0,1,0], [0,1,0,1], [1,1,0,0]])  # Pruned
    >>> plot_mask_before_after(mask_init, mask_final)
    """
    if mask_before.shape != mask_after.shape:
        raise ValueError(
            f"Masks must have same shape, got {mask_before.shape} and {mask_after.shape}"
        )
    
    ensure_dir(output_dir)
    
    depth, num_edges = mask_before.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, depth * 0.8)))
    
    # Plot before
    if HAS_SEABORN:
        sns.heatmap(mask_before, annot=True, fmt="d", cmap="RdYlGn", vmin=0, vmax=1,
                   xticklabels=[f"Edge {e}" for e in range(num_edges)],
                   yticklabels=[f"Depth {d}" for d in range(depth)],
                   cbar_kws={"label": "Active (1) / Inactive (0)"},
                   linewidths=0.8, linecolor="black", ax=ax1, square=False)
    else:
        im1 = ax1.imshow(mask_before, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax1.set_xticks(range(num_edges))
        ax1.set_xticklabels([f"Edge {e}" for e in range(num_edges)])
        ax1.set_yticks(range(depth))
        ax1.set_yticklabels([f"Depth {d}" for d in range(depth)])
        for i in range(depth):
            for j in range(num_edges):
                ax1.text(j, i, int(mask_before[i, j]), ha="center", va="center",
                        color="black", fontweight="bold", fontsize=10)
        plt.colorbar(im1, ax=ax1, label="Active (1) / Inactive (0)")
    
    ax1.set_xlabel("Edge Index", fontsize=12)
    ax1.set_ylabel("Depth / Layer", fontsize=12)
    ax1.set_title(title_before, fontsize=14, fontweight="bold")
    
    # Plot after
    if HAS_SEABORN:
        sns.heatmap(mask_after, annot=True, fmt="d", cmap="RdYlGn", vmin=0, vmax=1,
                   xticklabels=[f"Edge {e}" for e in range(num_edges)],
                   yticklabels=[f"Depth {d}" for d in range(depth)],
                   cbar_kws={"label": "Active (1) / Inactive (0)"},
                   linewidths=0.8, linecolor="black", ax=ax2, square=False)
    else:
        im2 = ax2.imshow(mask_after, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(num_edges))
        ax2.set_xticklabels([f"Edge {e}" for e in range(num_edges)])
        ax2.set_yticks(range(depth))
        ax2.set_yticklabels([f"Depth {d}" for d in range(depth)])
        for i in range(depth):
            for j in range(num_edges):
                ax2.text(j, i, int(mask_after[i, j]), ha="center", va="center",
                        color="black", fontweight="bold", fontsize=10)
        plt.colorbar(im2, ax=ax2, label="Active (1) / Inactive (0)")
    
    ax2.set_xlabel("Edge Index", fontsize=12)
    ax2.set_ylabel("Depth / Layer", fontsize=12)
    ax2.set_title(title_after, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def compare_methods_bar(
    method_results: Dict[str, Dict[str, float]],
    fname: str = "method_comparison",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
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
            "compressed": {"acc": 0.83, "cost": 6}
        }
    fname : str, optional
        Filename (without extension) to save the plot (default: "method_comparison").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import compare_methods_bar
    >>> results = {
    ...     "baseline": {"acc": 0.85, "cost": 12},
    ...     "compressed": {"acc": 0.83, "cost": 6}
    ... }
    >>> compare_methods_bar(results)
    """
    if not method_results:
        warnings.warn("Empty method_results provided, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    methods = list(method_results.keys())
    accuracies = [method_results[m].get("acc", 0.0) for m in methods]
    costs = [method_results[m].get("cost", 0) for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color="#1f77b4", alpha=0.8, 
                    edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Accuracy", fontsize=13)
    ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax1.tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight="bold")
    
    # Plot cost comparison
    bars2 = ax2.bar(methods, costs, color="#d62728", alpha=0.8,
                    edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Two-Qubit Gate Count", fontsize=13)
    ax2.set_title("Gate Cost Comparison", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax2.tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight="bold")
    
    plt.suptitle("Method Comparison: Baseline vs Compressed", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_density_fidelity(
    fidelities: Union[List[float], np.ndarray],
    title: str = "Density Matrix Fidelity",
    fname: str = "fidelity_curve",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot density matrix fidelity over training iterations.
    
    Parameters
    ----------
    fidelities : Union[List[float], np.ndarray]
        List or array of fidelity values (between 0 and 1) for each iteration.
    title : str, optional
        Plot title (default: "Density Matrix Fidelity").
    fname : str, optional
        Filename (without extension) to save the plot (default: "fidelity_curve").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_density_fidelity
    >>> fids = [0.95, 0.96, 0.97, 0.98]
    >>> plot_density_fidelity(fids, title="State Fidelity Over Training")
    """
    try:
        fid_arr = _to_numpy_array(fidelities)
    except ValueError:
        warnings.warn("Empty fidelity history provided, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 6))
    iterations = np.arange(len(fid_arr))
    plt.plot(iterations, fid_arr, linewidth=2.5, label="Fidelity", 
             color="#9467bd", marker="o", markersize=5)
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Density Matrix Fidelity", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_pred_vs_true(
    preds: np.ndarray,
    labels: np.ndarray,
    title: str = "Predicted vs True Labels",
    fname: str = "pred_vs_true",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot scatter plot of predicted probabilities vs true labels.
    
    Parameters
    ----------
    preds : np.ndarray
        Array of predicted probabilities (shape: (n_samples,)).
    labels : np.ndarray
        Array of true labels (shape: (n_samples,)). Should be 0 or 1.
    title : str, optional
        Plot title (default: "Predicted vs True Labels").
    fname : str, optional
        Filename (without extension) to save the plot (default: "pred_vs_true").
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_pred_vs_true
    >>> import numpy as np
    >>> preds = np.array([0.2, 0.8, 0.3, 0.9])
    >>> labels = np.array([0, 1, 0, 1])
    >>> plot_pred_vs_true(preds, labels)
    """
    preds_arr = np.asarray(preds)
    labels_arr = np.asarray(labels)
    
    if preds_arr.size == 0 or labels_arr.size == 0:
        warnings.warn("Empty predictions or labels provided, skipping plot")
        return
    
    _validate_array_lengths(preds_arr, labels_arr, name="preds and labels")
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 6))
    
    # Separate predictions by true label
    preds_class0 = preds_arr[labels_arr == 0]
    preds_class1 = preds_arr[labels_arr == 1]
    
    # Plot scatter with different colors for each class
    plt.scatter(preds_class0, np.zeros_like(preds_class0), alpha=0.7, s=120,
               label="Class 0 (True)", color="#1f77b4", marker="o", edgecolors="black", linewidths=0.5)
    plt.scatter(preds_class1, np.ones_like(preds_class1), alpha=0.7, s=120,
               label="Class 1 (True)", color="#d62728", marker="s", edgecolors="black", linewidths=0.5)
    
    # Add diagonal reference line (perfect predictions)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label="Perfect Prediction")
    
    plt.xlabel("Predicted Probability", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11, loc="best")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


# ============================================================================
# High-Level Convenience Functions for Training Scripts
# ============================================================================

def plot_all_curves_from_history(
    history: Dict[str, Union[List, np.ndarray]],
    prefix: str = "training",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot all available curves from a training history dictionary.
    
    This is a high-level convenience function that automatically generates
    all relevant plots from a history dictionary. Useful for training scripts.
    
    Parameters
    ----------
    history : Dict[str, Union[List, np.ndarray]]
        Training history dictionary with keys like:
        - "iteration" or "iterations": iteration numbers
        - "loss": total loss values
        - "ce_loss": cross-entropy loss (optional)
        - "acc" or "accuracy": accuracy values (optional)
        - "gate_count" or "two_qubit_count": gate counts (optional)
        - "mask_sparsity": mask sparsity values (optional)
    prefix : str, optional
        Prefix for output filenames (default: "training").
        Files will be named like "{prefix}_loss.png".
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmations (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_all_curves_from_history
    >>> history = {
    ...     "iteration": [0, 1, 2, 3],
    ...     "loss": [0.5, 0.4, 0.3, 0.25],
    ...     "acc": [0.6, 0.7, 0.8, 0.85],
    ...     "gate_count": [12, 10, 8, 6]
    ... }
    >>> plot_all_curves_from_history(history, prefix="baseline")
    """
    if not history:
        warnings.warn("Empty history provided, skipping plots")
        return
    
    # Get iteration array (use as x-axis if available)
    iterations = None
    if "iteration" in history:
        iterations = np.asarray(history["iteration"])
    elif "iterations" in history:
        iterations = np.asarray(history["iterations"])
    
    # Plot loss
    if "loss" in history:
        plot_loss_curve(
            history["loss"],
            title=f"{prefix.capitalize()} Loss",
            fname=f"{prefix}_loss",
            output_dir=output_dir,
            verbose=verbose
        )
    
    # Plot CE loss if available
    if "ce_loss" in history:
        plot_loss_curve(
            history["ce_loss"],
            title=f"{prefix.capitalize()} Cross-Entropy Loss",
            fname=f"{prefix}_ce_loss",
            output_dir=output_dir,
            verbose=verbose
        )
    
    # Plot accuracy
    if "acc" in history:
        plot_accuracy_curve(
            history["acc"],
            title=f"{prefix.capitalize()} Accuracy",
            fname=f"{prefix}_accuracy",
            output_dir=output_dir,
            verbose=verbose
        )
    elif "accuracy" in history:
        plot_accuracy_curve(
            history["accuracy"],
            title=f"{prefix.capitalize()} Accuracy",
            fname=f"{prefix}_accuracy",
            output_dir=output_dir,
            verbose=verbose
        )
    
    # Plot gate cost
    gate_key = None
    if "gate_count" in history:
        gate_key = "gate_count"
    elif "two_qubit_count" in history:
        gate_key = "two_qubit_count"
    
    if gate_key:
        plot_gate_cost_curve(
            history[gate_key],
            title=f"{prefix.capitalize()} Gate Cost",
            fname=f"{prefix}_gate_cost",
            output_dir=output_dir,
            verbose=verbose
        )
    
    # Plot mask sparsity
    if "mask_sparsity" in history:
        plot_mask_sparsity(
            history,
            save_path=os.path.join(output_dir, f"{prefix}_mask_sparsity.png"),
            title=f"{prefix.capitalize()} Mask Sparsity",
            verbose=verbose
        )


def plot_baseline_vs_compressed_histories(
    baseline_history: Dict[str, Union[List, np.ndarray]],
    compressed_history: Dict[str, Union[List, np.ndarray]],
    output_dir: str = "results/figures",
    fname: str = "baseline_vs_compressed",
    verbose: bool = True
) -> None:
    """
    Plot side-by-side comparison of baseline and compressed training histories.
    
    Creates a comprehensive comparison figure with subplots for loss, gate cost,
    and accuracy (if available). This is ideal for the main figure in a paper.
    
    Parameters
    ----------
    baseline_history : Dict[str, Union[List, np.ndarray]]
        Training history for baseline method.
    compressed_history : Dict[str, Union[List, np.ndarray]]
        Training history for compressed method.
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    fname : str, optional
        Filename (without extension) to save the plot (default: "baseline_vs_compressed").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_baseline_vs_compressed_histories
    >>> baseline = {"iteration": [0,1,2], "loss": [0.5,0.4,0.3], "gate_count": [12,12,12]}
    >>> compressed = {"iteration": [0,1,2], "loss": [0.5,0.4,0.35], "gate_count": [12,8,6]}
    >>> plot_baseline_vs_compressed_histories(baseline, compressed)
    """
    ensure_dir(output_dir)
    
    # Determine which metrics are available
    has_loss = "loss" in baseline_history and "loss" in compressed_history
    has_gate = ("gate_count" in baseline_history or "two_qubit_count" in baseline_history) and \
               ("gate_count" in compressed_history or "two_qubit_count" in compressed_history)
    has_acc = ("acc" in baseline_history or "accuracy" in baseline_history) and \
              ("acc" in compressed_history or "accuracy" in compressed_history)
    
    # Count number of subplots needed
    n_plots = sum([has_loss, has_gate, has_acc])
    if n_plots == 0:
        warnings.warn("No common metrics found in histories, skipping comparison plot")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot loss
    if has_loss:
        baseline_loss = np.asarray(baseline_history["loss"])
        compressed_loss = np.asarray(compressed_history["loss"])
        
        baseline_iter = np.arange(len(baseline_loss))
        compressed_iter = np.arange(len(compressed_loss))
        
        axes[plot_idx].plot(baseline_iter, baseline_loss, linewidth=2.5,
                           label="Baseline", color="#1f77b4", marker="o", markersize=4)
        axes[plot_idx].plot(compressed_iter, compressed_loss, linewidth=2.5,
                           label="Compressed", color="#d62728", marker="s", markersize=4)
        axes[plot_idx].set_xlabel("Iteration", fontsize=12)
        axes[plot_idx].set_ylabel("Loss", fontsize=12)
        axes[plot_idx].set_title("Loss Comparison", fontsize=13, fontweight="bold")
        axes[plot_idx].grid(True, alpha=0.3, linestyle="--")
        axes[plot_idx].legend(fontsize=11)
        plot_idx += 1
    
    # Plot gate cost
    if has_gate:
        baseline_gate_key = "gate_count" if "gate_count" in baseline_history else "two_qubit_count"
        compressed_gate_key = "gate_count" if "gate_count" in compressed_history else "two_qubit_count"
        
        baseline_gate = np.asarray(baseline_history[baseline_gate_key])
        compressed_gate = np.asarray(compressed_history[compressed_gate_key])
        
        baseline_iter = np.arange(len(baseline_gate))
        compressed_iter = np.arange(len(compressed_gate))
        
        axes[plot_idx].plot(baseline_iter, baseline_gate, linewidth=2.5,
                           label="Baseline", color="#1f77b4", marker="o", markersize=4)
        axes[plot_idx].plot(compressed_iter, compressed_gate, linewidth=2.5,
                           label="Compressed", color="#d62728", marker="s", markersize=4)
        axes[plot_idx].set_xlabel("Iteration", fontsize=12)
        axes[plot_idx].set_ylabel("Two-Qubit Gate Count", fontsize=12)
        axes[plot_idx].set_title("Gate Cost Comparison", fontsize=13, fontweight="bold")
        axes[plot_idx].grid(True, alpha=0.3, linestyle="--")
        axes[plot_idx].legend(fontsize=11)
        plot_idx += 1
    
    # Plot accuracy
    if has_acc:
        baseline_acc_key = "acc" if "acc" in baseline_history else "accuracy"
        compressed_acc_key = "acc" if "acc" in compressed_history else "accuracy"
        
        baseline_acc = np.asarray(baseline_history[baseline_acc_key])
        compressed_acc = np.asarray(compressed_history[compressed_acc_key])
        
        baseline_iter = np.arange(len(baseline_acc))
        compressed_iter = np.arange(len(compressed_acc))
        
        axes[plot_idx].plot(baseline_iter, baseline_acc, linewidth=2.5,
                          label="Baseline", color="#1f77b4", marker="o", markersize=4)
        axes[plot_idx].plot(compressed_iter, compressed_acc, linewidth=2.5,
                          label="Compressed", color="#d62728", marker="s", markersize=4)
        axes[plot_idx].set_xlabel("Iteration", fontsize=12)
        axes[plot_idx].set_ylabel("Accuracy", fontsize=12)
        axes[plot_idx].set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
        axes[plot_idx].grid(True, alpha=0.3, linestyle="--")
        axes[plot_idx].set_ylim(-0.05, 1.05)
        axes[plot_idx].legend(fontsize=11)
        plot_idx += 1
    
    plt.suptitle("Baseline vs Compressed Training Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()


def plot_pareto_from_runs(
    runs: List[Dict[str, float]],
    output_dir: str = "results/figures",
    fname: str = "pareto_runs",
    plot_loss: bool = False,
    verbose: bool = True
) -> None:
    """
    Plot Pareto front from multiple run results.
    
    Parameters
    ----------
    runs : List[Dict[str, float]]
        List of dictionaries, each containing:
        - "name": str, method name
        - "acc": float, accuracy
        - "cost": int or float, gate count
        - "loss": float (optional), if plot_loss=True
    output_dir : str, optional
        Output directory for saving plots (default: "results/figures").
    fname : str, optional
        Filename (without extension) to save the plot (default: "pareto_runs").
    plot_loss : bool, optional
        If True, also create a cost vs loss plot (default: False).
    verbose : bool, optional
        If True, print save confirmation (default: True).
    
    Examples
    --------
    >>> from src.plots import plot_pareto_from_runs
    >>> runs = [
    ...     {"name": "baseline", "acc": 0.85, "cost": 20, "loss": 0.3},
    ...     {"name": "compressed", "acc": 0.83, "cost": 8, "loss": 0.35}
    ... ]
    >>> plot_pareto_from_runs(runs)
    """
    if not runs:
        warnings.warn("Empty runs list provided, skipping plot")
        return
    
    ensure_dir(output_dir)
    
    costs = [r.get("cost", 0) for r in runs]
    accs = [r.get("acc", 0.0) for r in runs]
    names = [r.get("name", f"Method {i}") for i, r in enumerate(runs)]
    
    # Plot cost vs accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(costs, accs, alpha=0.7, s=200, c=range(len(runs)),
                        cmap="viridis", edgecolors="black", linewidths=1.0, zorder=3)
    
    # Add labels for each point
    for i, name in enumerate(names):
        ax.annotate(name, (costs[i], accs[i]), xytext=(5, 5),
                   textcoords="offset points", fontsize=10, fontweight="bold")
    
    # Find and plot Pareto front
    if len(costs) > 0:
        costs_arr = np.array(costs)
        acc_arr = np.array(accs)
        
        pareto_mask = np.ones(len(costs), dtype=bool)
        for i in range(len(costs)):
            for j in range(len(costs)):
                if i != j:
                    if costs_arr[j] <= costs_arr[i] and acc_arr[j] >= acc_arr[i]:
                        if costs_arr[j] < costs_arr[i] or acc_arr[j] > acc_arr[i]:
                            pareto_mask[i] = False
                            break
        
        pareto_costs = costs_arr[pareto_mask]
        pareto_accs = acc_arr[pareto_mask]
        
        if len(pareto_costs) > 0:
            sort_idx = np.argsort(pareto_costs)
            pareto_costs = pareto_costs[sort_idx]
            pareto_accs = pareto_accs[sort_idx]
            ax.plot(pareto_costs, pareto_accs, 'r--', linewidth=2.5,
                   alpha=0.8, label="Pareto Front", zorder=2)
    
    ax.set_xlabel("Two-Qubit Gate Count (Cost)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Pareto Front: Accuracy vs Gate Cost", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot to {save_path}", verbose=verbose)
    plt.close()
    
    # Optionally plot cost vs loss
    if plot_loss and all("loss" in r for r in runs):
        losses = [r["loss"] for r in runs]
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(costs, losses, alpha=0.7, s=200, c=range(len(runs)),
                  cmap="viridis", edgecolors="black", linewidths=1.0, zorder=3)
        
        for i, name in enumerate(names):
            ax.annotate(name, (costs[i], losses[i]), xytext=(5, 5),
                      textcoords="offset points", fontsize=10, fontweight="bold")
        
        ax.set_xlabel("Two-Qubit Gate Count (Cost)", fontsize=13)
        ax.set_ylabel("Loss", fontsize=13)
        ax.set_title("Pareto Front: Loss vs Gate Cost", fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        save_path_loss = os.path.join(output_dir, f"{fname}_loss.png")
        plt.savefig(save_path_loss, dpi=300, bbox_inches="tight")
        log(f"Saved plot to {save_path_loss}", verbose=verbose)
        plt.close()


# ============================================================================
# Compatibility Functions (for existing code)
# ============================================================================

def plot_loss_curves(
    history,
    save_path: Optional[str] = None,
    title: str = "Training Loss Curves",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot loss curves from pandas DataFrame (for compatibility with existing code).
    
    Parameters
    ----------
    history : pd.DataFrame or Dict
        Training history with columns: iteration, loss, ce_loss
    save_path : str, optional
        Path to save the figure. If None, uses default naming.
    title : str, optional
        Plot title (default: "Training Loss Curves").
    output_dir : str, optional
        Output directory (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    """
    ensure_dir(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Handle both DataFrame and dict
    if hasattr(history, "columns"):  # DataFrame
        iter_key = "iteration"
        loss_key = "loss"
        ce_loss_key = "ce_loss"
    else:  # Dict
        iter_key = "iteration" if "iteration" in history else "iterations"
        loss_key = "loss"
        ce_loss_key = "ce_loss"
    
    # Plot total loss
    if iter_key in history and loss_key in history:
        axes[0].plot(history[iter_key], history[loss_key], 
                    label="Total Loss", linewidth=2.5, color="#1f77b4")
        axes[0].set_xlabel("Iteration", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Total Loss (CE + λ * CNOT)", fontsize=13, fontweight="bold")
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(fontsize=11)
    
    # Plot CE loss
    if iter_key in history and ce_loss_key in history:
        axes[1].plot(history[iter_key], history[ce_loss_key],
                    label="CE Loss", linewidth=2.5, color="#ff7f0e")
        axes[1].set_xlabel("Iteration", fontsize=12)
        axes[1].set_ylabel("Cross-Entropy Loss", fontsize=12)
        axes[1].set_title("Classification Loss", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend(fontsize=11)
    
    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
        log(f"Saved plot to {save_path}", verbose=verbose)
    else:
        save_path = os.path.join(output_dir, "loss_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log(f"Saved plot to {save_path}", verbose=verbose)
    
    plt.close()


def plot_mask_sparsity(
    history,
    save_path: Optional[str] = None,
    title: str = "Mask Sparsity Over Training",
    output_dir: str = "results/figures",
    verbose: bool = True
) -> None:
    """
    Plot mask sparsity from pandas DataFrame (for compatibility with existing code).
    
    Parameters
    ----------
    history : pd.DataFrame or Dict
        Training history with columns: iteration, mask_sparsity
    save_path : str, optional
        Path to save the figure. If None, uses default naming.
    title : str, optional
        Plot title (default: "Mask Sparsity Over Training").
    output_dir : str, optional
        Output directory (default: "results/figures").
    verbose : bool, optional
        If True, print save confirmation (default: True).
    """
    # Handle both DataFrame and dict
    if hasattr(history, "columns"):  # DataFrame
        if "mask_sparsity" not in history.columns:
            warnings.warn("mask_sparsity column not found in history")
            return
        iter_key = "iteration"
        sparsity_key = "mask_sparsity"
    else:  # Dict
        if "mask_sparsity" not in history:
            warnings.warn("mask_sparsity key not found in history")
            return
        iter_key = "iteration" if "iteration" in history else "iterations"
        sparsity_key = "mask_sparsity"
    
    ensure_dir(output_dir)
    
    plt.figure(figsize=(8, 5))
    plt.plot(history[iter_key], history[sparsity_key],
            linewidth=2.5, color="#2ca02c", marker="o", markersize=4)
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Mask Sparsity (fraction of active entanglers)", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
        log(f"Saved plot to {save_path}", verbose=verbose)
    else:
        save_path = os.path.join(output_dir, "mask_sparsity.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log(f"Saved plot to {save_path}", verbose=verbose)
    
    plt.close()
