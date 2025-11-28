"""
mask_ablation_pipeline.py

Complete ablation analysis for quantum compilation project.

This script performs a comprehensive ablation study comparing:
- Learned masks (trained with different λ values)
- Random masks (matching sparsity of learned masks)
- Baseline (no mask, λ=0)

For each configuration, it collects:
- Post-transpile CNOT count
- Accuracy (or cross-entropy)
- Mask sparsity
- Comparison metrics vs baseline

Usage:
    python mask_ablation_pipeline.py --lambdas 0.0 0.1 0.3 0.5 --n_random 5
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import warnings

# Import existing functions from the codebase
from src.data import get_toy_dataset
from src.learner import forward_loss, compute_accuracy
from src.ansatz import (
    get_default_pairs,
    init_random_theta,
    init_full_mask,
    init_sparse_mask,
    build_ansatz,
    validate_theta_and_mask,
)
from src.transpile_utils import transpile_and_count_2q
from src.train_compressed import (
    compute_accuracy_from_predictions,
    compute_mask_sparsity,
    main as train_compressed_main,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_cnot_count(
    theta: np.ndarray,
    mask: np.ndarray,
    n_qubits: int,
    depth: int,
    pairs: List[Tuple[int, int]],
    backend: Optional[Any] = None,
) -> int:
    """
    Compute CNOT count after transpilation for given theta and mask.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter tensor.
    mask : np.ndarray
        Binary mask tensor.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    pairs : List[Tuple[int, int]]
        Qubit pairs for coupling map.
    backend : optional
        Qiskit backend for transpilation.
    
    Returns
    -------
    int
        CNOT count after transpilation.
    """
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    _, cnot_count = transpile_and_count_2q(ansatz, backend=backend)
    return cnot_count


def evaluate_model(
    theta: np.ndarray,
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    depth: int,
    pairs: List[Tuple[int, int]],
    backend: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate a model and return metrics.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter tensor.
    mask : np.ndarray
        Binary mask tensor.
    X : np.ndarray
        Data features.
    y : np.ndarray
        Data labels.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    pairs : List[Tuple[int, int]]
        Qubit pairs.
    backend : optional
        Qiskit backend.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with metrics: accuracy, ce_loss, cnot_count, mask_sparsity.
    """
    # Compute loss and predictions
    result = forward_loss(
        theta=theta,
        mask=mask,
        X=X,
        y=y,
        lam=0.0,  # Use λ=0 for evaluation (no regularization)
        pairs=pairs,
        backend=backend,
        n_qubits=n_qubits,
        depth=depth,
    )
    
    # Compute accuracy
    preds = result["preds"]
    accuracy = compute_accuracy_from_predictions(preds, y)
    
    # Compute CNOT count
    cnot_count = compute_cnot_count(theta, mask, n_qubits, depth, pairs, backend)
    
    # Compute mask sparsity
    sparsity = compute_mask_sparsity(mask)
    
    return {
        "accuracy": accuracy,
        "ce_loss": result["ce_loss"],
        "cnot_count": cnot_count,
        "mask_sparsity": sparsity,
    }


def generate_random_mask_matching_sparsity(
    target_sparsity: float,
    depth: int,
    n_edges: int,
    seed: int,
) -> np.ndarray:
    """
    Generate a random mask with the same sparsity as the target.
    
    Parameters
    ----------
    target_sparsity : float
        Target sparsity (fraction of active gates).
    depth : int
        Circuit depth.
    n_edges : int
        Number of edges.
    seed : int
        Random seed.
    
    Returns
    -------
    np.ndarray
        Random mask with matching sparsity.
    """
    return init_sparse_mask(
        depth=depth,
        n_edges=n_edges,
        sparsity=target_sparsity,
        seed=seed,
        structured=False,
    )


# ============================================================================
# Main Experiment Functions
# ============================================================================

def run_learned_mask_experiment(
    lam: float,
    n_qubits: int,
    depth: int,
    dataset_name: str,
    n_iterations: int,
    lr: float,
    prune_every: int,
    tolerance: float,
    seed: int,
    channel_strength: float,
    optimizer_type: str,
    output_dir: str,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run full training with learned mask for a given λ.
    
    Parameters
    ----------
    lam : float
        Regularization strength.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    dataset_name : str
        Dataset name.
    n_iterations : int
        Number of training iterations.
    lr : float
        Learning rate.
    prune_every : int
        Prune every N iterations.
    tolerance : float
        Pruning tolerance.
    seed : int
        Random seed.
    channel_strength : float
        Channel strength.
    optimizer_type : str
        Optimizer type.
    output_dir : str
        Output directory.
    backend : optional
        Qiskit backend.
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary with theta, mask, and metrics.
    """
    logger.info(f"Running learned mask experiment for λ={lam}")
    
    # Run training
    results = train_compressed_main(
        n_qubits=n_qubits,
        depth=depth,
        n_iterations=n_iterations,
        lr=lr,
        lam=lam,
        prune_every=prune_every,
        tolerance=tolerance,
        seed=seed,
        dataset_name=dataset_name,
        channel_strength=channel_strength,
        optimizer_type=optimizer_type,
        output_dir=output_dir,
    )
    
    # Load dataset for evaluation
    X, y = get_toy_dataset(name=dataset_name)
    pairs = get_default_pairs(n_qubits)
    
    # Evaluate final model
    metrics = evaluate_model(
        theta=results["theta"],
        mask=results["mask"],
        X=X,
        y=y,
        n_qubits=n_qubits,
        depth=depth,
        pairs=pairs,
        backend=backend,
    )
    
    return {
        "theta": results["theta"],
        "mask": results["mask"],
        "metrics": metrics,
        "lambda": lam,
    }


def run_random_mask_ablation(
    learned_mask: np.ndarray,
    learned_theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    depth: int,
    pairs: List[Tuple[int, int]],
    n_random: int,
    base_seed: int,
    backend: Optional[Any] = None,
) -> List[Dict[str, float]]:
    """
    Run random mask ablation for K random masks matching learned mask sparsity.
    
    Parameters
    ----------
    learned_mask : np.ndarray
        Learned mask to match sparsity.
    learned_theta : np.ndarray
        Learned parameters (used for random mask experiments).
    X : np.ndarray
        Data features.
    y : np.ndarray
        Data labels.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    pairs : List[Tuple[int, int]]
        Qubit pairs.
    n_random : int
        Number of random masks to generate.
    base_seed : int
        Base random seed.
    backend : optional
        Qiskit backend.
    
    Returns
    -------
    List[Dict[str, float]]
        List of metrics for each random mask.
    """
    # Compute target sparsity
    target_sparsity = compute_mask_sparsity(learned_mask)
    n_edges = len(pairs)
    
    logger.info(f"Running random mask ablation: {n_random} masks with sparsity={target_sparsity:.4f}")
    
    results = []
    for k in range(n_random):
        seed = base_seed + k
        np.random.seed(seed)
        
        # Generate random mask with matching sparsity
        random_mask = generate_random_mask_matching_sparsity(
            target_sparsity=target_sparsity,
            depth=depth,
            n_edges=n_edges,
            seed=seed,
        )
        
        # Evaluate with random mask (using learned theta)
        metrics = evaluate_model(
            theta=learned_theta,
            mask=random_mask,
            X=X,
            y=y,
            n_qubits=n_qubits,
            depth=depth,
            pairs=pairs,
            backend=backend,
        )
        
        metrics["seed"] = seed
        results.append(metrics)
    
    return results


def run_baseline_experiment(
    n_qubits: int,
    depth: int,
    dataset_name: str,
    n_iterations: int,
    lr: float,
    seed: int,
    channel_strength: float,
    optimizer_type: str,
    output_dir: str,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run baseline experiment (no mask, λ=0).
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    dataset_name : str
        Dataset name.
    n_iterations : int
        Number of training iterations.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    channel_strength : float
        Channel strength.
    optimizer_type : str
        Optimizer type.
    output_dir : str
        Output directory.
    backend : optional
        Qiskit backend.
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary with theta, mask, and metrics.
    """
    logger.info("Running baseline experiment (no mask, λ=0)")
    
    # Run training with λ=0 and full mask
    results = train_compressed_main(
        n_qubits=n_qubits,
        depth=depth,
        n_iterations=n_iterations,
        lr=lr,
        lam=0.0,
        prune_every=n_iterations + 1,  # Never prune
        tolerance=0.0,
        seed=seed,
        dataset_name=dataset_name,
        channel_strength=channel_strength,
        optimizer_type=optimizer_type,
        output_dir=output_dir,
    )
    
    # Load dataset for evaluation
    X, y = get_toy_dataset(name=dataset_name)
    pairs = get_default_pairs(n_qubits)
    
    # Evaluate final model
    metrics = evaluate_model(
        theta=results["theta"],
        mask=results["mask"],
        X=X,
        y=y,
        n_qubits=n_qubits,
        depth=depth,
        pairs=pairs,
        backend=backend,
    )
    
    return {
        "theta": results["theta"],
        "mask": results["mask"],
        "metrics": metrics,
        "lambda": 0.0,
    }


# ============================================================================
# Results Collection and Aggregation
# ============================================================================

def collect_all_results(
    lambda_values: List[float],
    n_random: int,
    n_qubits: int,
    depth: int,
    dataset_name: str,
    n_iterations: int,
    lr: float,
    prune_every: int,
    tolerance: float,
    base_seed: int,
    channel_strength: float,
    optimizer_type: str,
    output_dir: str,
    backend: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Collect all results from learned, random, and baseline experiments.
    
    Parameters
    ----------
    lambda_values : List[float]
        List of λ values to test.
    n_random : int
        Number of random masks per λ.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    dataset_name : str
        Dataset name.
    n_iterations : int
        Number of training iterations.
    lr : float
        Learning rate.
    prune_every : int
        Prune every N iterations.
    tolerance : float
        Pruning tolerance.
    base_seed : int
        Base random seed.
    channel_strength : float
        Channel strength.
    optimizer_type : str
        Optimizer type.
    output_dir : str
        Output directory.
    backend : optional
        Qiskit backend.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all results.
    """
    logger.info("=" * 80)
    logger.info("Starting mask ablation pipeline")
    logger.info("=" * 80)
    
    # Load dataset once
    X, y = get_toy_dataset(name=dataset_name)
    pairs = get_default_pairs(n_qubits)
    
    # Run baseline experiment
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Running baseline experiment")
    logger.info("=" * 80)
    
    baseline_results = run_baseline_experiment(
        n_qubits=n_qubits,
        depth=depth,
        dataset_name=dataset_name,
        n_iterations=n_iterations,
        lr=lr,
        seed=base_seed,
        channel_strength=channel_strength,
        optimizer_type=optimizer_type,
        output_dir=output_dir,
        backend=backend,
    )
    
    baseline_cnot = baseline_results["metrics"]["cnot_count"]
    baseline_accuracy = baseline_results["metrics"]["accuracy"]
    
    logger.info(f"Baseline CNOT count: {baseline_cnot}")
    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Collect all results
    all_results = []
    
    # Add baseline result
    all_results.append({
        "lambda": 0.0,
        "mask_type": "baseline",
        "cnot_count": baseline_cnot,
        "accuracy": baseline_accuracy,
        "accuracy_drop_vs_baseline": 0.0,
        "cnot_reduction_vs_baseline": 0.0,
        "mask_sparsity": 1.0,  # Full mask
        "seed": None,
    })
    
    # Run experiments for each λ
    for lam in tqdm(lambda_values, desc="Processing λ values"):
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 2: Processing λ={lam}")
        logger.info("=" * 80)
        
        # A. Learned mask experiment
        logger.info(f"\nA. Running learned mask experiment for λ={lam}")
        learned_results = run_learned_mask_experiment(
            lam=lam,
            n_qubits=n_qubits,
            depth=depth,
            dataset_name=dataset_name,
            n_iterations=n_iterations,
            lr=lr,
            prune_every=prune_every,
            tolerance=tolerance,
            seed=base_seed,
            channel_strength=channel_strength,
            optimizer_type=optimizer_type,
            output_dir=output_dir,
            backend=backend,
        )
        
        learned_metrics = learned_results["metrics"]
        learned_cnot = learned_metrics["cnot_count"]
        learned_accuracy = learned_metrics["accuracy"]
        learned_sparsity = learned_metrics["mask_sparsity"]
        
        # Add learned mask result
        all_results.append({
            "lambda": lam,
            "mask_type": "learned",
            "cnot_count": learned_cnot,
            "accuracy": learned_accuracy,
            "accuracy_drop_vs_baseline": baseline_accuracy - learned_accuracy,
            "cnot_reduction_vs_baseline": (baseline_cnot - learned_cnot) / baseline_cnot * 100.0,
            "mask_sparsity": learned_sparsity,
            "seed": None,
        })
        
        # B. Random mask ablation
        logger.info(f"\nB. Running random mask ablation for λ={lam} (K={n_random} masks)")
        random_results = run_random_mask_ablation(
            learned_mask=learned_results["mask"],
            learned_theta=learned_results["theta"],
            X=X,
            y=y,
            n_qubits=n_qubits,
            depth=depth,
            pairs=pairs,
            n_random=n_random,
            base_seed=base_seed + 1000,  # Offset seed for random masks
            backend=backend,
        )
        
        # Aggregate random mask results
        random_cnots = [r["cnot_count"] for r in random_results]
        random_accuracies = [r["accuracy"] for r in random_results]
        
        random_cnot_mean = np.mean(random_cnots)
        random_cnot_std = np.std(random_cnots)
        random_accuracy_mean = np.mean(random_accuracies)
        random_accuracy_std = np.std(random_accuracies)
        
        # Add mean random mask result
        all_results.append({
            "lambda": lam,
            "mask_type": "random",
            "cnot_count": random_cnot_mean,
            "accuracy": random_accuracy_mean,
            "accuracy_drop_vs_baseline": baseline_accuracy - random_accuracy_mean,
            "cnot_reduction_vs_baseline": (baseline_cnot - random_cnot_mean) / baseline_cnot * 100.0,
            "mask_sparsity": learned_sparsity,  # Same sparsity as learned
            "seed": None,  # Mean result
        })
        
        # Add individual random mask results
        for r in random_results:
            all_results.append({
                "lambda": lam,
                "mask_type": "random",
                "cnot_count": r["cnot_count"],
                "accuracy": r["accuracy"],
                "accuracy_drop_vs_baseline": baseline_accuracy - r["accuracy"],
                "cnot_reduction_vs_baseline": (baseline_cnot - r["cnot_count"]) / baseline_cnot * 100.0,
                "mask_sparsity": r["mask_sparsity"],
                "seed": r["seed"],
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    return df, baseline_cnot, baseline_accuracy


# ============================================================================
# Summary Tables
# ============================================================================

def generate_summary_tables(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate summary tables (CSV and LaTeX).
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_dir : Path
        Output directory.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Generating summary tables")
    logger.info("=" * 80)
    
    # Filter to mean results (exclude individual random seeds)
    df_mean = df[df["seed"].isna()].copy()
    
    # Create pivot table
    pivot = df_mean.pivot_table(
        index=["mask_type", "lambda"],
        values=["cnot_count", "accuracy", "cnot_reduction_vs_baseline", "accuracy_drop_vs_baseline"],
        aggfunc="first",
    )
    
    # Save CSV summary
    summary_path = output_dir / "mask_ablation_summary.csv"
    pivot.to_csv(summary_path)
    logger.info(f"Saved summary CSV to {summary_path}")
    
    # Generate LaTeX table
    latex_path = output_dir / "mask_ablation_table.tex"
    generate_latex_table(df, latex_path)
    logger.info(f"Saved LaTeX table to {latex_path}")


def generate_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate LaTeX table in academic paper format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_path : Path
        Output path for LaTeX file.
    """
    # Filter to mean results
    df_mean = df[df["seed"].isna()].copy()
    
    # Get unique lambda values
    lambda_values = sorted(df_mean[df_mean["lambda"] > 0]["lambda"].unique())
    
    # Compute statistics for random masks
    random_stats = {}
    for lam in lambda_values:
        random_df = df[(df["lambda"] == lam) & (df["mask_type"] == "random") & (df["seed"].notna())]
        if len(random_df) > 0:
            random_stats[lam] = {
                "cnot_mean": random_df["cnot_count"].mean(),
                "cnot_std": random_df["cnot_count"].std(),
                "accuracy_mean": random_df["accuracy"].mean(),
                "accuracy_std": random_df["accuracy"].std(),
            }
    
    # Build LaTeX table
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Mask Ablation Results}")
    lines.append("\\label{tab:mask_ablation}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Mask Type & $\\lambda$ & CNOT Count & Accuracy & CNOT Reduction \\\\")
    lines.append("\\midrule")
    
    # Baseline
    baseline = df_mean[df_mean["mask_type"] == "baseline"].iloc[0]
    lines.append(
        f"Baseline & 0.0 & {baseline['cnot_count']:.0f} & "
        f"{baseline['accuracy']:.4f} & 0.0\\% \\\\"
    )
    
    # Learned and random for each lambda
    for lam in lambda_values:
        learned = df_mean[(df_mean["lambda"] == lam) & (df_mean["mask_type"] == "learned")].iloc[0]
        lines.append(
            f"Learned & {lam:.3f} & {learned['cnot_count']:.0f} & "
            f"{learned['accuracy']:.4f} & {learned['cnot_reduction_vs_baseline']:.1f}\\% \\\\"
        )
        
        if lam in random_stats:
            stats = random_stats[lam]
            lines.append(
                f"Random & {lam:.3f} & "
                f"{stats['cnot_mean']:.0f} $\\pm$ {stats['cnot_std']:.0f} & "
                f"{stats['accuracy_mean']:.4f} $\\pm$ {stats['accuracy_std']:.4f} & "
                f"{(baseline['cnot_count'] - stats['cnot_mean']) / baseline['cnot_count'] * 100.0:.1f}\\% \\\\"
            )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Plots
# ============================================================================

def generate_plots(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate all plots for the paper.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_dir : Path
        Output directory.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Generating plots")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to mean results for main plots
    df_mean = df[df["seed"].isna()].copy()
    
    # 1. Accuracy vs CNOT Count scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Learned masks
    learned = df_mean[df_mean["mask_type"] == "learned"]
    ax.scatter(
        learned["cnot_count"],
        learned["accuracy"],
        s=200,
        marker="o",
        label="Learned",
        color="#1f77b4",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )
    
    # Random masks (mean)
    random = df_mean[df_mean["mask_type"] == "random"]
    ax.scatter(
        random["cnot_count"],
        random["accuracy"],
        s=200,
        marker="^",
        label="Random",
        color="#ff7f0e",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )
    
    # Baseline
    baseline = df_mean[df_mean["mask_type"] == "baseline"]
    ax.scatter(
        baseline["cnot_count"],
        baseline["accuracy"],
        s=300,
        marker="*",
        label="Baseline",
        color="#2ca02c",
        edgecolors="black",
        linewidths=2,
        zorder=4,
    )
    
    # Add labels for lambda values
    for idx, row in learned.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.3f}",
            (row["cnot_count"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    
    ax.set_xlabel("CNOT Count", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title("Accuracy vs CNOT Count", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = output_dir / "mask_ablation_accuracy_vs_cnot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")
    
    # 2. Bar plot: % CNOT reduction
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    lambda_values = sorted(learned["lambda"].unique())
    x = np.arange(len(lambda_values))
    width = 0.35
    
    learned_reductions = [learned[learned["lambda"] == lam]["cnot_reduction_vs_baseline"].values[0] for lam in lambda_values]
    random_reductions = [random[random["lambda"] == lam]["cnot_reduction_vs_baseline"].values[0] for lam in lambda_values]
    
    ax.bar(x - width/2, learned_reductions, width, label="Learned", color="#1f77b4", edgecolor="black", linewidth=1)
    ax.bar(x + width/2, random_reductions, width, label="Random", color="#ff7f0e", edgecolor="black", linewidth=1)
    
    ax.set_xlabel("$\\lambda$", fontsize=13, fontweight="bold")
    ax.set_ylabel("CNOT Reduction vs Baseline (%)", fontsize=13, fontweight="bold")
    ax.set_title("CNOT Reduction: Learned vs Random", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lam:.3f}" for lam in lambda_values])
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = output_dir / "mask_ablation_cnot_reduction.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")
    
    # 3. Line plot: Accuracy drop vs λ
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    learned_drops = [learned[learned["lambda"] == lam]["accuracy_drop_vs_baseline"].values[0] for lam in lambda_values]
    random_drops = [random[random["lambda"] == lam]["accuracy_drop_vs_baseline"].values[0] for lam in lambda_values]
    
    ax.plot(lambda_values, learned_drops, marker="o", linewidth=2.5, markersize=8, label="Learned", color="#1f77b4")
    ax.plot(lambda_values, random_drops, marker="^", linewidth=2.5, markersize=8, label="Random", color="#ff7f0e")
    
    ax.set_xlabel("$\\lambda$", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy Drop vs Baseline", fontsize=13, fontweight="bold")
    ax.set_title("Accuracy Drop vs $\\lambda$", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = output_dir / "mask_ablation_accuracy_drop.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run the complete ablation pipeline."""
    parser = argparse.ArgumentParser(
        description="Mask ablation analysis pipeline for quantum compilation"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5],
        help="List of λ values to test (default: [0.0, 0.1, 0.3, 0.5])",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=5,
        help="Number of random masks per λ (default: 5)",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=2,
        help="Number of qubits (default: 2)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Circuit depth (default: 3)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pothos_chater_small",
        choices=["pothos_chater_small", "pothos_chater_medium", "pothos_chater_large"],
        help="Dataset name (default: pothos_chater_small)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--prune_every",
        type=int,
        default=20,
        help="Prune every N iterations (default: 20)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Pruning tolerance (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--channel_strength",
        type=float,
        default=0.4,
        help="Evidence channel strength (default: 0.4)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="finite_diff",
        choices=["finite_diff", "adam"],
        help="Optimizer type (default: finite_diff)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    
    args = parser.parse_args()
    
    # Filter out λ=0 from lambda_values (baseline is separate)
    lambda_values = [lam for lam in args.lambdas if lam > 0]
    
    if len(lambda_values) == 0:
        logger.warning("No λ > 0 values provided. Only baseline will be run.")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments and collect results
    try:
        df, baseline_cnot, baseline_accuracy = collect_all_results(
            lambda_values=lambda_values,
            n_random=args.n_random,
            n_qubits=args.n_qubits,
            depth=args.depth,
            dataset_name=args.dataset,
            n_iterations=args.iterations,
            lr=args.lr,
            prune_every=args.prune_every,
            tolerance=args.tolerance,
            base_seed=args.seed,
            channel_strength=args.channel_strength,
            optimizer_type=args.optimizer,
            output_dir=str(output_dir),
            backend=None,
        )
        
        # Save raw results
        results_path = output_dir / "mask_ablation_results.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"\nSaved raw results to {results_path}")
        
        # Generate summary tables
        generate_summary_tables(df, output_dir)
        
        # Generate plots
        generate_plots(df, figures_dir)
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        
        # Print summary table
        df_mean = df[df["seed"].isna()].copy()
        print("\nSummary Table:")
        print(df_mean[["lambda", "mask_type", "cnot_count", "accuracy", "cnot_reduction_vs_baseline", "accuracy_drop_vs_baseline"]].to_string(index=False))
        
        # Print file paths
        print("\n" + "=" * 80)
        print("Output Files:")
        print("=" * 80)
        print(f"  Raw results: {results_path}")
        print(f"  Summary CSV: {output_dir / 'mask_ablation_summary.csv'}")
        print(f"  LaTeX table: {output_dir / 'mask_ablation_table.tex'}")
        print(f"  Figures: {figures_dir}")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"\n⚠️  Warning: {missing} missing values found in results")
            print("\nMissing values per column:")
            print(df.isnull().sum())
        
        logger.info("\n✓ Mask ablation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Error during ablation pipeline: {e}", exc_info=True)
        
        # Try to save partial results
        if 'df' in locals():
            partial_path = output_dir / "mask_ablation_results_partial.csv"
            df.to_csv(partial_path, index=False)
            logger.info(f"Saved partial results to {partial_path}")
        
        raise


if __name__ == "__main__":
    main()

