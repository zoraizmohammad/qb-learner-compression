"""
evaluate_model.py

Evaluation script for trained quantum Bayesian learner models.

This script loads trained baseline or compressed models and evaluates them
on a dataset, generating plots and summary statistics.

Usage:
    python -m src.evaluate_model --model_type baseline --dataset pothos_chater_small --n_qubits 2 --depth 3
    python -m src.evaluate_model --model_type compressed --dataset pothos_chater_small --n_qubits 2 --depth 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix

from .data import get_toy_dataset
from .learner import (
    QuantumBayesianLearner,
    init_belief,
    apply_unitary,
    apply_evidence_channel,
    predict_proba,
    predict_hard,
    compute_accuracy,
    forward_loss,
)
from .ansatz import build_ansatz, get_default_pairs, init_full_mask
from .transpile_utils import transpile_and_count_2q
from .plots import plot_pred_vs_true, plot_mask_heatmap
from .logging_utils import write_json, timestamp_str


# ============================================================================
# Model Loading
# ============================================================================

def load_baseline_model(
    results_dir: Path,
    n_qubits: int,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load baseline model parameters.
    
    Parameters
    ----------
    results_dir : Path
        Results directory containing saved models.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (theta, mask) where mask is all-ones.
    
    Raises
    ------
    FileNotFoundError
        If required files are not found.
    """
    # Try to load from legacy location first
    theta_path = results_dir / "baseline_final_theta.npy"
    
    if not theta_path.exists():
        # Try to load from run directory (most recent run)
        run_dirs = sorted((results_dir / "runs").glob("baseline_*"), reverse=True)
        if run_dirs:
            params_path = run_dirs[0] / "params_final.npz"
            if params_path.exists():
                data = np.load(params_path)
                theta = data["theta"]
            else:
                raise FileNotFoundError(
                    f"Could not find baseline model. Tried:\n"
                    f"  - {theta_path}\n"
                    f"  - {params_path}"
                )
        else:
            raise FileNotFoundError(
                f"Could not find baseline model at {theta_path} and no run directories found."
            )
    else:
        theta = np.load(theta_path)
    
    # Create all-ones mask for baseline
    pairs = get_default_pairs(n_qubits)
    n_edges = len(pairs)
    mask = init_full_mask(depth, n_edges)
    
    return theta, mask


def load_compressed_model(
    results_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load compressed model parameters.
    
    Parameters
    ----------
    results_dir : Path
        Results directory containing saved models.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (theta, mask) from saved npz file.
    
    Raises
    ------
    FileNotFoundError
        If required files are not found.
    """
    # Try to load from legacy location first
    params_path = results_dir / "compressed_final.npz"
    
    if not params_path.exists():
        # Try to load from run directory (most recent run)
        run_dirs = sorted((results_dir / "runs").glob("compressed_*"), reverse=True)
        if run_dirs:
            params_path = run_dirs[0] / "params_final.npz"
            if not params_path.exists():
                raise FileNotFoundError(
                    f"Could not find compressed model. Tried:\n"
                    f"  - {results_dir / 'compressed_final.npz'}\n"
                    f"  - {params_path}"
                )
        else:
            raise FileNotFoundError(
                f"Could not find compressed model at {params_path} and no run directories found."
            )
    
    data = np.load(params_path)
    theta = data["theta"]
    mask = data["mask"]
    
    return theta, mask


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_ce_loss(preds: np.ndarray, y_true: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute cross-entropy loss.
    
    Parameters
    ----------
    preds : np.ndarray
        Predicted probabilities (shape: (n_samples,)).
    y_true : np.ndarray
        True labels (shape: (n_samples,)).
    eps : float, optional
        Small epsilon to avoid log(0) (default: 1e-10).
    
    Returns
    -------
    float
        Mean cross-entropy loss.
    """
    preds_clipped = np.clip(preds, eps, 1.0 - eps)
    ce = -(y_true * np.log(preds_clipped) + (1 - y_true) * np.log(1 - preds_clipped))
    return float(np.mean(ce))


def evaluate_model(
    theta: np.ndarray,
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    depth: int,
    channel_strength: float = 0.4,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter tensor.
    mask : np.ndarray
        Binary mask tensor.
    X : np.ndarray
        Input features, shape (n_samples, n_features).
    y : np.ndarray
        True labels, shape (n_samples,).
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    channel_strength : float, optional
        Maximum channel strength (default: 0.4).
    backend : optional
        Qiskit backend for transpilation (default: None).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - accuracy: float
        - ce_loss: float
        - preds: np.ndarray
        - hard_preds: np.ndarray
        - two_qubit_count: int
        - active_entanglers: int
    """
    pairs = get_default_pairs(n_qubits)
    
    # Build ansatz and get gate count
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    transpiled_result = transpile_and_count_2q(ansatz, backend=backend, return_dict=True)
    two_qubit_count = transpiled_result["n_2q"]
    
    # Count active entanglers
    active_entanglers = int(np.sum(mask == 1))
    
    # Get predictions for all samples
    preds = []
    hard_preds = []
    
    for xi in X:
        # Start with maximally mixed state
        rho = init_belief(n_qubits)
        
        # Apply ansatz
        rho_U = apply_unitary(rho, ansatz)
        
        # Apply evidence channel
        rho_post = apply_evidence_channel(
            rho_U, xi, strength=channel_strength, qargs=[0]
        )
        
        # Make predictions
        prob = predict_proba(rho_post)
        hard_pred = predict_hard(rho_post)
        
        preds.append(prob)
        hard_preds.append(hard_pred)
    
    preds = np.array(preds)
    hard_preds = np.array(hard_preds)
    
    # Compute metrics
    accuracy = compute_accuracy(hard_preds, y)
    ce_loss = compute_ce_loss(preds, y)
    
    return {
        "accuracy": float(accuracy),
        "ce_loss": float(ce_loss),
        "preds": preds,
        "hard_preds": hard_preds,
        "two_qubit_count": int(two_qubit_count),
        "active_entanglers": int(active_entanglers),
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def generate_evaluation_plots(
    results: Dict[str, Any],
    y_true: np.ndarray,
    mask: Optional[np.ndarray],
    model_type: str,
    output_dir: Path,
    n_qubits: int,
    depth: int,
) -> None:
    """
    Generate evaluation plots.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Evaluation results dictionary.
    y_true : np.ndarray
        True labels for plotting.
    mask : Optional[np.ndarray]
        Mask array (for compressed models).
    model_type : str
        Model type: "baseline" or "compressed".
    output_dir : Path
        Output directory for plots.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicted vs True plot
    plot_pred_vs_true(
        preds=results["preds"],
        labels=y_true,
        title=f"Predicted vs True Labels ({model_type})",
        fname=f"eval_{model_type}_pred_vs_true",
        output_dir=str(output_dir),
        verbose=True,
    )
    
    # Mask heatmap (only for compressed)
    if model_type == "compressed" and mask is not None:
        plot_mask_heatmap(
            mask=mask,
            title=f"Final Mask Structure ({model_type})",
            fname=f"eval_{model_type}_mask",
            output_dir=str(output_dir),
            verbose=True,
        )
    
    # Gate cost visualization
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(
        ["2-qubit gates"],
        [results["two_qubit_count"]],
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    ax.set_ylabel("Count")
    ax.set_title(f"Two-Qubit Gate Count ({model_type})")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value label
    ax.text(
        0,
        results["two_qubit_count"],
        str(results["two_qubit_count"]),
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    
    plt.tight_layout()
    gate_cost_path = output_dir / f"eval_{model_type}_gate_cost.png"
    plt.savefig(gate_cost_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gate cost plot to {gate_cost_path}")


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained quantum Bayesian learner models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "compressed"],
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pothos_chater_small",
        help="Dataset name (default: pothos_chater_small)",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        required=True,
        help="Number of qubits",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Circuit depth",
    )
    parser.add_argument(
        "--channel_strength",
        type=float,
        default=0.4,
        help="Maximum channel strength (default: 0.4)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: results/figures)",
    )
    
    args = parser.parse_args()
    
    # Set up paths
    results_dir = Path(args.results_dir)
    if args.output_dir is None:
        output_dir = results_dir / "figures"
    else:
        output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"n_qubits: {args.n_qubits}")
    print(f"depth: {args.depth}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    X, y = get_toy_dataset(name=args.dataset)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print()
    
    # Load model
    print(f"Loading {args.model_type} model...")
    try:
        if args.model_type == "baseline":
            theta, mask = load_baseline_model(results_dir, args.n_qubits, args.depth)
        else:
            theta, mask = load_compressed_model(results_dir)
        print(f"Theta shape: {theta.shape}")
        print(f"Mask shape: {mask.shape}")
        print()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(
        theta=theta,
        mask=mask,
        X=X,
        y=y,
        n_qubits=args.n_qubits,
        depth=args.depth,
        channel_strength=args.channel_strength,
        backend=None,
    )
    
    # Print summary
    print("=" * 60)
    print("Final Evaluation Summary")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Mean CE Loss: {results['ce_loss']:.6f}")
    print(f"2Q Gate Cost After Transpile: {results['two_qubit_count']}")
    if args.model_type == "compressed":
        print(f"Active Entanglers: {results['active_entanglers']}")
        print(f"Mask Sparsity: {1.0 - results['active_entanglers'] / mask.size:.2%}")
    print()
    
    # Generate plots
    print("Generating plots...")
    generate_evaluation_plots(
        results=results,
        y_true=y,
        mask=mask if args.model_type == "compressed" else None,
        model_type=args.model_type,
        output_dir=output_dir,
        n_qubits=args.n_qubits,
        depth=args.depth,
    )
    print()
    
    # Save summary JSON
    summary = {
        "accuracy": results["accuracy"],
        "ce_loss": results["ce_loss"],
        "two_qubit_count": results["two_qubit_count"],
        "active_entanglers": results.get("active_entanglers", None),
        "model_type": args.model_type,
        "dataset": args.dataset,
        "n_qubits": args.n_qubits,
        "depth": args.depth,
        "channel_strength": args.channel_strength,
    }
    
    summary_path = results_dir / f"eval_{args.model_type}_summary.json"
    write_json(summary_path, summary, verbose=True)
    
    print("=" * 60)
    print("Evaluation complete!")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

