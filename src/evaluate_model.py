"""
evaluate_model.py

Evaluation script for trained quantum Bayesian learner models.

This script loads trained baseline and compressed models, compares them,
and generates comparison plots.

Usage:
    Run from repo root: python -m src.evaluate_model
    source venv/bin/activate && python -m src.evaluate_model
    python -m src.evaluate_model
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .data import get_toy_dataset
from .learner import forward_loss, compute_accuracy, predict_hard, predict_proba
from .ansatz import build_ansatz, get_default_pairs, init_full_mask
from .plots import (
    plot_loss_curve,
    plot_mask_heatmap,
    plot_pareto_front,
)


def load_training_logs(results_dir: Path = Path("results")) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load training logs for baseline and compressed models.
    
    Parameters
    ----------
    results_dir : Path, optional
        Results directory (default: Path("results")).
    
    Returns
    -------
    Dict[str, Optional[pd.DataFrame]]
        Dictionary with "baseline" and "compressed" keys, containing
        DataFrames or None if file not found.
    """
    logs = {}
    
    # Load baseline log
    baseline_log_path = results_dir / "logs" / "baseline_log.csv"
    if baseline_log_path.exists():
        logs["baseline"] = pd.read_csv(baseline_log_path)
        print(f"Loaded baseline log: {len(logs['baseline'])} rows")
    else:
        logs["baseline"] = None
        print(f"Warning: Baseline log not found at {baseline_log_path}")
    
    # Load compressed log
    compressed_log_path = results_dir / "logs" / "compressed_log.csv"
    if compressed_log_path.exists():
        logs["compressed"] = pd.read_csv(compressed_log_path)
        print(f"Loaded compressed log: {len(logs['compressed'])} rows")
    else:
        logs["compressed"] = None
        print(f"Warning: Compressed log not found at {compressed_log_path}")
    
    return logs


def load_compressed_model(results_dir: Path = Path("results")) -> Optional[Dict[str, np.ndarray]]:
    """
    Load compressed model parameters (theta + mask).
    
    Parameters
    ----------
    results_dir : Path, optional
        Results directory (default: Path("results")).
    
    Returns
    -------
    Optional[Dict[str, np.ndarray]]
        Dictionary with "theta" and "mask" keys, or None if file not found.
    """
    compressed_path = results_dir / "compressed_final.npz"
    
    if compressed_path.exists():
        data = np.load(compressed_path)
        model = {
            "theta": data["theta"],
            "mask": data["mask"],
        }
        print(f"Loaded compressed model from {compressed_path}")
        print(f"  Theta shape: {model['theta'].shape}")
        print(f"  Mask shape: {model['mask'].shape}")
        return model
    else:
        print(f"Warning: Compressed model not found at {compressed_path}")
        return None


def compute_metrics_from_logs(logs: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, Dict[str, float]]:
    """
    Compute final metrics from training logs.
    
    Parameters
    ----------
    logs : Dict[str, Optional[pd.DataFrame]]
        Dictionary of training logs.
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with metrics for each model type.
    """
    metrics = {}
    
    for model_type in ["baseline", "compressed"]:
        df = logs.get(model_type)
        if df is not None and len(df) > 0:
            # Get last row (final metrics)
            last_row = df.iloc[-1]
            metrics[model_type] = {
                "final_loss": float(last_row.get("loss", 0.0)),
                "final_ce_loss": float(last_row.get("ce_loss", 0.0)),
                "two_qubit_count": int(last_row.get("two_qubit_count", 0)),
            }
            
            # Get best (minimum) loss if available
            if "loss" in df.columns:
                metrics[model_type]["best_loss"] = float(df["loss"].min())
            else:
                metrics[model_type]["best_loss"] = metrics[model_type]["final_loss"]
        else:
            metrics[model_type] = None
    
    return metrics


def compute_accuracy_from_model(
    theta: np.ndarray,
    mask: np.ndarray,
    n_qubits: int,
    depth: int,
    dataset_name: str = "pothos_chater_small",
    channel_strength: float = 0.4,
) -> float:
    """
    Compute accuracy by re-running predictions on the dataset.
    
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
    dataset_name : str, optional
        Dataset name (default: "pothos_chater_small").
    channel_strength : float, optional
        Maximum channel strength (default: 0.4).
    
    Returns
    -------
    float
        Classification accuracy.
    """
    from .learner import (
        init_belief,
        apply_unitary,
        apply_evidence_channel,
        predict_hard,
    )
    
    # Load dataset
    X, y = get_toy_dataset(dataset_name)
    
    # Build ansatz
    pairs = get_default_pairs(n_qubits)
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    
    # Get predictions
    preds = []
    for xi in X:
        rho = init_belief(n_qubits)
        rho_U = apply_unitary(rho, ansatz)
        rho_post = apply_evidence_channel(
            rho_U, xi, strength=channel_strength, qargs=[0]
        )
        hard_pred = predict_hard(rho_post)
        preds.append(hard_pred)
    
    preds = np.array(preds)
    accuracy = compute_accuracy(preds, y)
    
    return float(accuracy)


def generate_comparison_plots(
    logs: Dict[str, Optional[pd.DataFrame]],
    compressed_model: Optional[Dict[str, np.ndarray]],
    metrics: Dict[str, Optional[Dict[str, float]]],
    output_dir: Path = Path("results/figures"),
) -> None:
    """
    Generate comparison plots for baseline vs compressed models.
    
    Parameters
    ----------
    logs : Dict[str, Optional[pd.DataFrame]]
        Training logs for both models.
    compressed_model : Optional[Dict[str, np.ndarray]]
        Compressed model parameters.
    metrics : Dict[str, Optional[Dict[str, float]]]
        Computed metrics for both models.
    output_dir : Path, optional
        Output directory for plots (default: Path("results/figures")).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves comparison
    if logs["baseline"] is not None and logs["compressed"] is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot total loss
        baseline_df = logs["baseline"]
        compressed_df = logs["compressed"]
        
        axes[0].plot(
            baseline_df["iteration"],
            baseline_df["loss"],
            label="Baseline",
            linewidth=2.5,
            color="#1f77b4",
            marker="o",
            markersize=4,
        )
        axes[0].plot(
            compressed_df["iteration"],
            compressed_df["loss"],
            label="Compressed",
            linewidth=2.5,
            color="#d62728",
            marker="s",
            markersize=4,
        )
        axes[0].set_xlabel("Iteration", fontsize=12)
        axes[0].set_ylabel("Total Loss", fontsize=12)
        axes[0].set_title("Total Loss Comparison", fontsize=13, fontweight="bold")
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(fontsize=11)
        
        # Plot CE loss
        axes[1].plot(
            baseline_df["iteration"],
            baseline_df["ce_loss"],
            label="Baseline",
            linewidth=2.5,
            color="#1f77b4",
            marker="o",
            markersize=4,
        )
        axes[1].plot(
            compressed_df["iteration"],
            compressed_df["ce_loss"],
            label="Compressed",
            linewidth=2.5,
            color="#d62728",
            marker="s",
            markersize=4,
        )
        axes[1].set_xlabel("Iteration", fontsize=12)
        axes[1].set_ylabel("CE Loss", fontsize=12)
        axes[1].set_title("CE Loss Comparison", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend(fontsize=11)
        
        plt.suptitle("Baseline vs Compressed Training Comparison", fontsize=15, fontweight="bold")
        plt.tight_layout()
        
        loss_comparison_path = output_dir / "eval_loss_comparison.png"
        plt.savefig(loss_comparison_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved loss comparison plot to {loss_comparison_path}")
    
    # Plot mask sparsity for compressed (if available)
    if logs["compressed"] is not None and "mask_sparsity" in logs["compressed"].columns:
        plot_loss_curve(
            logs["compressed"]["mask_sparsity"],
            title="Compressed Model: Mask Sparsity Over Training",
            fname="eval_compressed_mask_sparsity",
            output_dir=str(output_dir),
            verbose=True,
        )
    
    # Plot mask heatmap for compressed (if model available)
    if compressed_model is not None:
        plot_mask_heatmap(
            compressed_model["mask"],
            title="Compressed Model: Final Mask Structure",
            fname="eval_compressed_mask_heatmap",
            output_dir=str(output_dir),
            verbose=True,
        )
    
    # Plot Pareto front (2q cost vs accuracy or CE loss)
    if metrics["baseline"] is not None and metrics["compressed"] is not None:
        baseline_metrics = metrics["baseline"]
        compressed_metrics = metrics["compressed"]
        
        costs = [
            baseline_metrics["two_qubit_count"],
            compressed_metrics["two_qubit_count"],
        ]
        
        # Try to use accuracy if available, otherwise use 1 - CE_loss as proxy
        # (since Pareto expects higher is better, we'll use accuracy or inverse CE loss)
        import matplotlib.pyplot as plt
        
        # Check if we have accuracy data (would need to be passed in)
        # For now, create a simple comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot baseline point
        ax.scatter(
            baseline_metrics["two_qubit_count"],
            baseline_metrics["final_ce_loss"],
            s=200,
            color="#1f77b4",
            marker="o",
            label="Baseline",
            edgecolors="black",
            linewidths=1.5,
            zorder=3,
        )
        
        # Plot compressed point
        ax.scatter(
            compressed_metrics["two_qubit_count"],
            compressed_metrics["final_ce_loss"],
            s=200,
            color="#d62728",
            marker="s",
            label="Compressed",
            edgecolors="black",
            linewidths=1.5,
            zorder=3,
        )
        
        # Add labels
        ax.annotate(
            "Baseline",
            (baseline_metrics["two_qubit_count"], baseline_metrics["final_ce_loss"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
        ax.annotate(
            "Compressed",
            (compressed_metrics["two_qubit_count"], compressed_metrics["final_ce_loss"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
        
        ax.set_xlabel("Two-Qubit Gate Count (Cost)", fontsize=13)
        ax.set_ylabel("CE Loss", fontsize=13)
        ax.set_title("Baseline vs Compressed: CE Loss vs 2Q Gate Cost", fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=11)
        plt.tight_layout()
        
        pareto_path = output_dir / "eval_pareto_ce_loss_vs_cost.png"
        plt.savefig(pareto_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Pareto comparison plot to {pareto_path}")


def main() -> None:
    """
    Main evaluation function.
    
    Loads baseline and compressed results, computes metrics, and generates plots.
    """
    results_dir = Path("results")
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Model Evaluation: Baseline vs Compressed")
    print("=" * 60)
    print()
    
    # Load training logs
    print("Loading training logs...")
    logs = load_training_logs(results_dir)
    print()
    
    # Load compressed model
    print("Loading compressed model...")
    compressed_model = load_compressed_model(results_dir)
    print()
    
    # Compute metrics from logs
    print("Computing metrics from logs...")
    metrics = compute_metrics_from_logs(logs)
    print()
    
    # Optionally compute accuracy by re-running predictions
    print("Computing accuracy by re-running predictions...")
    accuracies = {}
    
    # Baseline accuracy (if we can load the model)
    baseline_theta_path = results_dir / "baseline_final_theta.npy"
    if baseline_theta_path.exists():
        try:
            baseline_theta = np.load(baseline_theta_path)
            # Infer n_qubits and depth from theta shape
            depth, max_dim, _ = baseline_theta.shape
            n_qubits = max_dim  # This is a guess, but should work for small cases
            pairs = get_default_pairs(n_qubits)
            mask = init_full_mask(depth, len(pairs))
            
            accuracies["baseline"] = compute_accuracy_from_model(
                baseline_theta, mask, n_qubits, depth
            )
            print(f"  Baseline accuracy: {accuracies['baseline']:.4f}")
        except Exception as e:
            print(f"  Warning: Could not compute baseline accuracy: {e}")
            accuracies["baseline"] = None
    else:
        print("  Baseline model not found, skipping accuracy computation")
        accuracies["baseline"] = None
    
    # Compressed accuracy
    if compressed_model is not None:
        try:
            theta = compressed_model["theta"]
            mask = compressed_model["mask"]
            depth, max_dim, _ = theta.shape
            n_qubits = max_dim
            
            accuracies["compressed"] = compute_accuracy_from_model(
                theta, mask, n_qubits, depth
            )
            print(f"  Compressed accuracy: {accuracies['compressed']:.4f}")
        except Exception as e:
            print(f"  Warning: Could not compute compressed accuracy: {e}")
            accuracies["compressed"] = None
    else:
        accuracies["compressed"] = None
    
    print()
    
    # Generate plots
    print("Generating comparison plots...")
    generate_comparison_plots(logs, compressed_model, metrics, output_dir)
    print()
    
    # Print summary
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    # Baseline summary
    if metrics["baseline"] is not None:
        baseline_metrics = metrics["baseline"]
        print("Baseline:")
        print(f"  Final loss: {baseline_metrics['final_loss']:.6f}")
        print(f"  Final CE loss: {baseline_metrics['final_ce_loss']:.6f}")
        print(f"  2Q gate count: {baseline_metrics['two_qubit_count']}")
        if accuracies["baseline"] is not None:
            print(f"  Accuracy: {accuracies['baseline']:.4f}")
    else:
        print("Baseline: No data available")
    
    print()
    
    # Compressed summary
    if metrics["compressed"] is not None:
        compressed_metrics = metrics["compressed"]
        print("Compressed:")
        print(f"  Final loss: {compressed_metrics['final_loss']:.6f}")
        print(f"  Final CE loss: {compressed_metrics['final_ce_loss']:.6f}")
        print(f"  2Q gate count: {compressed_metrics['two_qubit_count']}")
        if compressed_model is not None:
            mask = compressed_model["mask"]
            sparsity = float(np.sum(mask == 1) / mask.size)
            print(f"  Mask sparsity: {sparsity:.4f}")
        if accuracies["compressed"] is not None:
            print(f"  Accuracy: {accuracies['compressed']:.4f}")
    else:
        print("Compressed: No data available")
    
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
