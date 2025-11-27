"""
train_compressed.py

Compressed training script with structural pruning for the quantum Bayesian learner.

This script trains the learner with greedy pruning of entangling gates. It starts
with all entanglers active, optimizes theta continuously, and periodically prunes
entanglers that don't significantly impact the loss.

Usage:
    Run from repo root: python -m src.train_compressed
    
    Quick test (40 iterations):
    python -m src.train_compressed --iterations 40
    
    Full training with custom parameters:
    python -m src.train_compressed --iterations 100 --n_qubits 2 --depth 3 --prune_every 20 --tolerance 0.01
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from .data import get_toy_dataset
from .learner import forward_loss, compute_accuracy
from .ansatz import (
    get_default_pairs, build_parameter_shapes, init_random_theta,
    init_full_mask, validate_theta_and_mask
)
from .plots import (
    plot_all_curves_from_history, plot_baseline_vs_compressed_histories,
    plot_pareto_from_runs, plot_mask_heatmap, plot_mask_before_after,
    plot_pred_vs_true
)
from .logging_utils import (
    timestamp_str, save_training_config, save_final_metrics,
    save_training_history, save_parameters, save_loss_history,
    save_mask_history
)


# ============================================================================
# Optimizer Classes
# ============================================================================

class AdamOptimizer:
    """
    Adam optimizer for parameter updates.
    
    Implements the Adam (Adaptive Moment Estimation) algorithm for
    gradient-based optimization.
    """
    
    def __init__(self, shape: tuple, lr: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Parameters
        ----------
        shape : tuple
            Shape of parameters to optimize.
        lr : float, optional
            Learning rate (default: 0.01).
        beta1 : float, optional
            Exponential decay rate for first moment (default: 0.9).
        beta2 : float, optional
            Exponential decay rate for second moment (default: 0.999).
        eps : float, optional
            Small constant for numerical stability (default: 1e-8).
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape, dtype=np.float64)  # First moment
        self.v = np.zeros(shape, dtype=np.float64)  # Second moment
        self.t = 0  # Time step
    
    def step(self, grad: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.
        
        Parameters
        ----------
        grad : np.ndarray
            Gradient of the loss function.
        
        Returns
        -------
        np.ndarray
            Parameter update (to be subtracted from current parameters).
        """
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Compute parameter update
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return update


# ============================================================================
# Utility Functions
# ============================================================================

def finite_diff_gradient(
    loss_fn,
    theta: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Compute gradient using finite differences.
    
    Parameters
    ----------
    loss_fn : callable
        Function that returns loss given theta.
    theta : np.ndarray
        Current parameters.
    h : float
        Step size for finite differences.
    
    Returns
    -------
    np.ndarray
        Gradient estimate.
    """
    grad = np.zeros_like(theta)
    loss_base = loss_fn(theta)
    
    # Flatten for iteration
    theta_flat = theta.flatten()
    grad_flat = grad.flatten()
    
    for i in range(len(theta_flat)):
        theta_perturbed = theta_flat.copy()
        theta_perturbed[i] += h
        theta_pert = theta_perturbed.reshape(theta.shape)
        
        loss_pert = loss_fn(theta_pert)
        grad_flat[i] = (loss_pert - loss_base) / h
    
    return grad_flat.reshape(theta.shape)


def compute_ce_loss_only(
    theta: np.ndarray,
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    pairs: list,
    n_qubits: int,
    depth: int,
    channel_strength: float = 0.4,
) -> float:
    """
    Compute only the cross-entropy loss (without regularization).
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter tensor.
    mask : np.ndarray
        Binary mask tensor.
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    pairs : list
        Qubit pairs for coupling map.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    channel_strength : float
        Maximum strength for evidence channels.
    
    Returns
    -------
    float
        Cross-entropy loss only.
    """
    loss_dict = forward_loss(
        theta=theta,
        mask=mask,
        X=X,
        y=y,
        lam=0.0,  # Set lambda to 0 to get only CE loss
        pairs=pairs,
        backend=None,
        n_qubits=n_qubits,
        depth=depth,
        channel_strength=channel_strength,
    )
    return loss_dict["ce_loss"]


def compute_accuracy_from_predictions(
    preds: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute classification accuracy from predictions.
    
    Parameters
    ----------
    preds : np.ndarray
        Predicted probabilities (shape: (n_samples,)).
    y : np.ndarray
        True labels (shape: (n_samples,)).
    
    Returns
    -------
    float
        Accuracy (between 0 and 1).
    """
    # Convert probabilities to hard predictions (threshold = 0.5)
    hard_preds = (preds >= 0.5).astype(int)
    return compute_accuracy(hard_preds, y)


def compute_mask_sparsity(mask: np.ndarray) -> float:
    """
    Compute the sparsity of the mask (fraction of active entanglers).
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask with shape (depth, n_edges).
    
    Returns
    -------
    float
        Fraction of active entanglers (between 0 and 1).
    """
    return float(np.sum(mask == 1) / mask.size)


def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    depth: int,
    pairs: list
) -> None:
    """
    Validate input shapes and values.
    
    Parameters
    ----------
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    pairs : list
        Qubit pairs.
    
    Raises
    ------
    ValueError
        If shapes or values are invalid.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Check X shape
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array (n_samples, n_features), got shape {X.shape}")
    
    n_samples, n_features = X.shape
    
    # Check y shape
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array (n_samples,), got shape {y.shape}")
    
    if len(y) != n_samples:
        raise ValueError(
            f"X and y must have same number of samples: {n_samples} != {len(y)}"
        )
    
    # Check y values
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1 values")
    
    # Warn if dataset is very small
    if n_samples < 5:
        import warnings
        warnings.warn(
            f"Dataset is very small ({n_samples} samples). "
            f"Results may not be reliable.",
            UserWarning
        )
    
    # Validate pairs
    n_edges = len(pairs)
    if n_edges == 0:
        raise ValueError("pairs list cannot be empty")
    
    for i, j in pairs:
        if i < 0 or i >= n_qubits or j < 0 or j >= n_qubits:
            raise ValueError(
                f"Invalid qubit pair ({i}, {j}) for n_qubits={n_qubits}"
            )


# ============================================================================
# Pruning Functions
# ============================================================================

def greedy_prune(
    theta: np.ndarray,
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    pairs: list,
    n_qubits: int,
    depth: int,
    lam: float,
    tolerance: float,
    channel_strength: float = 0.4,
    recompute_baseline: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Perform greedy pruning of entangling gates.
    
    For each active entangler, temporarily disable it and check if the loss
    increases by more than the tolerance. If not, permanently disable it.
    
    Parameters
    ----------
    theta : np.ndarray
        Current parameters.
    mask : np.ndarray
        Current mask (will be copied, not modified in-place).
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    pairs : list
        Qubit pairs for coupling map.
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    lam : float
        Regularization strength.
    tolerance : float
        Maximum loss increase allowed when pruning an entangler.
    channel_strength : float
        Maximum strength for evidence channels.
    recompute_baseline : bool
        If True, recompute baseline loss after each accepted prune (default: True).
    
    Returns
    -------
    Tuple[np.ndarray, int]
        (updated_mask, number_of_pruned_gates)
    """
    # Ensure mask is numpy int array
    mask = np.asarray(mask, dtype=int)
    
    # Compute baseline loss
    baseline_result = forward_loss(
        theta=theta,
        mask=mask,
        X=X,
        y=y,
        lam=lam,
        pairs=pairs,
        backend=None,
        n_qubits=n_qubits,
        depth=depth,
        channel_strength=channel_strength,
    )
    baseline_loss = baseline_result["total_loss"]
    
    n_pruned = 0
    mask_new = mask.copy()
    
    # Try pruning each active entangler
    for d in range(depth):
        for k in range(len(pairs)):
            if mask_new[d, k] == 1:  # Only try pruning active gates
                # Temporarily disable this entangler
                mask_test = mask_new.copy()
                mask_test[d, k] = 0
                
                # Compute loss with this gate disabled
                test_result = forward_loss(
                    theta=theta,
                    mask=mask_test,
                    X=X,
                    y=y,
                    lam=lam,
                    pairs=pairs,
                    backend=None,
                    n_qubits=n_qubits,
                    depth=depth,
                    channel_strength=channel_strength,
                )
                test_loss = test_result["total_loss"]
                
                # If loss doesn't increase by more than tolerance, prune it
                if test_loss <= baseline_loss + tolerance:
                    mask_new[d, k] = 0
                    n_pruned += 1
                    
                    # Recompute baseline if requested
                    if recompute_baseline:
                        baseline_result = forward_loss(
                            theta=theta,
                            mask=mask_new,
                            X=X,
                            y=y,
                            lam=lam,
                            pairs=pairs,
                            backend=None,
                            n_qubits=n_qubits,
                            depth=depth,
                            channel_strength=channel_strength,
                        )
                        baseline_loss = baseline_result["total_loss"]
                    else:
                        baseline_loss = test_loss
    
    # Ensure output is numpy int array
    mask_new = np.asarray(mask_new, dtype=int)
    
    return mask_new, n_pruned


# ============================================================================
# Main Training Function
# ============================================================================

def main(
    n_qubits: int = 2,
    depth: int = 3,
    n_iterations: int = 100,
    lr: float = 0.01,
    lam: float = 0.1,
    prune_every: int = 20,
    tolerance: float = 0.01,
    seed: int = 42,
    dataset_name: str = "pothos_chater_small",
    channel_strength: float = 0.4,
    optimizer_type: str = "finite_diff",
    debug_predictions_every: Optional[int] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Main training function for compressed model with pruning.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth.
    n_iterations : int
        Number of training iterations.
    lr : float
        Learning rate.
    lam : float
        Regularization strength for two-qubit gate penalty.
    prune_every : int
        Perform pruning every N iterations.
    tolerance : float
        Maximum loss increase allowed when pruning an entangler.
    seed : int
        Random seed for reproducibility.
    dataset_name : str
        Name of dataset to use.
    channel_strength : float
        Maximum strength for evidence channels.
    optimizer_type : str
        Optimizer to use: "finite_diff" or "adam" (default: "finite_diff").
    debug_predictions_every : int, optional
        Print prediction debug info every N iterations (default: None).
    output_dir : str
        Output directory for results (default: "results").
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with training results:
        - theta: final parameters
        - mask: final mask
        - loss: final loss value
        - accuracy: final accuracy
        - two_qubit_count: final two-qubit gate count
        - history: DataFrame with training history
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create run directory with timestamp
    timestamp = timestamp_str()
    run_name = f"compressed_{timestamp}"
    base_results_dir = Path(output_dir)
    run_dir = base_results_dir / "runs" / run_name
    
    # Also create legacy directories for backward compatibility
    results_dir = base_results_dir
    results_dir.mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    
    # Prepare config for saving
    config = {
        "experiment_name": run_name,
        "mode": "compressed",
        "training": {
            "n_qubits": n_qubits,
            "depth": depth,
            "n_iterations": n_iterations,
            "lr": lr,
            "lam": lam,
            "prune_every": prune_every,
            "tolerance": tolerance,
            "seed": seed,
            "channel_strength": channel_strength,
            "optimizer": optimizer_type,
            "debug_predictions_every": debug_predictions_every,
        },
        "dataset": {
            "name": dataset_name,
        }
    }
    
    # Save config
    save_training_config(run_dir, config, verbose=True)
    
    # Load dataset
    X, y = get_toy_dataset(name=dataset_name)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Set up coupling map (default: linear chain)
    pairs = get_default_pairs(n_qubits)
    n_edges = len(pairs)
    
    # Validate inputs
    validate_inputs(X, y, n_qubits, depth, pairs)
    
    # Initialize parameters
    theta = init_random_theta(n_qubits, depth, n_edges, scale=0.1, seed=seed)
    mask = init_full_mask(depth, n_edges)
    
    # Validate shapes
    validate_theta_and_mask(theta, mask, n_qubits, n_edges, depth)
    
    print(f"Initialized parameters: theta shape {theta.shape}, mask shape {mask.shape}")
    print(f"Total trainable parameters: {np.prod(theta.shape)}")
    print(f"Initial active entanglers: {np.sum(mask == 1)} / {mask.size}")
    
    # Store initial mask for comparison
    mask_initial = mask.copy()
    
    # Initialize optimizer
    if optimizer_type.lower() == "adam":
        optimizer = AdamOptimizer(theta.shape, lr=lr)
        print(f"Using Adam optimizer (lr={lr})")
    else:
        optimizer = None
        print(f"Using finite-difference gradient descent (lr={lr})")
    
    # Training history
    history = []
    
    # Arrays for loss history (for NPZ saving)
    loss_history = []
    ce_loss_history = []
    gate_cost_history = []
    
    # Array for mask history (for compressed training)
    mask_history = []
    
    # Define loss function for optimization
    def loss_fn(theta_flat: np.ndarray) -> float:
        """Loss function that takes flattened theta."""
        theta_reshaped = theta_flat.reshape(theta.shape)
        result = forward_loss(
            theta=theta_reshaped,
            mask=mask,
            X=X,
            y=y,
            lam=lam,
            pairs=pairs,
            backend=None,
            n_qubits=n_qubits,
            depth=depth,
            channel_strength=channel_strength,
        )
        return result["total_loss"]
    
    # Training loop
    print(f"\nStarting training ({n_iterations} iterations)...")
    print(f"Pruning every {prune_every} iterations with tolerance {tolerance}")
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Compute current loss and metrics
        result = forward_loss(
            theta=theta,
            mask=mask,
            X=X,
            y=y,
            lam=lam,
            pairs=pairs,
            backend=None,
            n_qubits=n_qubits,
            depth=depth,
            channel_strength=channel_strength,
        )
        
        total_loss = result["total_loss"]
        ce_loss = result["ce_loss"]
        twoq = result["two_q_cost"]
        preds = result["preds"]
        
        # Compute accuracy
        accuracy = compute_accuracy_from_predictions(preds, y)
        
        # Compute mask sparsity
        sparsity = compute_mask_sparsity(mask)
        
        # Debug predictions if requested
        if debug_predictions_every is not None and (iteration + 1) % debug_predictions_every == 0:
            print(f"\n[Iteration {iteration+1}] Prediction Debug:")
            print(f"  Predictions: {preds[:5]}... (showing first 5)")
            print(f"  True labels: {y[:5]}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CE loss: {ce_loss:.6f}")
            print(f"  Mask sparsity: {sparsity:.4f}")
        
        # Log metrics
        history.append({
            "iteration": iteration,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "accuracy": accuracy,
            "two_qubit_count": twoq,
            "mask_sparsity": sparsity,
            "active_entanglers": int(np.sum(mask == 1)),
        })
        
        # Store for loss history NPZ
        loss_history.append(total_loss)
        ce_loss_history.append(ce_loss)
        gate_cost_history.append(twoq)
        
        # Store mask for mask history
        mask_history.append(mask.copy())
        
        # Update theta using gradient descent
        grad = finite_diff_gradient(loss_fn, theta, h=1e-5)
        
        if optimizer is not None:
            # Use Adam optimizer
            update = optimizer.step(grad)
            theta = theta - update
        else:
            # Use finite-difference gradient descent
            theta = theta - lr * grad
        
        # Perform pruning periodically
        if (iteration + 1) % prune_every == 0 and iteration > 0:
            print(f"\n[Iteration {iteration+1}] Performing greedy pruning...")
            mask_before = mask.copy()
            n_active_before = np.sum(mask == 1)
            
            mask, n_pruned = greedy_prune(
                theta=theta,
                mask=mask,
                X=X,
                y=y,
                pairs=pairs,
                n_qubits=n_qubits,
                depth=depth,
                lam=lam,
                tolerance=tolerance,
                channel_strength=channel_strength,
                recompute_baseline=True,
            )
            
            n_active_after = np.sum(mask == 1)
            sparsity_after = compute_mask_sparsity(mask)
            print(f"  Pruned {n_pruned} entanglers ({n_active_before} -> {n_active_after} active)")
            print(f"  New sparsity: {sparsity_after:.4f}")
    
    # Final evaluation
    print("\nComputing final metrics...")
    final_result = forward_loss(
        theta=theta,
        mask=mask,
        X=X,
        y=y,
        lam=lam,
        pairs=pairs,
        backend=None,
        n_qubits=n_qubits,
        depth=depth,
        channel_strength=channel_strength,
    )
    
    final_loss = final_result["total_loss"]
    final_ce_loss = final_result["ce_loss"]
    final_twoq = final_result["two_q_cost"]
    final_preds = final_result["preds"]
    final_accuracy = compute_accuracy_from_predictions(final_preds, y)
    final_sparsity = compute_mask_sparsity(mask)
    
    # Convert history to DataFrame
    df_history = pd.DataFrame(history)
    
    # Save all outputs using logging utilities
    print(f"\nSaving outputs to {run_dir}...")
    
    # Save training history CSV
    save_training_history(run_dir, df_history, verbose=True)
    
    # Also save to legacy location for backward compatibility
    log_path = results_dir / "logs" / "compressed_log.csv"
    df_history.to_csv(log_path, index=False)
    
    # Save final parameters
    save_parameters(run_dir, theta, mask=mask, verbose=True)
    
    # Also save to legacy location
    final_params_path = results_dir / "compressed_final.npz"
    np.savez(
        final_params_path,
        theta=theta,
        mask=mask,
    )
    
    # Save loss history
    save_loss_history(
        run_dir,
        loss=np.array(loss_history),
        ce_loss=np.array(ce_loss_history),
        gate_cost=np.array(gate_cost_history),
        verbose=True
    )
    
    # Save mask history
    mask_history_array = np.array(mask_history)  # Shape: (n_iterations, depth, n_edges)
    save_mask_history(run_dir, mask_history_array, verbose=True)
    
    # Save final metrics
    final_metrics = {
        "final_loss": float(final_loss),
        "final_ce_loss": float(final_ce_loss),
        "final_accuracy": float(final_accuracy),
        "final_two_qubit_count": int(final_twoq),
        "final_mask_sparsity": float(final_sparsity),
        "final_active_entanglers": int(np.sum(mask == 1)),
        "total_entanglers": int(mask.size),
        "n_iterations": n_iterations,
        "n_qubits": n_qubits,
        "depth": depth,
    }
    save_final_metrics(run_dir, final_metrics, verbose=True)
    
    # Generate plots (save to both run directory and legacy location)
    print("\nGenerating plots...")
    
    # Save to run directory
    plot_all_curves_from_history(
        df_history.to_dict("list"),
        prefix="compressed",
        output_dir=str(run_dir / "figures"),
        verbose=True
    )
    
    # Also save to legacy location
    plot_all_curves_from_history(
        df_history.to_dict("list"),
        prefix="compressed",
        output_dir=str(results_dir / "figures"),
        verbose=False
    )
    
    # Plot mask visualizations
    plot_mask_heatmap(
        mask,
        title="Final Pruned Mask",
        fname="compressed_mask_final",
        output_dir=str(run_dir / "figures"),
        verbose=True
    )
    
    plot_mask_before_after(
        mask_initial,
        mask,
        fname="compressed_mask_before_after",
        output_dir=str(run_dir / "figures"),
        title_before="Initial Mask (All Active)",
        title_after="Final Mask (Pruned)",
        verbose=True
    )
    
    # Plot Pareto front
    if len(df_history) > 1:
        plot_pareto_from_runs(
            [{
                "name": "compressed",
                "acc": final_accuracy,
                "cost": final_twoq,
                "loss": final_loss
            }],
            output_dir=str(run_dir / "figures"),
            fname="compressed_pareto",
            verbose=True
        )
    
    # Plot predictions vs true labels
    plot_pred_vs_true(
        final_preds,
        y,
        title="Compressed: Predicted vs True Labels",
        fname="compressed_pred_vs_true",
        output_dir=str(run_dir / "figures"),
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Run name: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final CE loss: {final_ce_loss:.6f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Final two-qubit gate count: {final_twoq}")
    print(f"Final mask sparsity: {final_sparsity:.4f}")
    print(f"Final active entanglers: {np.sum(mask == 1)} / {mask.size}")
    print(f"Total iterations: {n_iterations}")
    print("="*50)
    
    # Print sparsity report
    print("\nSparsity Report:")
    print(f"  Initial active entanglers: {np.sum(mask_initial == 1)}")
    print(f"  Final active entanglers: {np.sum(mask == 1)}")
    print(f"  Reduction: {np.sum(mask_initial == 1) - np.sum(mask == 1)} gates")
    print(f"  Compression ratio: {final_sparsity:.2%}")
    
    return {
        "theta": theta,
        "mask": mask,
        "loss": float(final_loss),
        "accuracy": float(final_accuracy),
        "two_qubit_count": int(final_twoq),
        "history": df_history,
        "run_name": run_name,
        "run_dir": str(run_dir),
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train compressed quantum Bayesian learner with pruning"
    )
    parser.add_argument("--n_qubits", type=int, default=2,
                       help="Number of qubits (default: 2)")
    parser.add_argument("--depth", type=int, default=3,
                       help="Circuit depth (default: 3)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of training iterations (default: 100)")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate (default: 0.01)")
    parser.add_argument("--lam", type=float, default=0.1,
                       help="Regularization strength (default: 0.1)")
    parser.add_argument("--prune_every", type=int, default=20,
                       help="Perform pruning every N iterations (default: 20)")
    parser.add_argument("--tolerance", type=float, default=0.01,
                       help="Pruning tolerance (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--dataset", type=str, default="pothos_chater_small",
                       choices=["pothos_chater_small", "pothos_chater_medium", "pothos_chater_large"],
                       help="Dataset name (default: pothos_chater_small)")
    parser.add_argument("--channel_strength", type=float, default=0.4,
                       help="Evidence channel strength (default: 0.4)")
    parser.add_argument("--optimizer", type=str, default="finite_diff",
                       choices=["finite_diff", "adam"],
                       help="Optimizer type (default: finite_diff)")
    parser.add_argument("--debug_predictions_every", type=int, default=None,
                       help="Debug predictions every N iterations (default: None)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory (default: results)")
    
    args = parser.parse_args()
    
    results = main(
        n_qubits=args.n_qubits,
        depth=args.depth,
        n_iterations=args.iterations,
        lr=args.lr,
        lam=args.lam,
        prune_every=args.prune_every,
        tolerance=args.tolerance,
        seed=args.seed,
        dataset_name=args.dataset,
        channel_strength=args.channel_strength,
        optimizer_type=args.optimizer,
        debug_predictions_every=args.debug_predictions_every,
        output_dir=args.output_dir,
    )
    
    print("\nTraining completed successfully!")
