"""
train_baseline.py

Baseline training script for the quantum Bayesian learner.

This script trains the learner with all entangling gates active (no compression).
It optimizes only the continuous parameters theta while keeping the mask fixed
at all 1s. Supports both Adam optimizer and finite-difference gradient descent.

Usage:
    Run from repo root: python -m src.train_baseline
    
    Quick test (20 iterations):
    python -m src.train_baseline --iterations 20
    
    Full training with custom parameters:
    python -m src.train_baseline --iterations 100 --n_qubits 2 --depth 3 --lr 0.01 --lam 0.1
    source venv/bin/activate && python -m src.train_compressed --iterations 40
    Command to run: cd /Users/mzoraiz/Desktop/qb-learner-compression && source venv/bin/activate && python src/train_baseline.py
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from .data import get_toy_dataset
from .learner import forward_loss, compute_accuracy, predict_hard
from .ansatz import (
    get_default_pairs, build_parameter_shapes, init_random_theta,
    init_full_mask, validate_theta_and_mask
)
from .plots import (
    plot_all_curves_from_history, plot_pareto_from_runs, plot_pred_vs_true
)
from .logging_utils import (
    timestamp_str, save_training_config, save_final_metrics,
    save_training_history, save_parameters, save_loss_history
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
    readout_alpha: float = 4.0,
    feature_scale: float = np.pi,
) -> float:
    """
    Compute only the cross-entropy loss (without regularization).
    
    This is used for logging purposes to track classification performance
    separately from the complexity penalty.
    
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
        readout_alpha=readout_alpha,
        feature_scale=feature_scale,
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
# Main Training Function
# ============================================================================

def main(
    n_qubits: int = 2,
    depth: int = 3,
    n_iterations: int = 100,
    lr: float = 0.01,
    lam: float = 0.1,
    seed: int = 42,
    dataset_name: str = "pothos_chater_small",
    channel_strength: float = 0.4,
    optimizer_type: str = "finite_diff",
    debug_predictions_every: Optional[int] = None,
    output_dir: str = "results",
    readout_alpha: float = 4.0,
    feature_scale: float = np.pi,
) -> Dict[str, Any]:
    """
    Main training function for baseline model.
    
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
    readout_alpha : float, optional
        Temperature scaling factor for sigmoid readout (default: 4.0).
        Higher values (3-5) provide stronger nonlinearity.
    feature_scale : float, optional
        Scaling factor for feature encoding rotations (default: π).
        Features are multiplied by this before applying RY/RZ rotations.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with training results:
        - theta: final parameters
        - loss: final loss value
        - accuracy: final accuracy
        - two_qubit_count: final two-qubit gate count
        - history: DataFrame with training history
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create run directory with timestamp
    timestamp = timestamp_str()
    run_name = f"baseline_{timestamp}"
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
        "mode": "baseline",
        "training": {
            "n_qubits": n_qubits,
            "depth": depth,
            "n_iterations": n_iterations,
            "lr": lr,
            "lam": lam,
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
            readout_alpha=readout_alpha,
            feature_scale=feature_scale,
        )
        return result["total_loss"]
    
    # Training loop
    print(f"\nStarting training ({n_iterations} iterations)...")
    
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
            readout_alpha=readout_alpha,
            feature_scale=feature_scale,
        )
        
        total_loss = result["total_loss"]
        ce_loss = result["ce_loss"]
        twoq = result["two_q_cost"]
        preds = result["preds"]
        
        # Compute accuracy
        accuracy = compute_accuracy_from_predictions(preds, y)
        
        # Debug predictions if requested
        if debug_predictions_every is not None and (iteration + 1) % debug_predictions_every == 0:
            print(f"\n[Iteration {iteration+1}] Prediction Debug:")
            print(f"  Predictions: {preds[:5]}... (showing first 5)")
            print(f"  True labels: {y[:5]}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CE loss: {ce_loss:.6f}")
        
        # Log metrics
        history.append({
            "iteration": iteration,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "accuracy": accuracy,
            "two_qubit_count": twoq,
        })
        
        # Store for loss history NPZ
        loss_history.append(total_loss)
        ce_loss_history.append(ce_loss)
        gate_cost_history.append(twoq)
        
        # Compute gradient and update parameters
        grad = finite_diff_gradient(loss_fn, theta, h=1e-5)
        
        if optimizer is not None:
            # Use Adam optimizer
            update = optimizer.step(grad)
            theta = theta - update
        else:
            # Use finite-difference gradient descent
            theta = theta - lr * grad
    
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
        readout_alpha=readout_alpha,
        feature_scale=feature_scale,
    )
    
    final_loss = final_result["total_loss"]
    final_ce_loss = final_result["ce_loss"]
    final_twoq = final_result["two_q_cost"]
    final_preds = final_result["preds"]
    final_accuracy = compute_accuracy_from_predictions(final_preds, y)
    
    # Convert history to DataFrame
    df_history = pd.DataFrame(history)
    
    # Save all outputs using logging utilities
    print(f"\nSaving outputs to {run_dir}...")
    
    # Save training history CSV
    save_training_history(run_dir, df_history, verbose=True)
    
    # Also save to legacy location for backward compatibility
    log_path = results_dir / "logs" / "baseline_log.csv"
    df_history.to_csv(log_path, index=False)
    
    # Save final parameters
    save_parameters(run_dir, theta, mask=None, verbose=True)
    
    # Also save to legacy location
    theta_path = results_dir / "baseline_final_theta.npy"
    np.save(theta_path, theta)
    
    # Save loss history
    save_loss_history(
        run_dir,
        loss=np.array(loss_history),
        ce_loss=np.array(ce_loss_history),
        gate_cost=np.array(gate_cost_history),
        verbose=True
    )
    
    # Save final metrics
    final_metrics = {
        "final_loss": float(final_loss),
        "final_ce_loss": float(final_ce_loss),
        "final_accuracy": float(final_accuracy),
        "final_two_qubit_count": int(final_twoq),
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
        prefix="baseline",
        output_dir=str(run_dir / "figures"),
        verbose=True
    )
    
    # Also save to legacy location
    plot_all_curves_from_history(
        df_history.to_dict("list"),
        prefix="baseline",
        output_dir=str(results_dir / "figures"),
        verbose=False
    )
    
    # Plot Pareto front (if we have enough data)
    if len(df_history) > 1:
        plot_pareto_from_runs(
            [{
                "name": "baseline",
                "acc": final_accuracy,
                "cost": final_twoq,
                "loss": final_loss
            }],
            output_dir=str(run_dir / "figures"),
            fname="baseline_pareto",
            verbose=True
        )
    
    # Plot predictions vs true labels
    plot_pred_vs_true(
        final_preds,
        y,
        title="Baseline: Predicted vs True Labels",
        fname="baseline_pred_vs_true",
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
    print(f"Total iterations: {n_iterations}")
    print("="*50)
    
    return {
        "theta": theta,
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
        description="Train baseline quantum Bayesian learner"
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
        seed=args.seed,
        dataset_name=args.dataset,
        channel_strength=args.channel_strength,
        optimizer_type=args.optimizer,
        debug_predictions_every=args.debug_predictions_every,
        output_dir=args.output_dir,
    )
    
    print("\nTraining completed successfully!")
