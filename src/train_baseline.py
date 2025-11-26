"""
train_baseline.py

Baseline training script for the quantum Bayesian learner.

This script trains the learner with all entangling gates active (no compression).
It optimizes only the continuous parameters theta while keeping the mask fixed
at all 1s. Uses Adam optimizer or finite-difference gradient descent.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from .data import get_toy_dataset
from .learner import forward_loss


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
    
    This is used for logging purposes to track classification performance
    separately from the complexity penalty.
    """
    # Build ansatz and get two-qubit count (needed for forward_loss)
    from .ansatz import build_ansatz
    from .transpile_utils import transpile_and_count_2q
    
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    _, twoq = transpile_and_count_2q(ansatz, backend=None)
    
    # Compute full loss
    total_loss, _ = forward_loss(
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
    
    return total_loss


def main(
    n_qubits: int = 2,
    depth: int = 3,
    n_iterations: int = 100,
    lr: float = 0.01,
    lam: float = 0.1,
    seed: int = 42,
    dataset_name: str = "pothos_chater_small",
    channel_strength: float = 0.4,
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
    Returns
    -------
    Dict[str, Any]
        Dictionary with training results:
        - theta: final parameters
        - loss: final loss value
        - two_qubit_count: final two-qubit gate count
        - history: DataFrame with training history
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_dir.joinpath("logs").mkdir(exist_ok=True)
    results_dir.joinpath("figures").mkdir(exist_ok=True)
    
    # Load dataset
    X, y = get_toy_dataset(name=dataset_name)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Set up coupling map (default: linear chain)
    if n_qubits == 2:
        pairs = [(0, 1)]
    else:
        pairs = [(i, i+1) for i in range(n_qubits - 1)]
    
    n_edges = len(pairs)
    
    # Initialize parameters
    # theta shape: (depth, max(n_qubits, n_edges), 5)
    max_dim = max(n_qubits, n_edges)
    theta = np.random.randn(depth, max_dim, 5) * 0.1  # Small random initialization
    
    # Initialize mask: all entanglers active
    mask = np.ones((depth, n_edges), dtype=int)
    
    print(f"Initialized parameters: theta shape {theta.shape}, mask shape {mask.shape}")
    print(f"Total trainable parameters: {np.prod(theta.shape)}")
    
    # Training history
    history = []
    
    # Define loss function for optimization
    def loss_fn(theta_flat: np.ndarray) -> float:
        """Loss function that takes flattened theta."""
        theta_reshaped = theta_flat.reshape(theta.shape)
        loss, _ = forward_loss(
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
        return loss
    
    # Training loop
    print(f"\nStarting training ({n_iterations} iterations)...")
    print(f"Using finite-difference gradient descent optimizer")
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Compute current loss and metrics
        total_loss, twoq = forward_loss(
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
        
        # Compute CE loss only (for logging)
        ce_loss = compute_ce_loss_only(
            theta=theta,
            mask=mask,
            X=X,
            y=y,
            pairs=pairs,
            n_qubits=n_qubits,
            depth=depth,
            channel_strength=channel_strength,
        )
        
        # Log metrics
        history.append({
            "iteration": iteration,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "two_qubit_count": twoq,
        })
        
        # Compute gradient and update parameters
        # Use finite-difference gradient descent (Adam requires maintaining state)
        grad = finite_diff_gradient(loss_fn, theta, h=1e-5)
        theta = theta - lr * grad
    
    # Final evaluation
    print("\nComputing final metrics...")
    final_loss, final_twoq = forward_loss(
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
    
    # Save history to CSV
    df_history = pd.DataFrame(history)
    log_path = results_dir / "logs" / "baseline_log.csv"
    df_history.to_csv(log_path, index=False)
    print(f"Saved training history to {log_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final two-qubit gate count: {final_twoq}")
    print(f"Final CE loss: {ce_loss:.6f}")
    print(f"Total iterations: {n_iterations}")
    print("="*50)
    
    return {
        "theta": theta,
        "loss": float(final_loss),
        "two_qubit_count": int(final_twoq),
        "history": df_history,
    }


if __name__ == "__main__":
    # Run training with default parameters
    results = main(
        n_qubits=2,
        depth=3,
        n_iterations=50,  # Reduced for faster testing
        lr=0.01,
        lam=0.1,
        seed=42,
        dataset_name="pothos_chater_small",
    )
    print("\nTraining completed successfully!")

