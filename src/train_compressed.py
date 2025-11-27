"""
train_compressed.py

Compressed training script with structural pruning for the quantum Bayesian learner.

This script trains the learner with greedy pruning of entangling gates. It starts
with all entanglers active, optimizes theta continuously, and periodically prunes
entanglers that don't significantly impact the loss.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
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
    """
    result = forward_loss(
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
    return result['ce_loss']


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
        Current mask (will be modified in-place).
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
    
    Returns
    -------
    Tuple[np.ndarray, int]
        (updated_mask, number_of_pruned_gates)
    """
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
    baseline_loss = baseline_result['total_loss']
    
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
                test_loss = test_result['total_loss']
                
                # If loss doesn't increase by more than tolerance, prune it
                if test_loss <= baseline_loss + tolerance:
                    mask_new[d, k] = 0
                    n_pruned += 1
                    # Update baseline for next iteration (optional: recompute)
                    baseline_loss = test_loss
    
    return mask_new, n_pruned


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
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with training results:
        - theta: final parameters
        - mask: final mask
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
    print(f"Initial active entanglers: {np.sum(mask == 1)} / {mask.size}")
    
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
        
        total_loss = result['total_loss']
        ce_loss = result['ce_loss']
        twoq = result['two_q_cost']
        
        # Compute mask sparsity
        sparsity = compute_mask_sparsity(mask)
        
        # Log metrics
        history.append({
            "iteration": iteration,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "two_qubit_count": twoq,
            "mask_sparsity": sparsity,
            "active_entanglers": int(np.sum(mask == 1)),
        })
        
        # Update theta using gradient descent
        grad = finite_diff_gradient(loss_fn, theta, h=1e-5)
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
            )
            
            n_active_after = np.sum(mask == 1)
            print(f"  Pruned {n_pruned} entanglers ({n_active_before} -> {n_active_after} active)")
    
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
    
    final_loss = final_result['total_loss']
    final_twoq = final_result['two_q_cost']
    final_ce_loss = final_result['ce_loss']
    
    final_sparsity = compute_mask_sparsity(mask)
    
    # Save history to CSV
    df_history = pd.DataFrame(history)
    log_path = results_dir / "logs" / "compressed_log.csv"
    df_history.to_csv(log_path, index=False)
    print(f"Saved training history to {log_path}")
    
    # Save final parameters
    final_params_path = results_dir / "compressed_final.npz"
    np.savez(
        final_params_path,
        theta=theta,
        mask=mask,
    )
    print(f"Saved final parameters to {final_params_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final CE loss: {final_ce_loss:.6f}")
    print(f"Final two-qubit gate count: {final_twoq}")
    print(f"Final mask sparsity: {final_sparsity:.4f}")
    print(f"Final active entanglers: {np.sum(mask == 1)} / {mask.size}")
    print(f"Total iterations: {n_iterations}")
    print("="*50)
    
    return {
        "theta": theta,
        "mask": mask,
        "loss": float(final_loss),
        "two_qubit_count": int(final_twoq),
        "history": df_history,
    }


if __name__ == "__main__":
    # Run training with default parameters
    results = main(
        n_qubits=2,
        depth=3,
        n_iterations=60,  # Reduced for faster testing
        lr=0.01,
        lam=0.1,
        prune_every=20,
        tolerance=0.01,
        seed=42,
        dataset_name="pothos_chater_small",
    )
    print("\nTraining completed successfully!")

