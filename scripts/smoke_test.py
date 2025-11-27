"""
smoke_test.py

Smoke test to verify that the quantum-learning project environment works.

This script:
1. Imports all core modules
2. Loads a toy dataset and prints shapes
3. Builds a tiny ansatz (n_qubits=2, depth=1)
4. Prints the circuit
5. Calls forward_loss with a single sample
6. Prints the returned loss and 2-qubit gate count
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path to allow imports from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Import all core modules
from src.data import get_toy_dataset
from src.ansatz import (
    build_ansatz,
    get_default_pairs,
    init_random_theta,
    init_full_mask,
)
from src.channels import evidence_kraus
from src.learner import forward_loss
from src.transpile_utils import transpile_and_count_2q
from src.plots import plot_loss_curve  # Just to verify import works


def run_smoke_test():
    """Run the smoke test."""
    print("=" * 60)
    print("Quantum Learning Project - Smoke Test")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    X, y = get_toy_dataset("pothos_chater_small")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   X sample: {X[0]}")
    print(f"   y sample: {y[0]}")
    
    # 2. Build tiny ansatz
    print("\n2. Building tiny ansatz...")
    n_qubits = 2
    depth = 1
    
    # Get pairs for connectivity
    pairs = get_default_pairs(n_qubits)
    n_edges = len(pairs)
    print(f"   n_qubits: {n_qubits}")
    print(f"   depth: {depth}")
    print(f"   pairs: {pairs}")
    print(f"   n_edges: {n_edges}")
    
    # Initialize simple theta and mask
    theta = init_random_theta(n_qubits, depth, n_edges, scale=0.1, seed=42)
    mask = init_full_mask(depth, n_edges)
    print(f"   theta shape: {theta.shape}")
    print(f"   mask shape: {mask.shape}")
    print(f"   mask: {mask}")
    
    # Build the circuit
    qc = build_ansatz(n_qubits, depth, theta, mask, pairs)
    print(f"\n3. Circuit built:")
    print(f"   Number of qubits: {qc.num_qubits}")
    print(f"   Circuit depth: {qc.depth()}")
    print(f"   Number of gates: {len(qc.data)}")
    print(f"\n   Circuit:")
    print(qc)
    
    # 4. Call forward_loss with a single sample
    print("\n4. Calling forward_loss with single sample...")
    # Take first sample
    X_single = X[0:1]  # Keep as 2D array (1, n_features)
    y_single = y[0:1]  # Keep as 1D array (1,)
    
    print(f"   X_single shape: {X_single.shape}")
    print(f"   y_single shape: {y_single.shape}")
    print(f"   X_single: {X_single[0]}")
    print(f"   y_single: {y_single[0]}")
    print(f"   lam: 0.1")
    
    # Call forward_loss
    result = forward_loss(
        theta=theta,
        mask=mask,
        X=X_single,
        y=y_single,
        lam=0.1,
        pairs=pairs,
        n_qubits=n_qubits,
        depth=depth,
    )
    
    print(f"\n5. Results from forward_loss:")
    print(f"   total_loss: {result['total_loss']:.6f}")
    print(f"   ce_loss: {result['ce_loss']:.6f}")
    print(f"   two_q_cost: {result['two_q_cost']}")
    print(f"   avg_pred: {result['avg_pred']:.6f}")
    
    print("\n" + "=" * 60)
    print("✓ Smoke test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

