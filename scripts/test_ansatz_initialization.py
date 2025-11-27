#!/usr/bin/env python3
"""
Test if ansatz initialization creates non-uniform states.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from qiskit.quantum_info import DensityMatrix
from src.learner import init_belief, apply_unitary, predict_proba
from src.ansatz import build_ansatz, init_random_theta, init_full_mask, get_default_pairs

print("=" * 60)
print("Ansatz Initialization Test")
print("=" * 60)

n_qubits, depth = 2, 3
pairs = get_default_pairs(n_qubits)
n_edges = len(pairs)

# Test different initialization scales
scales = [0.01, 0.1, 0.5, 1.0, 2.0]

print("\nTesting different theta initialization scales:")
print("-" * 60)

for scale in scales:
    theta = init_random_theta(n_qubits, depth, n_edges, scale=scale, seed=42)
    mask = init_full_mask(depth, n_edges)
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    
    # Start with maximally mixed
    rho = init_belief(n_qubits)
    
    # Apply ansatz
    rho_U = apply_unitary(rho, ansatz)
    
    # Check prediction
    p1 = predict_proba(rho_U)
    
    print(f"Scale {scale:4.2f}: theta_std={theta.std():.6f}, prediction={p1:.6f}")

# Test if zero initialization gives 0.5
print("\n" + "-" * 60)
print("Testing zero initialization:")
print("-" * 60)
theta_zero = np.zeros((depth, max(n_qubits, n_edges), 5))
mask = init_full_mask(depth, n_edges)
ansatz = build_ansatz(n_qubits, depth, theta_zero, mask, pairs)
rho = init_belief(n_qubits)
rho_U = apply_unitary(rho, ansatz)
p1 = predict_proba(rho_U)
print(f"Zero theta: prediction={p1:.6f}")

# Test with larger random initialization
print("\n" + "-" * 60)
print("Testing larger random initialization (scale=5.0):")
print("-" * 60)
theta_large = init_random_theta(n_qubits, depth, n_edges, scale=5.0, seed=42)
ansatz = build_ansatz(n_qubits, depth, theta_large, mask, pairs)
rho = init_belief(n_qubits)
rho_U = apply_unitary(rho, ansatz)
p1 = predict_proba(rho_U)
print(f"Large scale (5.0): prediction={p1:.6f}")

print("\n" + "=" * 60)

