#!/usr/bin/env python3
"""
Diagnostic script to check if gradients are being computed correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data import get_toy_dataset
from src.learner import forward_loss
from src.ansatz import init_random_theta, init_full_mask, get_default_pairs

print("=" * 60)
print("Gradient Diagnostic Test")
print("=" * 60)

# Load dataset
X, y = get_toy_dataset("pothos_chater_large")
print(f"\nLoaded dataset: {X.shape[0]} samples")

# Initialize parameters
n_qubits, depth = 2, 3
pairs = get_default_pairs(n_qubits)
n_edges = len(pairs)
theta = init_random_theta(n_qubits, depth, n_edges, scale=0.1, seed=42)
mask = init_full_mask(depth, n_edges)

print(f"Initialized theta: shape {theta.shape}, mean={theta.mean():.6f}, std={theta.std():.6f}")

# Compute initial loss
result = forward_loss(theta, mask, X, y, lam=0.01, n_qubits=n_qubits, depth=depth, channel_strength=0.6)
print(f"\nInitial loss: {result['total_loss']:.6f}")
print(f"Initial CE loss: {result['ce_loss']:.6f}")
print(f"Initial avg prediction: {result['avg_pred']:.6f}")
print(f"Prediction std: {result['preds'].std():.6f}")

# Test finite-difference gradients
print("\n" + "-" * 60)
print("Testing finite-difference gradients:")
print("-" * 60)

eps = 1e-5
grad_norms = []
grad_max = []
grad_min = []

# Sample a few parameters to test
test_indices = [
    (0, 0, 0),  # First layer, first qubit, first param
    (0, 0, 1),
    (1, 0, 0),
    (2, 0, 0),
]

for idx in test_indices:
    d, q, p = idx
    
    # Forward difference
    theta_plus = theta.copy()
    theta_plus[d, q, p] += eps
    result_plus = forward_loss(theta_plus, mask, X, y, lam=0.01, n_qubits=n_qubits, depth=depth, channel_strength=0.6)
    
    # Backward difference
    theta_minus = theta.copy()
    theta_minus[d, q, p] -= eps
    result_minus = forward_loss(theta_minus, mask, X, y, lam=0.01, n_qubits=n_qubits, depth=depth, channel_strength=0.6)
    
    # Central difference gradient
    grad = (result_plus['total_loss'] - result_minus['total_loss']) / (2 * eps)
    
    grad_norms.append(abs(grad))
    grad_max.append(grad)
    grad_min.append(grad)
    
    print(f"  Gradient at theta[{d},{q},{p}]: {grad:.8f}")

print(f"\nGradient statistics:")
print(f"  Mean |grad|: {np.mean(grad_norms):.8f}")
print(f"  Max grad: {np.max(grad_max):.8f}")
print(f"  Min grad: {np.min(grad_min):.8f}")

if np.mean(grad_norms) < 1e-8:
    print("\n⚠️  WARNING: Gradients are extremely small or zero!")
    print("   This suggests the loss landscape is flat or gradients aren't computed correctly.")
else:
    print("\n✓ Gradients are non-zero. The issue may be with learning rate or initialization.")

# Check if predictions change with parameter changes
print("\n" + "-" * 60)
print("Testing if predictions change with parameter perturbation:")
print("-" * 60)

theta_perturbed = theta.copy()
theta_perturbed += np.random.randn(*theta.shape) * 0.1
result_pert = forward_loss(theta_perturbed, mask, X, y, lam=0.01, n_qubits=n_qubits, depth=depth, channel_strength=0.6)

print(f"Original avg prediction: {result['avg_pred']:.6f}")
print(f"Perturbed avg prediction: {result_pert['avg_pred']:.6f}")
print(f"Difference: {abs(result_pert['avg_pred'] - result['avg_pred']):.6f}")

if abs(result_pert['avg_pred'] - result['avg_pred']) < 1e-6:
    print("\n⚠️  WARNING: Predictions don't change with parameter perturbation!")
    print("   This suggests the ansatz isn't affecting the predictions.")
else:
    print("\n✓ Predictions change with parameters. The ansatz has some effect.")

print("\n" + "=" * 60)

