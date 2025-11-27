#!/usr/bin/env python3
"""
Test if evidence channels produce different predictions for different inputs.

This diagnostic script checks whether the evidence channel mechanism
is working correctly and can create separation between different categories.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data import get_toy_dataset
from src.learner import (
    init_belief,
    apply_unitary,
    apply_evidence_channel,
    predict_proba,
)
from src.ansatz import (
    build_ansatz,
    init_random_theta,
    init_full_mask,
    get_default_pairs,
)

print("=" * 60)
print("Evidence Channel Diagnostic Test")
print("=" * 60)

# Load dataset
X, y = get_toy_dataset("pothos_chater_large")
print(f"\nLoaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Get category A and B samples
X_A = X[y == 0]
X_B = X[y == 1]
print(f"Category A (label 0): {len(X_A)} samples")
print(f"Category B (label 1): {len(X_B)} samples")

# Initialize ansatz
n_qubits, depth = 2, 3
pairs = get_default_pairs(n_qubits)
theta = init_random_theta(n_qubits, depth, len(pairs), seed=42)
mask = init_full_mask(depth, len(pairs))
ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)

print(f"\nInitialized ansatz: {n_qubits} qubits, depth {depth}")
print(f"Number of entangling pairs: {len(pairs)}")

# Test predictions for different samples
print("\n" + "-" * 60)
print("Testing predictions for Category A samples (label 0):")
print("-" * 60)
preds_A = []
for i, xi in enumerate(X_A[:5]):
    rho = init_belief(n_qubits)
    rho_U = apply_unitary(rho, ansatz)
    rho_post = apply_evidence_channel(rho_U, xi, strength=0.4)
    p1 = predict_proba(rho_post)
    preds_A.append(p1)
    print(f"  Sample {i}: features={xi}, pred={p1:.6f}")

print("\n" + "-" * 60)
print("Testing predictions for Category B samples (label 1):")
print("-" * 60)
preds_B = []
for i, xi in enumerate(X_B[:5]):
    rho = init_belief(n_qubits)
    rho_U = apply_unitary(rho, ansatz)
    rho_post = apply_evidence_channel(rho_U, xi, strength=0.4)
    p1 = predict_proba(rho_post)
    preds_B.append(p1)
    print(f"  Sample {i}: features={xi}, pred={p1:.6f}")

# Statistics
preds_A = np.array(preds_A)
preds_B = np.array(preds_B)

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(f"Category A predictions: mean={preds_A.mean():.6f}, std={preds_A.std():.6f}")
print(f"Category B predictions: mean={preds_B.mean():.6f}, std={preds_B.std():.6f}")
print(f"Difference in means: {abs(preds_B.mean() - preds_A.mean()):.6f}")

if abs(preds_B.mean() - preds_A.mean()) < 0.01:
    print("\n⚠️  WARNING: Predictions are nearly identical for both categories!")
    print("   The evidence channel is not creating separation.")
    print("   This suggests the model cannot learn a decision boundary.")
else:
    print("\n✓ Predictions differ between categories.")
    print("  The evidence channel mechanism appears to be working.")

# Test with all samples
print("\n" + "-" * 60)
print("Testing with all samples:")
print("-" * 60)
all_preds = []
for xi, yi in zip(X, y):
    rho = init_belief(n_qubits)
    rho_U = apply_unitary(rho, ansatz)
    rho_post = apply_evidence_channel(rho_U, xi, strength=0.4)
    p1 = predict_proba(rho_post)
    all_preds.append(p1)

all_preds = np.array(all_preds)
preds_A_all = all_preds[y == 0]
preds_B_all = all_preds[y == 1]

print(f"All Category A: mean={preds_A_all.mean():.6f}, std={preds_A_all.std():.6f}")
print(f"All Category B: mean={preds_B_all.mean():.6f}, std={preds_B_all.std():.6f}")
print(f"Overall mean prediction: {all_preds.mean():.6f}")
print(f"Overall std prediction: {all_preds.std():.6f}")

if all_preds.std() < 0.01:
    print("\n⚠️  CRITICAL: All predictions are nearly constant!")
    print("   The model cannot distinguish between different inputs.")
    print("   This explains why accuracy stays at 0.5.")
else:
    print(f"\n✓ Predictions vary (std={all_preds.std():.6f}).")
    print("  The model has potential to learn, but may need:")
    print("  - Better initialization")
    print("  - Different learning rate")
    print("  - More iterations")
    print("  - Different optimizer")

print("\n" + "=" * 60)

