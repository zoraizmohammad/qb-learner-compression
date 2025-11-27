#!/usr/bin/env python3
"""
Test the full pipeline: ansatz -> evidence channel -> prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from qiskit.quantum_info import DensityMatrix
from src.data import get_toy_dataset
from src.learner import init_belief, apply_unitary, apply_evidence_channel, predict_proba
from src.ansatz import build_ansatz, init_random_theta, init_full_mask, get_default_pairs

print("=" * 60)
print("Full Pipeline Test")
print("=" * 60)

# Load dataset
X, y = get_toy_dataset("pothos_chater_large")
X_A = X[y == 0][:3]
X_B = X[y == 1][:3]

# Initialize
n_qubits, depth = 2, 3
pairs = get_default_pairs(n_qubits)
n_edges = len(pairs)
theta = init_random_theta(n_qubits, depth, n_edges, scale=1.0, seed=42)
mask = init_full_mask(depth, n_edges)
ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)

print(f"\nAnsatz has {len(ansatz.data)} instructions")
print(f"First few gates: {[str(gate) for gate in ansatz.data[:5]]}")

# Test step by step
print("\n" + "-" * 60)
print("Testing step-by-step for Category A sample:")
print("-" * 60)
xi = X_A[0]
print(f"Input features: {xi}")

# Step 1: Initial state
rho0 = init_belief(n_qubits)
p0 = predict_proba(rho0)
print(f"1. Initial state: prediction={p0:.6f}")
print(f"   Density matrix diagonal: {np.diag(rho0.data)}")

# Step 2: After ansatz
rho1 = apply_unitary(rho0, ansatz)
p1 = predict_proba(rho1)
print(f"2. After ansatz: prediction={p1:.6f}")
print(f"   Density matrix diagonal: {np.diag(rho1.data)}")
print(f"   Change from initial: {abs(p1-p0):.8f}")

# Step 3: After evidence channel
rho2 = apply_evidence_channel(rho1, xi, strength=0.6)
p2 = predict_proba(rho2)
print(f"3. After evidence channel: prediction={p2:.6f}")
print(f"   Density matrix diagonal: {np.diag(rho2.data)}")
print(f"   Change from after ansatz: {abs(p2-p1):.8f}")

# Test with Category B
print("\n" + "-" * 60)
print("Testing step-by-step for Category B sample:")
print("-" * 60)
xi = X_B[0]
print(f"Input features: {xi}")

rho0 = init_belief(n_qubits)
rho1 = apply_unitary(rho0, ansatz)
rho2 = apply_evidence_channel(rho1, xi, strength=0.6)
p2 = predict_proba(rho2)
print(f"Final prediction: {p2:.6f}")
print(f"Density matrix diagonal: {np.diag(rho2.data)}")

# Check if maximally mixed state is preserved by unitaries
print("\n" + "-" * 60)
print("Testing if maximally mixed state is invariant under unitaries:")
print("-" * 60)
rho_mixed = DensityMatrix(np.eye(4) / 4)  # 2-qubit maximally mixed
print(f"Maximally mixed state diagonal: {np.diag(rho_mixed.data)}")
rho_after = apply_unitary(rho_mixed, ansatz)
print(f"After ansatz diagonal: {np.diag(rho_after.data)}")
print(f"Are they equal? {np.allclose(rho_mixed.data, rho_after.data)}")

print("\n" + "=" * 60)

