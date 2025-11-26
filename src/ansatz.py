"""
ansatz.py

Variational ansatz for the quantum Bayesian learner.

Builds layered quantum circuits with Heisenberg-type entangling gates.
Uses a mask to selectively enable/disable entangling blocks, which helps
with hardware constraints and circuit compression.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate


def heisenberg_entangler_block(
    qc: QuantumCircuit,
    i: int,
    j: int,
    params: np.ndarray | List[float] | Tuple[float, float, float]
) -> None:
    """
    Add a Heisenberg entangling block between two qubits.
    
    This adds three rotation gates in sequence: RXX, RYY, and RZZ, all
    applied to the same qubit pair. The Heisenberg interaction is useful
    because it creates rich entanglement while still being relatively
    hardware-friendly.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to add gates to (modified in-place).
    i : int
        First qubit index.
    j : int
        Second qubit index.
    params : array-like of length 3
        Three rotation angles [α, β, γ] for RXX, RYY, and RZZ gates.
        Can be floats or Qiskit Parameter objects.
    
    Returns
    -------
    None
        Modifies the circuit in-place.
    
    Example
    -------
    >>> from qiskit import QuantumCircuit
    >>> import numpy as np
    >>> 
    >>> qc = QuantumCircuit(2)
    >>> params = [0.1, 0.2, 0.3]
    >>> heisenberg_entangler_block(qc, 0, 1, params)
    >>> print(qc)
    """
    params = np.asarray(params).flatten()
    if len(params) != 3:
        raise ValueError(f"params must have length 3, got {len(params)}")
    
    alpha, beta, gamma = params[0], params[1], params[2]
    
    # Append Heisenberg entangling gates
    qc.append(RXXGate(alpha), [i, j])
    qc.append(RYYGate(beta), [i, j])
    qc.append(RZZGate(gamma), [i, j])


def build_ansatz(
    n_qubits: int,
    depth: int,
    theta: np.ndarray,
    mask: np.ndarray,
    pairs: List[Tuple[int, int]]
) -> QuantumCircuit:
    """
    Build a layered variational ansatz with masked entangling gates.
    
    The circuit alternates between single-qubit rotations (RX, RZ) and
    entangling layers. The mask lets you turn entangling blocks on/off,
    which is handy for compression and respecting hardware constraints.
    
    Parameters
    ----------
    n_qubits : int
        How many qubits the circuit should have.
    depth : int
        Number of layers. Each layer has single-qubit rotations followed
        by entangling gates.
    theta : np.ndarray
        Parameter array with shape (depth, max(n_qubits, n_edges), 5).
        
        The structure is:
        - theta[d, q, 0] : RX angle for qubit q at depth d
        - theta[d, q, 1] : RZ angle for qubit q at depth d
        - theta[d, k, 2] : RXX angle (α) for edge k at depth d
        - theta[d, k, 3] : RYY angle (β) for edge k at depth d
        - theta[d, k, 4] : RZZ angle (γ) for edge k at depth d
        
        Use q < n_qubits for single-qubit params, k < len(pairs) for entangling.
    mask : np.ndarray
        Binary mask with shape (depth, n_edges). Set mask[d, k] = 1 to enable
        the entangling block on pairs[k] at depth d, or 0 to skip it.
        Can be boolean or integer array.
    pairs : List[Tuple[int, int]]
        List of qubit pairs that can have entangling gates. Should match
        your backend's coupling map.
    
    Returns
    -------
    QuantumCircuit
        The built parameterized circuit.
    
    Example
    -------
    >>> import numpy as np
    >>> from src.ansatz import build_ansatz
    >>> 
    >>> n_qubits = 2
    >>> depth = 3
    >>> pairs = [(0, 1)]
    >>> 
    >>> theta = np.random.randn(depth, max(n_qubits, len(pairs)), 5)
    >>> mask = np.ones((depth, len(pairs)), dtype=int)  # all blocks active
    >>> 
    >>> qc = build_ansatz(n_qubits, depth, theta, mask, pairs)
    >>> print(qc)
    """
    # Basic sanity checks
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if len(pairs) == 0:
        raise ValueError("pairs list cannot be empty")
    
    n_edges = len(pairs)
    
    # Make sure theta has the right shape
    expected_shape = (depth, max(n_qubits, n_edges), 5)
    if theta.shape != expected_shape:
        raise ValueError(
            f"theta must have shape {expected_shape}, got {theta.shape}"
        )
    
    # Make sure mask has the right shape
    expected_mask_shape = (depth, n_edges)
    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"mask must have shape {expected_mask_shape}, got {mask.shape}"
        )
    
    # Convert mask to integers (handles boolean arrays too)
    mask = np.asarray(mask, dtype=int)
    
    # Check that all qubit pairs are valid
    for i, j in pairs:
        if i < 0 or i >= n_qubits or j < 0 or j >= n_qubits:
            raise ValueError(
                f"Invalid qubit pair ({i}, {j}) for n_qubits={n_qubits}"
            )
        if i == j:
            raise ValueError(f"Self-loops not allowed: pair ({i}, {j})")
    
    # Start building the circuit
    qc = QuantumCircuit(n_qubits)
    
    # Build each layer
    for d in range(depth):
        # First, single-qubit rotations on all qubits
        for q in range(n_qubits):
            rx_angle = theta[d, q, 0]
            rz_angle = theta[d, q, 1]
            qc.rx(rx_angle, q)
            qc.rz(rz_angle, q)
        
        # Then, entangling gates (only where mask says to)
        for k, (i, j) in enumerate(pairs):
            if mask[d, k] == 1:
                heisenberg_params = theta[d, k, 2:5]  # [α, β, γ]
                heisenberg_entangler_block(qc, i, j, heisenberg_params)
    
    return qc


def count_parameters(theta: np.ndarray, mask: np.ndarray) -> int:
    """
    Count how many trainable parameters are actually used in the ansatz.
    
    Counts all single-qubit parameters (RX and RZ) plus only the entangling
    parameters for edges that are enabled by the mask. Useful for knowing
    how many parameters your optimizer needs to handle.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter array with shape (depth, max(n_qubits, n_edges), 5).
        Same structure as in build_ansatz.
    mask : np.ndarray
        Binary mask with shape (depth, n_edges). Same as in build_ansatz.
    
    Returns
    -------
    int
        Total number of active parameters.
    
    Notes
    -----
    The formula is:
        total = (depth * n_qubits * 2) + (number of active edges * 3)
    
    The first part counts RX and RZ rotations, the second counts the three
    parameters (α, β, γ) in each active Heisenberg block.
    
    Example
    -------
    >>> import numpy as np
    >>> from src.ansatz import count_parameters
    >>> 
    >>> depth = 3
    >>> n_qubits = 2
    >>> n_edges = 1
    >>> theta = np.random.randn(depth, max(n_qubits, n_edges), 5)
    >>> mask = np.ones((depth, n_edges), dtype=int)
    >>> 
    >>> n_params = count_parameters(theta, mask)
    >>> print(f"Active parameters: {n_params}")  # Should be 21
    """
    depth, max_dim, _ = theta.shape
    n_edges = mask.shape[1]
    
    # We don't know n_qubits exactly, so use max_dim as an upper bound
    # This gives a conservative estimate
    n_qubits = max_dim
    
    # Count single-qubit params: 2 per qubit per depth (RX + RZ)
    n_single_qubit_params = depth * n_qubits * 2
    
    # Count entangling params: 3 per active edge (α, β, γ)
    n_active_edges = int(np.sum(mask == 1))
    n_entangling_params = n_active_edges * 3
    
    total_params = n_single_qubit_params + n_entangling_params
    
    return total_params

