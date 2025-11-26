"""
learner.py

Core quantum Bayesian learner implementation.

This module implements the main learning loop for the quantum Bayesian learner.
It handles belief initialization, applying quantum channels for evidence updates,
constructing variational ansätze, transpiling circuits, and computing hybrid
loss functions that balance classification accuracy with circuit complexity.

The learner workflow:
1. Initialize belief state as maximally mixed (complete uncertainty)
2. Apply variational ansatz (unitary transformation) to transform the state
3. Apply evidence channel based on the stimulus features
4. Make predictions from the updated quantum state
5. Compute loss: binary cross-entropy + λ * two-qubit gate count

Parameter shapes:
- theta: (depth, max(n_qubits, n_edges), 5) - parameter tensor for ansatz
- mask: (depth, n_edges) - binary mask controlling which entangling blocks are active

Integration:
- channels.py: provides evidence_kraus() for building evidence channels
- ansatz.py: provides build_ansatz() for constructing variational circuits
- transpile_utils.py: provides transpile_and_count_2q() for gate counting
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit import QuantumCircuit

from .channels import evidence_kraus, apply_channel_to_density_matrix
from .ansatz import build_ansatz
from .transpile_utils import transpile_and_count_2q

# Alias for compatibility with prompt specification
def build_evidence_channel(x: np.ndarray, strength: float = 0.4):
    """
    Build an evidence channel from a stimulus vector.
    
    This is a convenience wrapper around evidence_kraus() that matches
    the interface specified in the prompt.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features (1D array).
    strength : float, optional
        Maximum channel strength (default: 0.4).
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel.
    """
    return evidence_kraus(x, strength=strength)


def init_belief(n_qubits: int) -> DensityMatrix:
    """
    Initialize the learner's belief state as a maximally mixed state.
    
    A maximally mixed state represents complete uncertainty - the learner
    starts with no prior information about which category a stimulus belongs to.
    This is the quantum analog of a uniform prior in classical Bayesian learning.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum state.
    
    Returns
    -------
    DensityMatrix
        The maximally mixed state: I / (2^n_qubits)
    
    Example
    -------
    >>> rho = init_belief(2)
    >>> print(rho)  # Should be I/4 for 2 qubits
    """
    dim = 2 ** n_qubits
    return DensityMatrix(np.eye(dim, dtype=complex) / dim)


def apply_evidence_channel(rho: DensityMatrix, x: np.ndarray, strength: float = 0.4) -> DensityMatrix:
    """
    Apply the evidence channel for stimulus x to update the belief state.
    
    This takes the current quantum state and applies a channel that encodes
    evidence from the stimulus. The channel strength depends on the stimulus
    features. This is the core belief update mechanism in the quantum Bayesian learner.
    
    The channel is applied to the first qubit only. For multi-qubit states,
    the channel acts on qubit 0 while leaving other qubits unchanged.
    
    Parameters
    ----------
    rho : DensityMatrix
        Current belief state (density matrix).
    x : np.ndarray
        Stimulus features (1D array).
    strength : float, optional
        Maximum channel strength (default: 0.4).
    
    Returns
    -------
    DensityMatrix
        Updated belief state after applying the evidence channel.
    
    Notes
    -----
    The channel is built using evidence_kraus() from channels.py, which creates
    an amplitude-damping-like channel where the damping parameter depends on
    the stimulus features. For multi-qubit states, we apply it to qubit 0 using
    the qargs parameter.
    """
    channel = build_evidence_channel(x, strength=strength)
    
    # Apply channel to first qubit only (qargs=[0])
    # For single-qubit states, qargs is not needed
    n_qubits = int(np.log2(rho.dim))
    if n_qubits == 1:
        new_rho = apply_channel_to_density_matrix(rho, channel)
    else:
        # Apply channel to qubit 0 only
        new_rho = rho.evolve(channel, qargs=[0])
    
    # Normalize for safety (should already be normalized, but just in case)
    trace = np.trace(new_rho.data)
    if abs(trace) > 1e-10:  # Avoid division by zero
        new_rho = DensityMatrix(new_rho.data / trace)
    
    return new_rho


def predict_label(rho: DensityMatrix) -> float:
    """
    Predict the class label from the quantum state.
    
    The prediction is the probability of measuring |1⟩ on the first qubit.
    This gives us a continuous value between 0 and 1 that we can interpret
    as the probability of class 1.
    
    Parameters
    ----------
    rho : DensityMatrix
        The quantum state (density matrix) to make a prediction from.
    
    Returns
    -------
    float
        Probability of class 1 (between 0 and 1).
    
    Notes
    -----
    This computes the reduced density matrix of the first qubit by tracing
    out all other qubits, then returns the (1,1) element which is the
    probability of measuring |1⟩. For multi-qubit states, we reshape the
    density matrix to isolate the first qubit, then trace over the rest.
    """
    # Get the full density matrix
    full_rho = rho.data
    
    # For multi-qubit states, we need to trace out all qubits except the first
    n_qubits = int(np.log2(full_rho.shape[0]))
    
    if n_qubits == 1:
        # Single qubit case: just return the (1,1) element
        p1 = np.real(full_rho[1, 1])
    else:
        # Multi-qubit case: trace out qubits 1 through n_qubits-1
        # Reshape to separate first qubit from the rest: (2, 2^(n-1), 2, 2^(n-1))
        dim_first = 2
        dim_rest = 2 ** (n_qubits - 1)
        reshaped = full_rho.reshape(dim_first, dim_rest, dim_first, dim_rest)
        
        # Trace over the remaining qubits (partial trace)
        reduced = np.trace(reshaped, axis1=1, axis2=3)
        
        # Get probability of |1⟩ on first qubit
        p1 = np.real(reduced[1, 1])
    
    # Clamp to [0, 1] for safety
    return np.clip(p1, 0.0, 1.0)


def forward_loss(
    theta: np.ndarray,
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    pairs: Optional[List[Tuple[int, int]]] = None,
    backend: Optional[Any] = None,
    n_qubits: int = 2,
    depth: int = 3,
    channel_strength: float = 0.4,
) -> Tuple[float, int]:
    """
    Compute the hybrid loss for given parameters.
    
    The loss combines:
    - Classification loss: binary cross-entropy between predictions and labels
    - Complexity penalty: number of two-qubit gates after transpilation
    
    Loss = CE_loss + λ * two_qubit_gate_count
    
    For each stimulus:
    1. Start with maximally mixed state
    2. Apply ansatz unitary transformation
    3. Apply evidence channel based on stimulus
    4. Compute predicted class probability
    5. Compute binary cross-entropy
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter tensor with shape (depth, max(n_qubits, n_edges), 5).
        See build_ansatz documentation for structure.
    mask : np.ndarray
        Binary mask with shape (depth, n_edges) controlling which entangling
        blocks are active.
    X : np.ndarray
        Training data, shape (n_samples, n_features).
    y : np.ndarray
        Training labels, shape (n_samples,). Should be 0 or 1.
    lam : float
        Regularization strength (λ) for the two-qubit gate penalty.
    pairs : List[Tuple[int, int]], optional
        Qubit pairs for the coupling map. Defaults to [(0, 1)] for 2 qubits.
    backend : optional
        Qiskit backend for transpilation. If None, counts gates in original circuit.
    n_qubits : int, optional
        Number of qubits (default: 2).
    depth : int, optional
        Circuit depth (default: 3).
    channel_strength : float, optional
        Maximum strength for evidence channels (default: 0.4).
    
    Returns
    -------
    Tuple[float, int]
        (total_loss, two_qubit_gate_count)
    """
    # Default coupling map if not provided
    if pairs is None:
        if n_qubits == 2:
            pairs = [(0, 1)]
        else:
            pairs = [(i, i+1) for i in range(n_qubits - 1)]
    
    # Build the ansatz circuit
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    
    # Transpile and count two-qubit gates
    _, twoq = transpile_and_count_2q(ansatz, backend=backend)
    
    # Compute classification loss
    ce_loss = 0.0
    eps = 1e-8  # Small epsilon to avoid log(0)
    
    for xi, yi in zip(X, y):
        # Start with maximally mixed state
        rho = init_belief(n_qubits)
        
        # Apply the ansatz (unitary transformation)
        U = Operator(ansatz).data
        rho_U = DensityMatrix(U @ rho.data @ U.conj().T)
        
        # Apply evidence channel based on stimulus
        rho_post = apply_evidence_channel(rho_U, xi, strength=channel_strength)
        
        # Make prediction
        p1 = predict_label(rho_post)
        
        # Binary cross-entropy
        if yi == 1:
            ce_loss += -np.log(p1 + eps)
        else:
            ce_loss += -np.log(1.0 - p1 + eps)
    
    # Average over samples
    ce_loss /= len(X)
    
    # Total loss: classification + regularization
    total_loss = ce_loss + lam * twoq
    
    return total_loss, twoq


class QuantumBayesianLearner:
    """
    Wrapper class for the quantum Bayesian learner.
    
    This class encapsulates the learner's configuration and provides
    a convenient interface for computing losses during training.
    
    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits (default: 2).
    depth : int, optional
        Circuit depth (default: 3).
    lam : float, optional
        Regularization strength for two-qubit gate penalty (default: 0.1).
    pairs : List[Tuple[int, int]], optional
        Qubit pairs for coupling map. Defaults to [(0, 1)] for 2 qubits.
    backend : optional
        Qiskit backend for transpilation.
    channel_strength : float, optional
        Maximum strength for evidence channels (default: 0.4).
    
    Example
    -------
    >>> learner = QuantumBayesianLearner(n_qubits=2, depth=3, lam=0.1)
    >>> loss, twoq = learner.loss(theta, mask, X, y)
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        depth: int = 3,
        lam: float = 0.1,
        pairs: Optional[List[Tuple[int, int]]] = None,
        backend: Optional[Any] = None,
        channel_strength: float = 0.4,
    ):
        self.n_qubits = n_qubits
        self.depth = depth
        self.lam = lam
        self.backend = backend
        self.channel_strength = channel_strength
        
        # Default coupling map: all adjacent pairs
        if pairs is None:
            if n_qubits == 2:
                self.pairs = [(0, 1)]
            else:
                # For more qubits, create a linear chain
                self.pairs = [(i, i+1) for i in range(n_qubits - 1)]
        else:
            self.pairs = pairs
    
    def loss(
        self,
        theta: np.ndarray,
        mask: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Compute the loss for given parameters and data.
        
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
        
        Returns
        -------
        Tuple[float, int]
            (total_loss, two_qubit_gate_count)
        """
        return forward_loss(
            theta=theta,
            mask=mask,
            X=X,
            y=y,
            lam=self.lam,
            pairs=self.pairs,
            backend=self.backend,
            n_qubits=self.n_qubits,
            depth=self.depth,
            channel_strength=self.channel_strength,
        )

