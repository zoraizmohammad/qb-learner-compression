"""
channels.py

Quantum evidence channels for updating the learner's belief state.

This module implements quantum channels (in Kraus representation) that transform
the learner's density matrix based on observed stimulus features. The channels
are inspired by amplitude-damping processes, where the damping strength is
modulated by the stimulus vector x.

The channel represents evidence-based belief updating: when a stimulus x is
observed, it provides evidence that modulates the quantum state, allowing the
learner to update its beliefs about category membership.

References:
- Used in the quantum-Bayesian learner loop to update density matrices
- Compatible with Qiskit's quantum_info module
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Kraus, DensityMatrix


def evidence_kraus(x: np.ndarray, strength: float = 0.4) -> Kraus:
    """
    Construct a quantum evidence channel from a stimulus vector.

    The channel is an amplitude-damping-like process where the damping
    parameter p is modulated by the features in x. This allows the stimulus
    to provide evidence that updates the learner's belief state.

    Parameters
    ----------
    x : np.ndarray
        1D array of stimulus features, typically of shape (n_features,).
        For the SHJ/Pothos-Chater dataset, this is typically shape (2,) or (3,).
        Features should be in [0, 1] for stable behavior.
    strength : float, optional
        Maximum damping strength (default: 0.4). The actual damping parameter
        p will satisfy 0 < p < strength.

    Returns
    -------
    Kraus
        A Qiskit Kraus channel representing the evidence-based update.
        The channel acts on a single qubit (2x2 density matrix).

    Notes
    -----
    The bias parameter p is computed as:
        p = strength * (0.5 + 0.5 * (x.mean() - 0.5))
    
    This ensures:
    - When x.mean() = 0.5 (neutral), p = 0.5 * strength
    - When x.mean() = 0 (low features), p = 0.25 * strength
    - When x.mean() = 1 (high features), p = 0.75 * strength
    
    The channel is defined by Kraus operators:
        K0 = [[1, 0],
              [0, sqrt(1 - p)]]
        K1 = [[0, sqrt(p)],
              [0, 0]]

    Use in learner loop:
        channel = evidence_kraus(stimulus_x, strength=0.4)
        updated_rho = apply_channel_to_density_matrix(current_rho, channel)
    """
    # Ensure x is 1D
    x = np.asarray(x).flatten()
    
    # Compute bias parameter p from stimulus features
    # Maps x.mean() from [0, 1] to p in [0.25*strength, 0.75*strength]
    p = strength * (0.5 + 0.5 * (x.mean() - 0.5))
    
    # Ensure p is in valid range: 0 < p < strength
    # Add small epsilon to avoid exactly 0 or strength
    eps = 1e-8
    p = np.clip(p, eps, strength - eps)
    
    # Construct Kraus operators for amplitude-damping-like channel
    sqrt_p = np.sqrt(p)
    sqrt_one_minus_p = np.sqrt(1.0 - p)
    
    K0 = np.array([[1.0, 0.0],
                   [0.0, sqrt_one_minus_p]], dtype=complex)
    
    K1 = np.array([[0.0, sqrt_p],
                   [0.0, 0.0]], dtype=complex)
    
    return Kraus([K0, K1])


def apply_channel_to_density_matrix(rho: DensityMatrix, channel: Kraus) -> DensityMatrix:
    """
    Apply a quantum channel to a density matrix.

    This helper function applies the channel to the density matrix using
    Qiskit's evolve method.

    Parameters
    ----------
    rho : DensityMatrix
        The current belief state (density matrix) to be updated.
    channel : Kraus
        The quantum channel (evidence channel) to apply.

    Returns
    -------
    DensityMatrix
        The updated density matrix after applying the channel.

    Notes
    -----
    The operation performed is:
        new_rho = rho.evolve(channel)
    
    This is equivalent to the quantum channel action:
        new_rho = sum_i K_i @ rho @ K_i^dagger
    
    where {K_i} are the Kraus operators of the channel.

    Example
    -------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> import numpy as np
    >>> 
    >>> # Initialize belief state
    >>> rho = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])
    >>> 
    >>> # Create evidence channel from stimulus
    >>> x = np.array([0.3, 0.7, 0.5])
    >>> channel = evidence_kraus(x, strength=0.4)
    >>> 
    >>> # Update belief state
    >>> updated_rho = apply_channel_to_density_matrix(rho, channel)
    """
    return rho.evolve(channel)

