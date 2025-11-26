"""
channels.py

Quantum evidence channels for updating the learner's belief state.

When the learner sees a stimulus, it needs to update its beliefs. This module
implements that update using quantum channels (specifically, Kraus operators).
The channels work like amplitude-damping, but the damping strength depends on
the stimulus features.

The idea is that different stimuli provide different amounts of evidence, which
gets encoded into how strongly the quantum state gets updated.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Kraus, DensityMatrix


def evidence_kraus(x: np.ndarray, strength: float = 0.4) -> Kraus:
    """
    Create a quantum channel from a stimulus that updates the belief state.

    The channel works like amplitude-damping: it gradually "damps" the quantum
    state, but how much damping happens depends on the stimulus features. This
    lets the stimulus provide evidence that changes the learner's beliefs.

    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array. For typical datasets, this might be
        shape (2,) or (3,). Features should be between 0 and 1.
    strength : float, optional
        Maximum damping strength (default: 0.4). The actual damping will be
        somewhere between 0 and this value, depending on x.

    Returns
    -------
    Kraus
        A Qiskit Kraus channel that can update a single-qubit density matrix.

    Notes
    -----
    The damping parameter p is computed from the mean of x:
        p = strength * (0.5 + 0.5 * (x.mean() - 0.5))
    
    So:
    - Neutral stimulus (mean = 0.5) → p = 0.5 * strength
    - Low features (mean = 0) → p = 0.25 * strength
    - High features (mean = 1) → p = 0.75 * strength
    
    The channel uses two Kraus operators K0 and K1 that implement the damping.

    Example
    -------
    >>> channel = evidence_kraus(stimulus_x, strength=0.4)
    >>> updated_rho = apply_channel_to_density_matrix(current_rho, channel)
    """
    # Flatten x in case it's not already 1D
    x = np.asarray(x).flatten()
    
    # Figure out how much damping based on the stimulus features
    # Maps the mean of x to a damping parameter between 0.25*strength and 0.75*strength
    p = strength * (0.5 + 0.5 * (x.mean() - 0.5))
    
    # Make sure p stays in a safe range (avoid exactly 0 or strength)
    eps = 1e-8
    p = np.clip(p, eps, strength - eps)
    
    # Build the Kraus operators for the amplitude-damping channel
    sqrt_p = np.sqrt(p)
    sqrt_one_minus_p = np.sqrt(1.0 - p)
    
    K0 = np.array([[1.0, 0.0],
                   [0.0, sqrt_one_minus_p]], dtype=complex)
    
    K1 = np.array([[0.0, sqrt_p],
                   [0.0, 0.0]], dtype=complex)
    
    return Kraus([K0, K1])


def apply_channel_to_density_matrix(rho: DensityMatrix, channel: Kraus) -> DensityMatrix:
    """
    Apply a quantum channel to update the density matrix.

    This is just a convenience wrapper around Qiskit's evolve method.
    Takes the current belief state and applies the channel to get the
    updated state.

    Parameters
    ----------
    rho : DensityMatrix
        The current belief state (density matrix).
    channel : Kraus
        The quantum channel to apply (usually from evidence_kraus).

    Returns
    -------
    DensityMatrix
        The updated belief state after applying the channel.

    Notes
    -----
    Under the hood, this does rho.evolve(channel), which is equivalent to
    the standard quantum channel formula: sum_i K_i @ rho @ K_i^dagger,
    where the K_i are the Kraus operators.

    Example
    -------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> import numpy as np
    >>> 
    >>> rho = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])
    >>> x = np.array([0.3, 0.7, 0.5])
    >>> channel = evidence_kraus(x, strength=0.4)
    >>> updated_rho = apply_channel_to_density_matrix(rho, channel)
    """
    return rho.evolve(channel)

