"""
channels.py

Quantum evidence channels for updating the learner's belief state.

When the learner sees a stimulus, it needs to update its beliefs. This module
implements that update using quantum channels (specifically, Kraus operators).
The channels work like amplitude-damping, but the damping strength depends on
the stimulus features.

The idea is that different stimuli provide different amounts of evidence, which
gets encoded into how strongly the quantum state gets updated.

This module provides multiple channel types for experimentation:
- Amplitude damping (default): energy dissipation channel
- Phase damping: Z-basis decoherence channel
- Rotation channel: coherent unitary update
- Projective update: Bayesian-like soft measurement
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
from qiskit.quantum_info import Kraus, DensityMatrix


# ============================================================================
# Strength Computation Utilities
# ============================================================================

def compute_strength_from_stimulus(
    x: np.ndarray,
    strength_max: float = 0.4,
    method: str = "mean"
) -> float:
    """
    Compute channel strength parameter from stimulus features.
    
    This function maps stimulus features to a channel strength parameter
    in a smooth, reproducible way. The strength determines how strongly
    the evidence channel affects the quantum state.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array. Features should be between 0 and 1.
    strength_max : float, optional
        Maximum channel strength (default: 0.4).
    method : str, optional
        Method for computing strength from features:
        - "mean": Use mean of features (default)
        - "max": Use maximum feature value
        - "min": Use minimum feature value
        - "norm": Use L2 norm normalized to [0, 1]
    
    Returns
    -------
    float
        Channel strength parameter between eps and (strength_max - eps).
        The value is clipped to avoid exactly 0 or strength_max for
        numerical stability.
    
    Notes
    -----
    The formula for "mean" method:
        p = strength_max * (0.5 + 0.5 * (x.mean() - 0.5))
    
    This maps:
    - Neutral stimulus (mean = 0.5) → p = 0.5 * strength_max
    - Low features (mean = 0) → p = 0.25 * strength_max
    - High features (mean = 1) → p = 0.75 * strength_max
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7, 0.5])
    >>> strength = compute_strength_from_stimulus(x, strength_max=0.4)
    >>> print(f"{strength:.4f}")  # Approximately 0.2
    """
    x = np.asarray(x).flatten()
    
    if method == "mean":
        feature_value = np.mean(x)
    elif method == "max":
        feature_value = np.max(x)
    elif method == "min":
        feature_value = np.min(x)
    elif method == "norm":
        # Normalize L2 norm to [0, 1]
        norm = np.linalg.norm(x)
        max_norm = np.sqrt(len(x))  # Maximum possible norm
        feature_value = norm / max_norm if max_norm > 0 else 0.5
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of: mean, max, min, norm")
    
    # Map feature value to strength: smooth mapping from [0, 1] to [0.25*max, 0.75*max]
    # This ensures strength is never exactly 0 or max, which helps numerical stability
    p = strength_max * (0.5 + 0.5 * (feature_value - 0.5))
    
    # Clip to safe range (avoid exactly 0 or strength_max)
    eps = 1e-8
    p = np.clip(p, eps, strength_max - eps)
    
    return float(p)


# ============================================================================
# Channel Primitives
# ============================================================================

def evidence_amplitude_damping(
    x: np.ndarray,
    strength: float = 0.4,
    method: str = "mean"
) -> Kraus:
    """
    Create an amplitude-damping evidence channel from a stimulus.
    
    Amplitude damping models energy dissipation: it gradually moves the state
    toward |0⟩. The damping strength depends on the stimulus features.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array.
    strength : float, optional
        Maximum damping strength (default: 0.4).
    method : str, optional
        Method for computing strength from features (default: "mean").
        See compute_strength_from_stimulus for details.
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel implementing amplitude damping.
    
    Notes
    -----
    The amplitude-damping channel has Kraus operators:
        K0 = [[1, 0], [0, √(1-p)]]
        K1 = [[0, √p], [0, 0]]
    
    where p is the damping parameter computed from x.
    
    This channel is CPTP (completely positive trace-preserving).
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> channel = evidence_amplitude_damping(x, strength=0.4)
    >>> rho = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])
    >>> rho_new = rho.evolve(channel)
    """
    p = compute_strength_from_stimulus(x, strength_max=strength, method=method)
    
    # Build Kraus operators for amplitude damping
    sqrt_p = np.sqrt(p)
    sqrt_one_minus_p = np.sqrt(1.0 - p)
    
    K0 = np.array([[1.0, 0.0],
                   [0.0, sqrt_one_minus_p]], dtype=complex)
    
    K1 = np.array([[0.0, sqrt_p],
                   [0.0, 0.0]], dtype=complex)
    
    return Kraus([K0, K1])


def evidence_phase_damping(
    x: np.ndarray,
    strength: float = 0.4,
    method: str = "mean"
) -> Kraus:
    """
    Create a phase-damping evidence channel from a stimulus.
    
    Phase damping models Z-basis decoherence: it gradually destroys off-diagonal
    coherences without changing populations. The decoherence strength depends on
    the stimulus features.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array.
    strength : float, optional
        Maximum decoherence strength (default: 0.4).
    method : str, optional
        Method for computing strength from features (default: "mean").
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel implementing phase damping.
    
    Notes
    -----
    The phase-damping channel has Kraus operators:
        K0 = √(1-p) * I
        K1 = √p * |0⟩⟨0|
        K2 = √p * |1⟩⟨1|
    
    where p is the decoherence parameter computed from x.
    
    This channel is CPTP and preserves the diagonal elements of the density matrix.
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> channel = evidence_phase_damping(x, strength=0.4)
    >>> rho = DensityMatrix([[0.5, 0.2], [0.2, 0.5]])
    >>> rho_new = rho.evolve(channel)  # Off-diagonal elements reduced
    """
    p = compute_strength_from_stimulus(x, strength_max=strength, method=method)
    
    # Build Kraus operators for phase damping
    sqrt_one_minus_p = np.sqrt(1.0 - p)
    sqrt_p = np.sqrt(p)
    
    K0 = sqrt_one_minus_p * np.eye(2, dtype=complex)
    K1 = sqrt_p * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    K2 = sqrt_p * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    return Kraus([K0, K1, K2])


def evidence_rotation_channel(
    x: np.ndarray,
    strength: float = 0.4,
    method: str = "mean",
    axis: str = "x"
) -> Kraus:
    """
    Create a coherent rotation evidence channel from a stimulus.
    
    This channel applies a unitary rotation (RX or RY) whose angle depends on
    the stimulus features. Unlike damping channels, this is a coherent (unitary)
    update that preserves purity.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array.
    strength : float, optional
        Maximum rotation angle in radians (default: 0.4).
    method : str, optional
        Method for computing strength from features (default: "mean").
    axis : str, optional
        Rotation axis: "x" for RX or "y" for RY (default: "x").
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel implementing a rotation (single Kraus operator).
    
    Notes
    -----
    For a rotation channel, there is a single Kraus operator (the rotation matrix):
        K0 = R_X(θ) or R_Y(θ)
    
    where θ is the rotation angle computed from x.
    
    This channel is unitary (preserves purity) and CPTP.
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> channel = evidence_rotation_channel(x, strength=0.4, axis="x")
    >>> rho = DensityMatrix([[1.0, 0.0], [0.0, 0.0]])
    >>> rho_new = rho.evolve(channel)  # Rotated state
    """
    theta = compute_strength_from_stimulus(x, strength_max=strength, method=method)
    
    if axis.lower() == "x":
        # RX rotation: exp(-i * theta/2 * X)
        cos_theta_2 = np.cos(theta / 2.0)
        sin_theta_2 = np.sin(theta / 2.0)
        K0 = np.array([[cos_theta_2, -1j * sin_theta_2],
                       [-1j * sin_theta_2, cos_theta_2]], dtype=complex)
    elif axis.lower() == "y":
        # RY rotation: exp(-i * theta/2 * Y)
        cos_theta_2 = np.cos(theta / 2.0)
        sin_theta_2 = np.sin(theta / 2.0)
        K0 = np.array([[cos_theta_2, -sin_theta_2],
                       [sin_theta_2, cos_theta_2]], dtype=complex)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")
    
    # Single Kraus operator (unitary channel)
    return Kraus([K0])


def evidence_projective_update(
    x: np.ndarray,
    strength: float = 0.4,
    method: str = "mean"
) -> Kraus:
    """
    Create a Bayesian-like projective update channel from a stimulus.
    
    This channel implements a "soft measurement" that partially projects the
    state toward |1⟩ based on the stimulus. It's inspired by Bayesian updating
    where evidence gradually shifts beliefs.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array.
    strength : float, optional
        Maximum projection strength (default: 0.4).
    method : str, optional
        Method for computing strength from features (default: "mean").
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel implementing a projective update.
    
    Notes
    -----
    The projective update channel has Kraus operators:
        K0 = √(1-p) * I + √p * |1⟩⟨1|
        K1 = √p * |1⟩⟨0|
    
    where p is the projection strength computed from x.
    
    This channel gradually moves the state toward |1⟩, similar to a soft
    measurement outcome. It is CPTP.
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> channel = evidence_projective_update(x, strength=0.4)
    >>> rho = DensityMatrix([[1.0, 0.0], [0.0, 0.0]])  # |0⟩
    >>> rho_new = rho.evolve(channel)  # Partially shifted toward |1⟩
    """
    p = compute_strength_from_stimulus(x, strength_max=strength, method=method)
    
    # Build Kraus operators for projective update
    sqrt_p = np.sqrt(p)
    sqrt_one_minus_p = np.sqrt(1.0 - p)
    
    # K0: identity with partial projection to |1⟩
    K0 = sqrt_one_minus_p * np.eye(2, dtype=complex) + sqrt_p * np.array(
        [[0.0, 0.0], [0.0, 1.0]], dtype=complex
    )
    
    # K1: partial flip from |0⟩ to |1⟩
    K1 = sqrt_p * np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    
    return Kraus([K0, K1])


# ============================================================================
# Channel Embeddings
# ============================================================================

def embed_kraus(
    channel: Kraus,
    n_qubits: int,
    target_qubit: int
) -> Kraus:
    """
    Embed a single-qubit Kraus channel into an n-qubit system.
    
    This function lifts a 1-qubit channel to act on a specific target qubit
    of an n-qubit system using tensor products with identity on other qubits.
    
    Parameters
    ----------
    channel : Kraus
        Single-qubit Kraus channel to embed.
    n_qubits : int
        Total number of qubits in the system.
    target_qubit : int
        Index of the target qubit (0 to n_qubits-1) where the channel acts.
    
    Returns
    -------
    Kraus
        n-qubit Kraus channel that acts on the target qubit and leaves
        other qubits unchanged.
    
    Notes
    -----
    For a single-qubit channel with Kraus operators {K_i}, the embedded
    channel has operators {K_i ⊗ I ⊗ ... ⊗ I} where the identity acts
    on all qubits except the target.
    
    The channel is placed at the target_qubit position using appropriate
    tensor product ordering.
    
    Examples
    --------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> x = np.array([0.3, 0.7])
    >>> channel_1q = evidence_amplitude_damping(x)
    >>> channel_2q = embed_kraus(channel_1q, n_qubits=2, target_qubit=0)
    >>> rho = DensityMatrix(np.eye(4) / 4)  # 2-qubit maximally mixed
    >>> rho_new = rho.evolve(channel_2q, qargs=[0])  # Apply to qubit 0
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if target_qubit < 0 or target_qubit >= n_qubits:
        raise ValueError(
            f"target_qubit must be in [0, {n_qubits-1}], got {target_qubit}"
        )
    
    # Get Kraus operators from the channel
    kraus_ops = channel.data
    
    # Check that input is single-qubit
    if len(kraus_ops) == 0:
        raise ValueError("Channel has no Kraus operators")
    
    K0_shape = kraus_ops[0].shape
    if K0_shape != (2, 2):
        raise ValueError(
            f"Channel must be single-qubit (2x2), got shape {K0_shape}"
        )
    
    # Build embedded Kraus operators using tensor products
    embedded_ops = []
    I_2 = np.eye(2, dtype=complex)
    
    for K in kraus_ops:
        # Build tensor product: I ⊗ ... ⊗ I ⊗ K ⊗ I ⊗ ... ⊗ I
        # where K is at position target_qubit
        
        # Start with K on the target qubit
        embedded_K = K.copy()
        
        # Add identity qubits before target
        for _ in range(target_qubit):
            # Tensor product: I ⊗ embedded_K
            embedded_K = np.kron(I_2, embedded_K)
        
        # Add identity qubits after target
        for _ in range(n_qubits - target_qubit - 1):
            # Tensor product: embedded_K ⊗ I
            embedded_K = np.kron(embedded_K, I_2)
        
        embedded_ops.append(embedded_K)
    
    return Kraus(embedded_ops)


# ============================================================================
# Unified Channel Selector
# ============================================================================

def build_evidence_channel(
    x: np.ndarray,
    kind: str = "amplitude",
    strength: float = 0.4,
    method: str = "mean",
    **kwargs
) -> Kraus:
    """
    Unified function to build evidence channels of different types.
    
    This function dispatches to the appropriate channel constructor based
    on the `kind` parameter. It provides a single interface for creating
    different types of evidence channels.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features as a 1D array.
    kind : str, optional
        Type of channel to create (default: "amplitude"):
        - "amplitude": Amplitude-damping channel
        - "phase": Phase-damping channel
        - "rotation": Coherent rotation channel
        - "projective": Projective update channel
    strength : float, optional
        Maximum channel strength (default: 0.4).
    method : str, optional
        Method for computing strength from features (default: "mean").
    **kwargs
        Additional keyword arguments passed to specific channel constructors:
        - For "rotation": axis ("x" or "y", default: "x")
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel of the specified type.
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> 
    >>> # Amplitude damping (default)
    >>> channel1 = build_evidence_channel(x, kind="amplitude")
    >>> 
    >>> # Phase damping
    >>> channel2 = build_evidence_channel(x, kind="phase")
    >>> 
    >>> # Rotation channel
    >>> channel3 = build_evidence_channel(x, kind="rotation", axis="x")
    >>> 
    >>> # Projective update
    >>> channel4 = build_evidence_channel(x, kind="projective")
    """
    kind = kind.lower()
    
    if kind == "amplitude":
        return evidence_amplitude_damping(x, strength=strength, method=method)
    elif kind == "phase":
        return evidence_phase_damping(x, strength=strength, method=method)
    elif kind == "rotation":
        axis = kwargs.get("axis", "x")
        return evidence_rotation_channel(x, strength=strength, method=method, axis=axis)
    elif kind == "projective":
        return evidence_projective_update(x, strength=strength, method=method)
    else:
        raise ValueError(
            f"Unknown channel kind: {kind}. "
            f"Must be one of: amplitude, phase, rotation, projective"
        )


# ============================================================================
# Backward Compatibility: evidence_kraus
# ============================================================================

def evidence_kraus(x: np.ndarray, strength: float = 0.4) -> Kraus:
    """
    Create a quantum channel from a stimulus that updates the belief state.
    
    This is the original function name maintained for backward compatibility.
    It creates an amplitude-damping channel, which is the default channel type.
    
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
    This function is an alias for evidence_amplitude_damping() for backward
    compatibility. The damping parameter p is computed from the mean of x:
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
    return evidence_amplitude_damping(x, strength=strength, method="mean")


# ============================================================================
# Application Utilities
# ============================================================================

def apply_channel_to_density_matrix(
    rho: DensityMatrix,
    channel: Kraus,
    normalize: bool = True
) -> DensityMatrix:
    """
    Apply a quantum channel to update the density matrix.
    
    This is a convenience wrapper around Qiskit's evolve method with optional
    automatic normalization to ensure trace preservation.
    
    Parameters
    ----------
    rho : DensityMatrix
        The current belief state (density matrix).
    channel : Kraus
        The quantum channel to apply (usually from evidence_kraus or
        build_evidence_channel).
    normalize : bool, optional
        If True, automatically normalize the result if trace ≠ 1 (default: True).
        This helps with numerical stability.
    
    Returns
    -------
    DensityMatrix
        The updated belief state after applying the channel.
    
    Notes
    -----
    Under the hood, this does rho.evolve(channel), which is equivalent to
    the standard quantum channel formula: sum_i K_i @ rho @ K_i^dagger,
    where the K_i are the Kraus operators.
    
    If normalize=True and the trace deviates from 1 by more than 1e-8,
    the result is normalized. This helps maintain numerical stability
    during repeated channel applications.
    
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
    # Apply channel using Qiskit's evolve method
    new_rho = rho.evolve(channel)
    
    # Optional normalization for numerical stability
    if normalize:
        trace = np.trace(new_rho.data)
        trace_tolerance = 1e-8
        
        if abs(trace) > trace_tolerance:
            # Normalize if trace deviates significantly from 1
            if abs(trace - 1.0) > trace_tolerance:
                new_rho = DensityMatrix(new_rho.data / trace)
        else:
            # If trace is too small, something went wrong
            import warnings
            warnings.warn(
                f"Channel application resulted in near-zero trace ({trace:.2e}). "
                f"Result may be invalid.",
                UserWarning
            )
    
    return new_rho


def add_depolarizing_noise(
    channel: Kraus,
    noise_strength: float = 0.01
) -> Kraus:
    """
    Compose a channel with depolarizing noise.
    
    This function adds a small depolarizing channel after the given channel
    to model environmental noise or imperfections.
    
    Parameters
    ----------
    channel : Kraus
        Original channel to add noise to.
    noise_strength : float, optional
        Strength of depolarizing noise (default: 0.01).
        Should be small (typically 0.001 to 0.1).
    
    Returns
    -------
    Kraus
        Composed channel: original channel followed by depolarizing noise.
    
    Notes
    -----
    The depolarizing channel with strength p has Kraus operators:
        K0 = √(1 - 3p/4) * I
        K1 = √(p/4) * X
        K2 = √(p/4) * Y
        K3 = √(p/4) * Z
    
    The composition is computed by multiplying Kraus operators.
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> channel = evidence_amplitude_damping(x)
    >>> noisy_channel = add_depolarizing_noise(channel, noise_strength=0.01)
    """
    if noise_strength <= 0 or noise_strength >= 1:
        raise ValueError(f"noise_strength must be in (0, 1), got {noise_strength}")
    
    # Build depolarizing channel Kraus operators
    sqrt_one_minus_3p_4 = np.sqrt(1.0 - 3.0 * noise_strength / 4.0)
    sqrt_p_4 = np.sqrt(noise_strength / 4.0)
    
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    
    depol_K0 = sqrt_one_minus_3p_4 * I
    depol_K1 = sqrt_p_4 * X
    depol_K2 = sqrt_p_4 * Y
    depol_K3 = sqrt_p_4 * Z
    
    depol_channel = Kraus([depol_K0, depol_K1, depol_K2, depol_K3])
    
    # Compose channels: (depol ∘ original)
    # For composition, we need to compute all products K_depol @ K_orig
    original_ops = channel.data
    composed_ops = []
    
    for K_orig in original_ops:
        for K_depol in depol_channel.data:
            composed_ops.append(K_depol @ K_orig)
    
    return Kraus(composed_ops)


# ============================================================================
# Debug Utilities
# ============================================================================

def debug_channel(channel: Kraus) -> None:
    """
    Print debugging information about a quantum channel.
    
    This utility function helps visualize and verify channel properties,
    useful for notebooks and experiments.
    
    Parameters
    ----------
    channel : Kraus
        The channel to debug.
    
    Examples
    --------
    >>> x = np.array([0.3, 0.7])
    >>> channel = evidence_amplitude_damping(x)
    >>> debug_channel(channel)
    """
    print("=" * 60)
    print("Channel Debug Information")
    print("=" * 60)
    
    # Get Kraus operators
    kraus_ops = channel.data
    n_ops = len(kraus_ops)
    
    print(f"\nNumber of Kraus operators: {n_ops}")
    
    # Print shapes
    print("\nKraus operator shapes:")
    for i, K in enumerate(kraus_ops):
        print(f"  K{i}: {K.shape}")
        if K.shape[0] <= 4:  # Only print small matrices
            print(f"    {K}")
    
    # Check CPTP condition: sum_i K_i^dagger @ K_i = I
    print("\nCPTP Check (sum_i K_i^dagger @ K_i should equal I):")
    identity_check = np.zeros_like(kraus_ops[0], dtype=complex)
    for K in kraus_ops:
        identity_check += K.conj().T @ K
    
    expected_identity = np.eye(identity_check.shape[0], dtype=complex)
    max_deviation = np.max(np.abs(identity_check - expected_identity))
    print(f"  Max deviation from identity: {max_deviation:.2e}")
    
    if max_deviation < 1e-6:
        print("  ✓ Channel is CPTP (within tolerance)")
    else:
        print(f"  ⚠ Channel may not be CPTP (deviation > 1e-6)")
    
    # Check trace preservation on test states
    print("\nTrace Preservation Check:")
    dim = kraus_ops[0].shape[0]
    test_states = [
        np.eye(dim, dtype=complex) / dim,  # Maximally mixed
    ]
    
    if dim == 2:
        # Add pure states for single-qubit
        test_states.append(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex))  # |0⟩
        test_states.append(np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex))  # |1⟩
    
    for i, rho_test in enumerate(test_states):
        rho_dm = DensityMatrix(rho_test)
        rho_evolved = rho_dm.evolve(channel)
        trace_before = np.trace(rho_test)
        trace_after = np.trace(rho_evolved.data)
        trace_diff = abs(trace_after - trace_before)
        print(f"  Test state {i+1}: trace before={trace_before:.6f}, "
              f"after={trace_after:.6f}, diff={trace_diff:.2e}")
    
    print("=" * 60)
