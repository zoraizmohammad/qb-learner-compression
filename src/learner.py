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
import random
from typing import Tuple, Optional, Dict, Any, List, Union
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit import QuantumCircuit

from .channels import evidence_kraus, apply_channel_to_density_matrix
from .ansatz import build_ansatz, build_parameter_shapes, get_default_pairs
from .transpile_utils import transpile_and_count_2q


# ============================================================================
# Random Seeding Utility
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for numpy and Python's random module.
    
    This ensures reproducibility across experiments by seeding both
    numpy's random number generator and Python's built-in random module.
    
    Parameters
    ----------
    seed : int
        Random seed value.
    
    Examples
    --------
    >>> set_seed(42)
    >>> # All subsequent random operations will be deterministic
    """
    np.random.seed(seed)
    random.seed(seed)


# ============================================================================
# Shape & Parameter Validation
# ============================================================================

def validate_shapes(
    theta: np.ndarray,
    mask: np.ndarray,
    n_qubits: int,
    pairs: List[Tuple[int, int]]
) -> None:
    """
    Validate that theta and mask have correct shapes for given n_qubits and pairs.
    
    This function ensures consistency between parameter arrays and the circuit
    configuration. It validates against n_qubits and n_edges (from pairs) rather
    than using max() which can be ambiguous.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter array to validate.
    mask : np.ndarray
        Mask array to validate.
    n_qubits : int
        Number of qubits in the circuit.
    pairs : List[Tuple[int, int]]
        List of qubit pairs (defines n_edges = len(pairs)).
    
    Raises
    ------
    ValueError
        If shapes are inconsistent with n_qubits and pairs.
    
    Examples
    --------
    >>> theta = np.random.randn(3, 2, 5)
    >>> mask = np.ones((3, 1), dtype=int)
    >>> pairs = [(0, 1)]
    >>> validate_shapes(theta, mask, n_qubits=2, pairs=pairs)
    """
    n_edges = len(pairs)
    expected_theta_shape, expected_mask_shape = build_parameter_shapes(
        n_qubits, theta.shape[0], n_edges
    )
    
    if theta.shape != expected_theta_shape:
        raise ValueError(
            f"theta shape mismatch: expected {expected_theta_shape} "
            f"(for n_qubits={n_qubits}, depth={theta.shape[0]}, n_edges={n_edges}), "
            f"got {theta.shape}"
        )
    
    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"mask shape mismatch: expected {expected_mask_shape} "
            f"(for depth={mask.shape[0]}, n_edges={n_edges}), "
            f"got {mask.shape}"
        )


# ============================================================================
# Circuit Application Utilities
# ============================================================================

def apply_unitary(
    rho: DensityMatrix,
    qc: QuantumCircuit
) -> DensityMatrix:
    """
    Apply a unitary quantum circuit to a density matrix.
    
    This helper function robustly handles multi-qubit states and catches
    cases where the circuit might not be unitary (e.g., if noisy channels
    accidentally appear).
    
    Parameters
    ----------
    rho : DensityMatrix
        Input density matrix (quantum state).
    qc : QuantumCircuit
        Quantum circuit to apply. Should be unitary.
    
    Returns
    -------
    DensityMatrix
        Output density matrix after applying the circuit.
    
    Raises
    ------
    ValueError
        If the circuit dimension doesn't match the state dimension.
    
    Notes
    -----
    For a unitary circuit, the transformation is:
        rho' = U @ rho @ U†
    
    This function uses Qiskit's Operator class to extract the unitary matrix
    and applies it directly. For very large circuits, this may be memory-intensive.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> from qiskit.quantum_info import DensityMatrix
    >>> 
    >>> rho = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])
    >>> qc = QuantumCircuit(1)
    >>> qc.h(0)
    >>> rho_new = apply_unitary(rho, qc)
    """
    # Check dimension compatibility
    state_dim = rho.dim
    circuit_dim = 2 ** qc.num_qubits
    
    if state_dim != circuit_dim:
        raise ValueError(
            f"Dimension mismatch: state has {state_dim} dimensions, "
            f"circuit has {circuit_dim} dimensions (for {qc.num_qubits} qubits)"
        )
    
    try:
        # Extract unitary matrix from circuit
        U = Operator(qc).data
        
        # Check if U is approximately unitary (U @ U† ≈ I)
        U_dagger = U.conj().T
        identity_check = U @ U_dagger
        expected_identity = np.eye(U.shape[0], dtype=complex)
        max_deviation = np.max(np.abs(identity_check - expected_identity))
        
        if max_deviation > 1e-6:
            import warnings
            warnings.warn(
                f"Circuit may not be unitary: max deviation from identity = {max_deviation:.2e}",
                UserWarning
            )
        
        # Apply unitary: rho' = U @ rho @ U†
        rho_new = DensityMatrix(U @ rho.data @ U_dagger)
        
        # Normalize for numerical stability
        trace = np.trace(rho_new.data)
        if abs(trace) > 1e-10:
            rho_new = DensityMatrix(rho_new.data / trace)
        else:
            raise ValueError("Resulting state has zero trace (circuit may not be unitary)")
        
        return rho_new
        
    except Exception as e:
        raise ValueError(
            f"Failed to apply circuit as unitary: {e}. "
            f"Ensure the circuit contains only unitary gates."
        ) from e


# ============================================================================
# Evidence Channel Integration
# ============================================================================

# Alias for compatibility with prompt specification
def build_evidence_channel(x: np.ndarray, strength: float = 0.4):
    """
    Build an evidence channel from a stimulus vector.
    
    This is a convenience wrapper around evidence_kraus() that matches
    the interface specified in the prompt.
    
    Uses projective update channel which pushes toward |1⟩, creating
    better separation between categories compared to amplitude damping.
    
    Parameters
    ----------
    x : np.ndarray
        Stimulus features (1D array).
    strength : float, optional
        Maximum channel strength (default: 0.1).
    
    Returns
    -------
    Kraus
        A Qiskit Kraus channel.
    """
    # Use projective update instead of amplitude damping
    # Projective update pushes toward |1⟩, which creates better separation
    # High feature values → stronger push toward |1⟩ (category B)
    # Low feature values → weaker push (category A stays closer to |0⟩)
    from .channels import build_evidence_channel as build_channel
    return build_channel(x, kind="projective", strength=strength, method="mean")


def apply_evidence_channel(
    rho: DensityMatrix,
    x: np.ndarray,
    strength: float = 0.4,
    qargs: Optional[List[int]] = None
) -> DensityMatrix:
    """
    Apply the evidence channel for stimulus x to update the belief state.
    
    This takes the current quantum state and applies a channel that encodes
    evidence from the stimulus. The channel strength depends on the stimulus
    features. This is the core belief update mechanism in the quantum Bayesian learner.
    
    The channel can be applied to specific qubits via qargs, enabling multi-qubit
    indexing beyond just the first qubit.
    
    Parameters
    ----------
    rho : DensityMatrix
        Current belief state (density matrix).
    x : np.ndarray
        Stimulus features (1D array).
    strength : float, optional
        Maximum channel strength (default: 0.1).
    qargs : List[int], optional
        List of qubit indices to apply the channel to. For single-qubit channels,
        this should be a list with one element. If None, defaults to [0] for
        backward compatibility (default: None).
    
    Returns
    -------
    DensityMatrix
        Updated belief state after applying the evidence channel.
    
    Notes
    -----
    The channel is built using evidence_kraus() from channels.py, which creates
    an amplitude-damping-like channel where the damping parameter depends on
    the stimulus features. For multi-qubit states, we apply it to the specified
    qubits using the qargs parameter.
    
    The resulting state is normalized to ensure trace ≈ 1 for numerical stability.
    
    Examples
    --------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> import numpy as np
    >>> 
    >>> rho = DensityMatrix(np.eye(4) / 4)  # 2-qubit maximally mixed
    >>> x = np.array([0.3, 0.7])
    >>> rho_new = apply_evidence_channel(rho, x, qargs=[0])  # Apply to qubit 0
    """
    channel = build_evidence_channel(x, strength=strength)
    
    # Default to qubit 0 for backward compatibility
    if qargs is None:
        n_qubits = int(np.log2(rho.dim))
        qargs = [0] if n_qubits > 1 else None
    
    # Apply channel to specified qubits
    if qargs is None or len(qargs) == 0:
        # Single-qubit state: apply directly
        new_rho = apply_channel_to_density_matrix(rho, channel)
    elif len(qargs) == 1:
        # Single qubit: apply directly using evolve
        new_rho = rho.evolve(channel, qargs=qargs)
    else:
        # Multiple qubits: apply channel to each qubit individually
        # This is necessary because the channel is single-qubit (2x2 operators)
        # and Qiskit's evolve() expects matching dimensions when qargs has multiple qubits
        new_rho = rho
        for qubit in qargs:
            new_rho = new_rho.evolve(channel, qargs=[qubit])
    
    # Normalize for safety (should already be normalized, but ensure trace ≈ 1)
    trace = np.trace(new_rho.data)
    trace_tolerance = 1e-10
    
    if abs(trace) > trace_tolerance:
        new_rho = DensityMatrix(new_rho.data / trace)
    else:
        # If trace is too small, reinitialize as maximally mixed
        import warnings
        warnings.warn(
            f"Channel application resulted in near-zero trace ({trace:.2e}). "
            f"Reinitializing as maximally mixed state.",
            UserWarning
        )
        n_qubits = int(np.log2(new_rho.dim))
        new_rho = DensityMatrix(np.eye(2**n_qubits, dtype=complex) / (2**n_qubits))
    
    # Final trace check for numerical stability
    final_trace = np.trace(new_rho.data)
    if abs(final_trace - 1.0) > 1e-8:
        # Renormalize one more time
        new_rho = DensityMatrix(new_rho.data / final_trace)
    
    return new_rho


# ============================================================================
# Prediction Utilities
# ============================================================================

def predict_proba(rho: DensityMatrix, readout_alpha: float = 2.0) -> float:
    """
    Predict the probability of class 1 from the quantum state.
    
    This is an alias for predict_label() for clarity. The prediction uses
    a temperature-scaled sigmoid on the Z expectation value.
    
    Parameters
    ----------
    rho : DensityMatrix
        The quantum state (density matrix) to make a prediction from.
    readout_alpha : float, optional
        Temperature scaling factor for sigmoid (default: 2.0).
    
    Returns
    -------
    float
        Probability of class 1 (between 0 and 1).
    
    Examples
    --------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> rho = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])
    >>> prob = predict_proba(rho)
    >>> print(prob)  # ~0.5
    """
    return predict_label(rho, readout_alpha=readout_alpha)


def predict_label(rho, readout_alpha=2.0, use_multi_qubit=True):
    """
    Improved classification readout using Z expectations on all qubits.
    More stable than single-qubit readout and works well even when
    compressed circuits remove entanglers.
    """

    full_rho = rho.data
    n_qubits = int(np.log2(full_rho.shape[0]))

    # Pauli Z and X operators
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    # Helper: compute expectation <P_i> for qubit i
    def expval(P, i):
        # Build full operator: I ⊗ ... ⊗ P (at pos i) ⊗ ... I
        op = 1
        for q in range(n_qubits):
            op = np.kron(op, P if q == i else np.eye(2))
        return float(np.real(np.trace(full_rho @ op)))

    # ------------- NEW FEATURE VECTORS FOR CLASSIFICATION -------------
    # Feature 1: Z on qubit 0
    ez0 = expval(Z, 0)

    if use_multi_qubit:
        ez1 = expval(Z, 1) if n_qubits > 1 else 0.0
        ex0 = expval(X, 0)
        ex1 = expval(X, 1) if n_qubits > 1 else 0.0

        # Combine features linearly (learnable bias optional)
        logit = (
            readout_alpha * (0.6 * ez0 + 0.4 * ez1 + 0.3 * ex0 + 0.2 * ex1)
        )
    else:
        # Fall back to your old method
        logit = readout_alpha * ez0

    # Apply sigmoid
    p = 1 / (1 + np.exp(-logit))
    return float(np.clip(p, 0.0, 1.0))



def predict_hard(rho: DensityMatrix, threshold: float = 0.5) -> int:
    """
    Predict hard class label (0 or 1) from the quantum state.
    
    Parameters
    ----------
    rho : DensityMatrix
        The quantum state (density matrix) to make a prediction from.
    threshold : float, optional
        Classification threshold (default: 0.5). If probability >= threshold,
        predict class 1, otherwise predict class 0.
    
    Returns
    -------
    int
        Predicted class label (0 or 1).
    
    Examples
    --------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> rho = DensityMatrix([[0.2, 0.0], [0.0, 0.8]])
    >>> pred = predict_hard(rho)
    >>> print(pred)  # 1
    """
    prob = predict_proba(rho)
    return 1 if prob >= threshold else 0


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Parameters
    ----------
    preds : np.ndarray
        Predicted labels (0 or 1), shape (n_samples,).
    labels : np.ndarray
        True labels (0 or 1), shape (n_samples,).
    
    Returns
    -------
    float
        Accuracy (between 0 and 1).
    
    Examples
    --------
    >>> preds = np.array([0, 1, 1, 0])
    >>> labels = np.array([0, 1, 0, 0])
    >>> acc = compute_accuracy(preds, labels)
    >>> print(acc)  # 0.5
    """
    if len(preds) != len(labels):
        raise ValueError(
            f"preds and labels must have same length: {len(preds)} != {len(labels)}"
        )
    
    return float(np.mean(preds == labels))


def compute_fidelity(rho1: DensityMatrix, rho2: DensityMatrix) -> float:
    """
    Compute quantum fidelity between two density matrices.
    
    The fidelity F(ρ₁, ρ₂) measures how similar two quantum states are.
    F = 1 means identical states, F = 0 means orthogonal states.
    
    Parameters
    ----------
    rho1 : DensityMatrix
        First quantum state.
    rho2 : DensityMatrix
        Second quantum state.
    
    Returns
    -------
    float
        Fidelity between 0 and 1.
    
    Notes
    -----
    For density matrices, the fidelity is computed as:
        F(ρ₁, ρ₂) = Tr[√(√(ρ₁) ρ₂ √(ρ₁))]
    
    This implementation uses Qiskit's built-in fidelity function if available,
    otherwise falls back to a simplified computation using only numpy.
    
    Examples
    --------
    >>> from qiskit.quantum_info import DensityMatrix
    >>> rho1 = DensityMatrix([[1.0, 0.0], [0.0, 0.0]])  # |0⟩⟨0|
    >>> rho2 = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])  # Maximally mixed
    >>> fid = compute_fidelity(rho1, rho2)
    >>> print(fid)  # 0.707... (1/√2)
    """
    try:
        from qiskit.quantum_info import state_fidelity
        return float(state_fidelity(rho1, rho2))
    except (ImportError, AttributeError):
        # Fallback: use simplified computation with numpy only
        # For pure states or small systems, we can use eigenvalue decomposition
        # F(ρ₁, ρ₂) ≈ Tr[√(ρ₁ @ ρ₂)] for small systems
        try:
            # Try using numpy's matrix square root via eigendecomposition
            # This works for small matrices
            rho1_data = rho1.data
            rho2_data = rho2.data
            
            # For small systems, compute via eigendecomposition
            # F = Tr[√(√(ρ₁) ρ₂ √(ρ₁))]
            # Simplified: use Tr[√(ρ₁ @ ρ₂)] for small systems
            product = rho1_data @ rho2_data
            
            # Compute square root via eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(product)
            # Ensure eigenvalues are non-negative (numerical stability)
            eigenvals = np.maximum(eigenvals, 0.0)
            sqrt_product = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.conj().T
            
            fidelity = np.real(np.trace(sqrt_product))
            
            # Clamp to [0, 1] for numerical stability
            return float(np.clip(fidelity, 0.0, 1.0))
        except Exception:
            # Last resort: return overlap-based approximation
            # F ≈ |Tr[ρ₁ @ ρ₂]| for very simple cases
            overlap = np.abs(np.trace(rho1.data @ rho2.data))
            return float(np.clip(overlap, 0.0, 1.0))


# ============================================================================
# Belief State Initialization
# ============================================================================

def init_belief(n_qubits: int) -> DensityMatrix:
    """
    Initialize the learner's belief state as a non-uniform state to enable learning.
    
    Starting from a maximally mixed state makes the ansatz ineffective because
    maximally mixed states are invariant under unitaries. Instead, we start with
    a slight bias toward |0...0⟩ to break symmetry and allow the ansatz to
    create meaningful transformations.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum state.
    
    Returns
    -------
    DensityMatrix
        Initial state with bias toward |0...0⟩: 0.55|0...0⟩⟨0...0| + 0.45*I/dim
    
    Examples
    --------
    >>> rho = init_belief(2)
    >>> print(rho.data[0, 0])  # Should be > 0.25
    """
    dim = 2 ** n_qubits
    
    # Start with |0...0⟩ state (all zeros) with some mixing
    # This breaks the symmetry and allows learning
    # Reduced bias from 0.6 to 0.55 to make ansatz more effective
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 0.55  # Bias toward |0...0⟩ (reduced from 0.6)
    
    # Add uniform mixing for numerical stability and learnability
    uniform_component = 0.45 / dim  # Increased from 0.4
    rho += uniform_component * np.eye(dim, dtype=complex)
    
    return DensityMatrix(rho)


# ============================================================================
# Loss Function with Enhanced Return Value
# ============================================================================

# Cache for transpiled gate counts (keyed by mask hash)
_transpile_cache: Dict[int, int] = {}


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
    channel_strength: float = 0.1,
    use_cached_ansatz: bool = True,
    qargs: Optional[List[int]] = None,
    readout_alpha: float = 2.0,
    feature_scale: float = 1.0,
) -> Dict[str, Any]:
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
        Maximum strength for evidence channels (default: 0.1).
    use_cached_ansatz : bool, optional
        If True, cache transpiled gate counts by mask hash to speed up training
        (default: True).
    qargs : List[int], optional
        Qubit indices to apply evidence channel to. If None, defaults to [0]
        (default: None).
    readout_alpha : float, optional
        Temperature scaling factor for sigmoid readout (default: 2.0).
        Higher values (1-3) provide stronger nonlinearity.
    feature_scale : float, optional
        Scaling factor for feature encoding rotations (default: 1.0).
        Features are multiplied by this before applying RY/RZ rotations.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - total_loss: float - Total loss (CE + λ * gate_count)
        - ce_loss: float - Cross-entropy loss
        - two_q_cost: int - Two-qubit gate count
        - avg_pred: float - Average predicted probability
        - preds: np.ndarray - Array of predicted probabilities for each sample
    
    Examples
    --------
    >>> import numpy as np
    >>> theta = np.random.randn(3, 2, 5)
    >>> mask = np.ones((3, 1), dtype=int)
    >>> X = np.random.rand(10, 2)
    >>> y = np.array([0, 1] * 5)
    >>> result = forward_loss(theta, mask, X, y, lam=0.1, n_qubits=2, depth=3)
    >>> print(result['total_loss'])
    """
    # Validate shapes
    if pairs is None:
        pairs = get_default_pairs(n_qubits)
    
    validate_shapes(theta, mask, n_qubits, pairs)
    
    # Always build ansatz (needed for applying unitary)
    ansatz = build_ansatz(n_qubits, depth, theta, mask, pairs)
    
    # Compute or retrieve cached two-qubit gate count
    if use_cached_ansatz:
        mask_hash = hash(mask.tobytes())
        if mask_hash in _transpile_cache:
            twoq = _transpile_cache[mask_hash]
        else:
            # Count gates and cache the result
            _, twoq = transpile_and_count_2q(ansatz, backend=backend)
            _transpile_cache[mask_hash] = twoq
    else:
        # Count gates (no caching)
        _, twoq = transpile_and_count_2q(ansatz, backend=backend)
    
    # Compute classification loss and predictions
    ce_loss = 0.0
    eps = 1e-8  # Small epsilon to avoid log(0)
    preds = []
    
    for xi, yi in zip(X, y):
        # Start with maximally mixed state
        rho = init_belief(n_qubits)
        
        # Apply feature encoding: scaled RY and RZ rotations on each qubit
        # This strengthens the feature embedding by amplifying small feature values
        if feature_scale > 0 and len(xi) > 0:
            from qiskit import QuantumCircuit
            feature_qc = QuantumCircuit(n_qubits)
            for q in range(min(n_qubits, len(xi))):
                # Scale features by feature_scale (default: 1.0) to make rotations meaningful
                scaled_feature = feature_scale * xi[q]
                feature_qc.ry(scaled_feature, q)
                feature_qc.rz(scaled_feature, q)
            rho = apply_unitary(rho, feature_qc)
        
        # Apply the ansatz (unitary transformation)
        rho_U = apply_unitary(rho, ansatz)
        
        # Apply evidence channel based on stimulus
        # Apply to all qubits to capture entanglement effects
        if qargs is None:
            qargs = list(range(n_qubits))
        rho_post = apply_evidence_channel(
            rho_U, xi, strength=channel_strength, qargs=qargs
        )
        
        # Make prediction using temperature-scaled sigmoid
        p1 = predict_proba(rho_post, readout_alpha=readout_alpha)
        preds.append(p1)
        
        # Binary cross-entropy
        if yi == 1:
            ce_loss += -np.log(p1 + eps)
        else:
            ce_loss += -np.log(1.0 - p1 + eps)
    
    # Average over samples
    ce_loss /= len(X)
    preds = np.array(preds)
    avg_pred = float(np.mean(preds))
    
    # Total loss: classification + regularization
    total_loss = ce_loss + lam * twoq
    
    return {
        "total_loss": float(total_loss),
        "ce_loss": float(ce_loss),
        "two_q_cost": int(twoq),
        "avg_pred": float(avg_pred),
        "preds": preds,
    }


# ============================================================================
# QuantumBayesianLearner Class
# ============================================================================

class QuantumBayesianLearner:
    """
    Wrapper class for the quantum Bayesian learner.
    
    This class encapsulates the learner's configuration and provides
    a convenient interface for computing losses during training and evaluation.
    
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
        Maximum strength for evidence channels (default: 0.1).
    use_cached_ansatz : bool, optional
        If True, cache transpiled gate counts (default: True).
    qargs : List[int], optional
        Qubit indices to apply evidence channel to (default: None, uses [0]).
    
    Examples
    --------
    >>> learner = QuantumBayesianLearner(n_qubits=2, depth=3, lam=0.1)
    >>> result = learner.loss(theta, mask, X, y)
    >>> print(result['total_loss'])
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        depth: int = 3,
        lam: float = 0.1,
        pairs: Optional[List[Tuple[int, int]]] = None,
        backend: Optional[Any] = None,
        channel_strength: float = 0.1,
        use_cached_ansatz: bool = True,
        qargs: Optional[List[int]] = None,
        readout_alpha: float = 2.0,
        feature_scale: float = 1.0,
    ):
        self.n_qubits = n_qubits
        self.depth = depth
        self.lam = lam
        self.backend = backend
        self.channel_strength = channel_strength
        self.use_cached_ansatz = use_cached_ansatz
        self.qargs = qargs
        self.readout_alpha = readout_alpha
        self.feature_scale = feature_scale
        
        # Default coupling map: all adjacent pairs
        if pairs is None:
            self.pairs = get_default_pairs(n_qubits)
        else:
            self.pairs = pairs
    
    def loss(
        self,
        theta: np.ndarray,
        mask: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
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
        Dict[str, Any]
            Dictionary with loss metrics (see forward_loss for details).
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
            use_cached_ansatz=self.use_cached_ansatz,
            qargs=self.qargs,
            readout_alpha=self.readout_alpha,
            feature_scale=self.feature_scale,
        )
    
    def evaluate(
        self,
        theta: np.ndarray,
        mask: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate the learner on given data.
        
        This method computes accuracy, predictions, and optionally fidelity
        metrics for evaluation purposes.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter tensor.
        mask : np.ndarray
            Binary mask tensor.
        X : np.ndarray
            Evaluation data, shape (n_samples, n_features).
        y : np.ndarray
            True labels, shape (n_samples,).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - accuracy: float - Classification accuracy
            - preds: np.ndarray - Predicted probabilities
            - hard_preds: np.ndarray - Hard predictions (0 or 1)
            - fidelity: Optional[float] - Mean fidelity to ideal states (if applicable)
        
        Examples
        --------
        >>> learner = QuantumBayesianLearner(n_qubits=2, depth=3)
        >>> results = learner.evaluate(theta, mask, X_test, y_test)
        >>> print(f"Accuracy: {results['accuracy']:.3f}")
        """
        # Build ansatz
        ansatz = build_ansatz(self.n_qubits, self.depth, theta, mask, self.pairs)
        
        # Get predictions for all samples
        preds = []
        hard_preds = []
        
        for xi in X:
            # Start with maximally mixed state
            rho = init_belief(self.n_qubits)
            
            # Apply feature encoding: scaled RY and RZ rotations
            if self.feature_scale > 0 and len(xi) > 0:
                from qiskit import QuantumCircuit
                feature_qc = QuantumCircuit(self.n_qubits)
                for q in range(min(self.n_qubits, len(xi))):
                    scaled_feature = self.feature_scale * xi[q]
                    feature_qc.ry(scaled_feature, q)
                    feature_qc.rz(scaled_feature, q)
                rho = apply_unitary(rho, feature_qc)
            
            # Apply ansatz
            rho_U = apply_unitary(rho, ansatz)
            
            # Apply evidence channel
            # Apply to all qubits to capture entanglement effects
            eval_qargs = self.qargs if self.qargs is not None else list(range(self.n_qubits))
            rho_post = apply_evidence_channel(
                rho_U, xi, strength=self.channel_strength, qargs=eval_qargs
            )
            
            # Make predictions using temperature-scaled sigmoid
            prob = predict_proba(rho_post, readout_alpha=self.readout_alpha)
            hard_pred = predict_hard(rho_post)
            
            preds.append(prob)
            hard_preds.append(hard_pred)
        
        preds = np.array(preds)
        hard_preds = np.array(hard_preds)
        
        # Compute accuracy
        accuracy = compute_accuracy(hard_preds, y)
        
        # Compute fidelity (optional: compare to ideal states)
        # For now, we'll skip fidelity computation as it requires defining
        # what the "ideal" state should be for each sample
        fidelity = None
        
        return {
            "accuracy": float(accuracy),
            "preds": preds,
            "hard_preds": hard_preds,
            "fidelity": fidelity,
        }
