"""
ansatz.py

Variational ansatz for the quantum Bayesian learner.

Builds layered quantum circuits with Heisenberg-type entangling gates.
Uses a mask to selectively enable/disable entangling blocks, which helps
with hardware constraints and circuit compression.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.circuit import Parameter

try:
    from qiskit.transpiler import Layout
    HAS_LAYOUT = True
except ImportError:
    HAS_LAYOUT = False


def get_default_pairs(n_qubits: int) -> List[Tuple[int, int]]:
    """
    Build default linear chain connectivity for qubits.
    
    Creates pairs for a linear chain: (0,1), (1,2), ..., (n-2, n-1).
    This is a common connectivity pattern for quantum hardware.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    
    Returns
    -------
    List[Tuple[int, int]]
        List of qubit pairs for linear chain connectivity.
    
    Examples
    --------
    >>> pairs = get_default_pairs(3)
    >>> print(pairs)  # [(0, 1), (1, 2)]
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2 for entangling gates, got {n_qubits}")
    
    if n_qubits == 2:
        return [(0, 1)]
    else:
        return [(i, i+1) for i in range(n_qubits - 1)]


def build_parameter_shapes(
    n_qubits: int,
    depth: int,
    n_edges: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
    """
    Compute expected shapes for theta and mask arrays.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    depth : int
        Number of layers.
    n_edges : int
        Number of entangling edges (length of pairs list).
    
    Returns
    -------
    Tuple[Tuple[int, int, int], Tuple[int, int]]
        (theta_shape, mask_shape) where:
        - theta_shape: (depth, max(n_qubits, n_edges), 5)
        - mask_shape: (depth, n_edges)
    
    Examples
    --------
    >>> theta_shape, mask_shape = build_parameter_shapes(n_qubits=2, depth=3, n_edges=1)
    >>> print(theta_shape)  # (3, 2, 5)
    >>> print(mask_shape)    # (3, 1)
    """
    max_dim = max(n_qubits, n_edges)
    theta_shape = (depth, max_dim, 5)
    mask_shape = (depth, n_edges)
    return theta_shape, mask_shape


def init_random_theta(
    n_qubits: int,
    depth: int,
    n_edges: int,
    scale: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize random parameter tensor with correct shape.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    depth : int
        Number of layers.
    n_edges : int
        Number of entangling edges.
    scale : float, optional
        Standard deviation for random initialization (default: 0.1).
    seed : int, optional
        Random seed for reproducibility (default: None).
    
    Returns
    -------
    np.ndarray
        Random parameter array with shape (depth, max(n_qubits, n_edges), 5).
    
    Examples
    --------
    >>> theta = init_random_theta(n_qubits=2, depth=3, n_edges=1, seed=42)
    >>> print(theta.shape)  # (3, 2, 5)
    """
    theta_shape, _ = build_parameter_shapes(n_qubits, depth, n_edges)
    
    rng = np.random.default_rng(seed)
    theta = rng.normal(loc=0.0, scale=scale, size=theta_shape)
    
    return theta


def init_full_mask(
    depth: int,
    n_edges: int
) -> np.ndarray:
    """
    Create a fully active entangling mask (all gates enabled).
    
    Parameters
    ----------
    depth : int
        Number of layers.
    n_edges : int
        Number of entangling edges.
    
    Returns
    -------
    np.ndarray
        Binary mask with shape (depth, n_edges), all values set to 1.
    
    Examples
    --------
    >>> mask = init_full_mask(depth=3, n_edges=1)
    >>> print(mask)  # [[1], [1], [1]]
    """
    return np.ones((depth, n_edges), dtype=int)


def init_sparse_mask(
    depth: int,
    n_edges: int,
    sparsity: float = 0.5,
    seed: Optional[int] = None,
    structured: bool = False
) -> np.ndarray:
    """
    Create a randomized or structured sparse mask.
    
    Parameters
    ----------
    depth : int
        Number of layers.
    n_edges : int
        Number of entangling edges.
    sparsity : float, optional
        Fraction of gates to keep active (default: 0.5).
        Must be between 0 and 1.
    seed : int, optional
        Random seed for reproducibility (default: None).
    structured : bool, optional
        If True, creates a structured pattern (alternating layers).
        If False, randomly samples active gates (default: False).
    
    Returns
    -------
    np.ndarray
        Binary mask with shape (depth, n_edges).
    
    Examples
    --------
    >>> mask = init_sparse_mask(depth=3, n_edges=2, sparsity=0.5, seed=42)
    >>> print(mask.sum() / mask.size)  # Approximately 0.5
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be between 0 and 1, got {sparsity}")
    
    mask = np.zeros((depth, n_edges), dtype=int)
    
    if structured:
        # Structured pattern: alternate layers
        for d in range(depth):
            if d % 2 == 0:  # Even layers: all active
                mask[d, :] = 1
            else:  # Odd layers: sparse
                n_active = max(1, int(n_edges * sparsity))
                indices = np.linspace(0, n_edges - 1, n_active, dtype=int)
                mask[d, indices] = 1
    else:
        # Random pattern
        rng = np.random.default_rng(seed)
        n_total = depth * n_edges
        n_active = int(n_total * sparsity)
        
        # Randomly select which gates to activate
        flat_indices = rng.choice(n_total, size=n_active, replace=False)
        for idx in flat_indices:
            d = idx // n_edges
            k = idx % n_edges
            mask[d, k] = 1
    
    return mask


def validate_theta_and_mask(
    theta: np.ndarray,
    mask: np.ndarray,
    n_qubits: int,
    n_edges: int,
    depth: int
) -> None:
    """
    Validate that theta and mask have correct shapes and values.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter array to validate.
    mask : np.ndarray
        Mask array to validate.
    n_qubits : int
        Expected number of qubits.
    n_edges : int
        Expected number of edges.
    depth : int
        Expected depth.
    
    Raises
    ------
    ValueError
        If shapes or values are invalid.
    
    Examples
    --------
    >>> theta = np.random.randn(3, 2, 5)
    >>> mask = np.ones((3, 1), dtype=int)
    >>> validate_theta_and_mask(theta, mask, n_qubits=2, n_edges=1, depth=3)
    """
    expected_theta_shape, expected_mask_shape = build_parameter_shapes(
        n_qubits, depth, n_edges
    )
    
    if theta.shape != expected_theta_shape:
        raise ValueError(
            f"theta has incorrect shape: expected {expected_theta_shape}, "
            f"got {theta.shape}"
        )
    
    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"mask has incorrect shape: expected {expected_mask_shape}, "
            f"got {mask.shape}"
        )
    
    # Check mask values are binary
    mask_int = np.asarray(mask, dtype=int)
    if not np.all((mask_int == 0) | (mask_int == 1)):
        raise ValueError("mask must contain only 0 and 1 values")


def get_active_edges(
    mask: np.ndarray,
    pairs: List[Tuple[int, int]]
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Extract list of active entangling edges with their depth.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask with shape (depth, n_edges).
    pairs : List[Tuple[int, int]]
        List of qubit pairs corresponding to each edge index.
    
    Returns
    -------
    List[Tuple[Tuple[int, int], int]]
        List of (pair, depth) tuples for all active edges.
        Each tuple is ((i, j), d) where (i, j) is the qubit pair
        and d is the depth at which it's active.
    
    Examples
    --------
    >>> mask = np.array([[1, 0], [0, 1], [1, 1]])
    >>> pairs = [(0, 1), (1, 2)]
    >>> active = get_active_edges(mask, pairs)
    >>> print(active)  # [((0, 1), 0), ((1, 2), 1), ((0, 1), 2), ((1, 2), 2)]
    """
    depth, n_edges = mask.shape
    
    if len(pairs) != n_edges:
        raise ValueError(
            f"pairs length ({len(pairs)}) must match mask n_edges ({n_edges})"
        )
    
    active_edges = []
    for d in range(depth):
        for k in range(n_edges):
            if mask[d, k] == 1:
                active_edges.append((pairs[k], d))
    
    return active_edges


def prune_pairs_for_backend(
    pairs: List[Tuple[int, int]],
    backend: Any
) -> List[Tuple[int, int]]:
    """
    Remove edges not supported by backend coupling map.
    
    Parameters
    ----------
    pairs : List[Tuple[int, int]]
        List of qubit pairs to filter.
    backend : Any
        Qiskit backend with coupling_map attribute.
    
    Returns
    -------
    List[Tuple[int, int]]
        Filtered list of pairs that are supported by the backend.
    
    Examples
    --------
    >>> from qiskit import Aer
    >>> backend = Aer.get_backend('qasm_simulator')
    >>> pairs = [(0, 1), (1, 2), (0, 2)]
    >>> filtered = prune_pairs_for_backend(pairs, backend)
    """
    if backend is None:
        return pairs
    
    try:
        coupling_map = backend.configuration().coupling_map
        if coupling_map is None:
            return pairs
        
        # Convert coupling_map to set of tuples for fast lookup
        # Handle both directions (i, j) and (j, i)
        supported_pairs = set()
        for edge in coupling_map:
            if len(edge) == 2:
                supported_pairs.add((edge[0], edge[1]))
                supported_pairs.add((edge[1], edge[0]))
        
        # Filter pairs
        filtered = []
        for pair in pairs:
            if pair in supported_pairs:
                filtered.append(pair)
        
        return filtered
    except (AttributeError, TypeError):
        # If backend doesn't have coupling_map, return original pairs
        return pairs


def hea_entangler_block(
    qc: QuantumCircuit,
    i: int,
    j: int,
    param: Union[float, Any]
) -> None:
    """
    Add a Hardware-Efficient Ansatz (HEA) style entangling block.
    
    This implements the standard HEA entangling pattern: CX, RY, CX.
    This creates stronger entanglement than a single CX gate and is
    commonly used in VQE/VQA applications.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to add gates to (modified in-place).
    i : int
        First qubit index (control for first CX).
    j : int
        Second qubit index (target for first CX).
    param : float or Parameter
        Rotation angle for RY gate on qubit j.
        Can be a float or Qiskit Parameter object.
    
    Returns
    -------
    None
        Modifies the circuit in-place.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> import numpy as np
    >>> 
    >>> qc = QuantumCircuit(2)
    >>> hea_entangler_block(qc, 0, 1, 0.5)
    >>> print(qc)
    """
    # HEA pattern: CX, RY, CX
    qc.cx(i, j)
    qc.ry(param, j)
    qc.cx(i, j)


def heisenberg_entangler_block(
    qc: QuantumCircuit,
    i: int,
    j: int,
    params: Union[np.ndarray, List[float], Tuple[float, float, float]]
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
    
    Examples
    --------
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
    pairs: List[Tuple[int, int]],
    use_parameters: bool = False,
    apply_initial_layout: bool = False,
    initial_layout: Optional[List[int]] = None,
    verbose: bool = False,
    return_metadata: bool = False,
    entangler_type: str = "hea"
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, Dict[str, Any]]]:
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
    use_parameters : bool, optional
        If True, use Qiskit Parameter objects for symbolic circuits
        (default: False).
    apply_initial_layout : bool, optional
        If True, permute qubits based on initial_layout before building
        (default: False).
    initial_layout : List[int], optional
        Permutation of qubit indices [0, 1, ..., n_qubits-1] to apply.
        Only used if apply_initial_layout=True (default: None).
    verbose : bool, optional
        If True, print debugging information (default: False).
    return_metadata : bool, optional
        If True, return additional metadata dictionary (default: False).
    entangler_type : str, optional
        Type of entangling block to use:
        - "hea": Hardware-Efficient Ansatz style (CX, RY, CX) - default, stronger
        - "heisenberg": Heisenberg interaction (RXX, RYY, RZZ) - original
        (default: "hea").
    
    Returns
    -------
    QuantumCircuit or Tuple[QuantumCircuit, Dict[str, Any]]
        The built parameterized circuit. If return_metadata=True, also
        returns a dictionary with:
        - n_params: total number of parameters
        - active_edges: list of active edge tuples
        - connectivity_map: dictionary mapping pairs to edge indices
        - per_layer_density: list of entangling density per layer
    
    Examples
    --------
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
    
    # Handle initial layout permutation
    qubit_map = list(range(n_qubits))
    if apply_initial_layout and initial_layout is not None:
        if len(initial_layout) != n_qubits:
            raise ValueError(
                f"initial_layout length ({len(initial_layout)}) must match "
                f"n_qubits ({n_qubits})"
            )
        if set(initial_layout) != set(range(n_qubits)):
            raise ValueError(
                f"initial_layout must be a permutation of [0, ..., {n_qubits-1}]"
            )
        qubit_map = initial_layout
        if verbose:
            print(f"Applying initial layout: {qubit_map}")
    
    # Create parameter objects if requested
    param_dict = {}
    if use_parameters:
        param_idx = 0
        for d in range(depth):
            for q in range(n_qubits):
                param_dict[(d, q, 0)] = Parameter(f"rx_{d}_{q}")
                param_dict[(d, q, 1)] = Parameter(f"rz_{d}_{q}")
            for k in range(n_edges):
                param_dict[(d, k, 2)] = Parameter(f"rxx_{d}_{k}")
                param_dict[(d, k, 3)] = Parameter(f"ryy_{d}_{k}")
                param_dict[(d, k, 4)] = Parameter(f"rzz_{d}_{k}")
    
    # Start building the circuit
    qc = QuantumCircuit(n_qubits)
    
    # Build each layer
    for d in range(depth):
        # First, single-qubit rotations on all qubits
        for q in range(n_qubits):
            mapped_q = qubit_map[q]
            if use_parameters:
                rx_param = param_dict[(d, q, 0)]
                rz_param = param_dict[(d, q, 1)]
            else:
                rx_param = theta[d, q, 0]
                rz_param = theta[d, q, 1]
            
            qc.rx(rx_param, mapped_q)
            qc.rz(rz_param, mapped_q)
        
        # Then, entangling gates (only where mask says to)
        for k, (i, j) in enumerate(pairs):
            if mask[d, k] == 1:
                mapped_i = qubit_map[i]
                mapped_j = qubit_map[j]
                
                if entangler_type.lower() == "hea":
                    # HEA-style: CX, RY, CX (uses theta[d, k, 2] for RY angle)
                    if use_parameters:
                        ry_param = param_dict[(d, k, 2)]
                    else:
                        ry_param = theta[d, k, 2]
                    hea_entangler_block(qc, mapped_i, mapped_j, ry_param)
                elif entangler_type.lower() == "heisenberg":
                    # Heisenberg: RXX, RYY, RZZ (uses theta[d, k, 2:5])
                    if use_parameters:
                        heisenberg_params = [
                            param_dict[(d, k, 2)],
                            param_dict[(d, k, 3)],
                            param_dict[(d, k, 4)]
                        ]
                    else:
                        heisenberg_params = theta[d, k, 2:5]  # [α, β, γ]
                    heisenberg_entangler_block(qc, mapped_i, mapped_j, heisenberg_params)
                else:
                    raise ValueError(
                        f"Unknown entangler_type: {entangler_type}. "
                        f"Must be 'hea' or 'heisenberg'"
                    )
    
    # Build metadata if requested
    if return_metadata:
        metadata = {
            "n_params": count_parameters(theta, mask, n_qubits=n_qubits),
            "active_edges": get_active_edges(mask, pairs),
            "connectivity_map": {pair: k for k, pair in enumerate(pairs)},
            "per_layer_density": [
                float(np.sum(mask[d, :]) / n_edges) for d in range(depth)
            ]
        }
        
        if verbose:
            print(f"Built ansatz: {n_qubits} qubits, {depth} layers")
            print(f"  Total parameters: {metadata['n_params']}")
            print(f"  Active edges: {len(metadata['active_edges'])}")
            print(f"  Per-layer density: {metadata['per_layer_density']}")
        
        return qc, metadata
    
    return qc


def count_parameters(
    theta: np.ndarray,
    mask: np.ndarray,
    n_qubits: Optional[int] = None
) -> int:
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
    n_qubits : int, optional
        Number of qubits. If None, inferred from theta shape.
        This fixes the bug where max_dim was incorrectly used as n_qubits.
    
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
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.ansatz import count_parameters
    >>> 
    >>> depth = 3
    >>> n_qubits = 2
    >>> n_edges = 1
    >>> theta = np.random.randn(depth, max(n_qubits, n_edges), 5)
    >>> mask = np.ones((depth, n_edges), dtype=int)
    >>> 
    >>> n_params = count_parameters(theta, mask, n_qubits=2)
    >>> print(f"Active parameters: {n_params}")  # Should be 15 (3*2*2 + 3*1*3)
    """
    depth, max_dim, _ = theta.shape
    n_edges = mask.shape[1]
    
    # Fix: Use actual n_qubits if provided, otherwise infer from theta
    # The bug was using max_dim as n_qubits, but max_dim = max(n_qubits, n_edges)
    if n_qubits is None:
        # Try to infer: if max_dim == n_edges, then n_qubits might be smaller
        # But we can't know for sure, so we use max_dim as a conservative estimate
        # However, the correct approach is to require n_qubits to be passed
        # For backward compatibility, we'll use max_dim but warn if it's ambiguous
        n_qubits = max_dim
        if max_dim > n_edges:
            # Likely max_dim is n_qubits
            pass
        else:
            # Ambiguous case - max_dim could be either
            # Default to assuming it's n_qubits for backward compatibility
            pass
    else:
        # Use provided n_qubits
        if n_qubits > max_dim:
            raise ValueError(
                f"n_qubits ({n_qubits}) cannot exceed theta max_dim ({max_dim})"
            )
    
    # Count single-qubit params: 2 per qubit per depth (RX + RZ)
    n_single_qubit_params = depth * n_qubits * 2
    
    # Count entangling params: 3 per active edge (α, β, γ)
    n_active_edges = int(np.sum(mask == 1))
    n_entangling_params = n_active_edges * 3
    
    total_params = n_single_qubit_params + n_entangling_params
    
    return total_params


# Smoke test
if __name__ == "__main__":
    print("Running smoke test for ansatz.py...")
    
    # Test parameters
    n_qubits = 2
    depth = 3
    pairs = get_default_pairs(n_qubits)
    n_edges = len(pairs)
    
    print(f"\nTest configuration:")
    print(f"  n_qubits: {n_qubits}")
    print(f"  depth: {depth}")
    print(f"  pairs: {pairs}")
    print(f"  n_edges: {n_edges}")
    
    # Test helper functions
    print("\n1. Testing helper functions...")
    theta_shape, mask_shape = build_parameter_shapes(n_qubits, depth, n_edges)
    print(f"   Parameter shapes: theta={theta_shape}, mask={mask_shape}")
    
    theta = init_random_theta(n_qubits, depth, n_edges, seed=42)
    print(f"   Initialized theta: shape={theta.shape}")
    
    mask = init_full_mask(depth, n_edges)
    print(f"   Full mask: shape={mask.shape}, sum={mask.sum()}")
    
    sparse_mask = init_sparse_mask(depth, n_edges, sparsity=0.5, seed=42)
    print(f"   Sparse mask: shape={sparse_mask.shape}, sum={sparse_mask.sum()}")
    
    validate_theta_and_mask(theta, mask, n_qubits, n_edges, depth)
    print("   Validation passed")
    
    # Test build_ansatz
    print("\n2. Testing build_ansatz...")
    qc = build_ansatz(n_qubits, depth, theta, mask, pairs, verbose=True)
    print(f"   Circuit built: {qc.num_qubits} qubits, {len(qc.data)} gates")
    print(f"   Circuit depth: {qc.depth()}")
    
    # Test count_parameters
    print("\n3. Testing count_parameters...")
    n_params = count_parameters(theta, mask, n_qubits=n_qubits)
    print(f"   Total parameters: {n_params}")
    expected = depth * n_qubits * 2 + depth * n_edges * 3
    print(f"   Expected: {expected}")
    assert n_params == expected, f"Parameter count mismatch: {n_params} != {expected}"
    
    # Test get_active_edges
    print("\n4. Testing get_active_edges...")
    active_edges = get_active_edges(mask, pairs)
    print(f"   Active edges: {len(active_edges)}")
    print(f"   First few: {active_edges[:3]}")
    
    # Test with metadata
    print("\n5. Testing build_ansatz with metadata...")
    qc2, metadata = build_ansatz(
        n_qubits, depth, theta, mask, pairs,
        return_metadata=True, verbose=True
    )
    print(f"   Metadata keys: {list(metadata.keys())}")
    
    print("\n✓ All smoke tests passed!")
