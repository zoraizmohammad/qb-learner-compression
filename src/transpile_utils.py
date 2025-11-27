"""
transpile_utils.py

Utilities for transpiling quantum circuits and counting two-qubit gates.

This module handles device-aware compilation, measuring the post-transpile
two-qubit gate count which serves as the compression metric in the quantum
Bayesian learner framework.

The module provides:
- Backend-aware transpilation with support for various Qiskit backends
- Detailed gate counting (entangling vs non-entangling two-qubit gates)
- Hardware-aware optimization for compression experiments
- Comprehensive logging and debugging utilities

Integration:
- ansatz.py: builds circuits that need transpilation
- learner.py: uses gate counts for loss computation
- train_baseline.py, train_compressed.py: evaluate circuit complexity
"""

from __future__ import annotations

import warnings
from typing import Optional, Any, Dict, Tuple, Union
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.providers import Backend
from qiskit.providers.fake_provider import FakeBackend


# ============================================================================
# Entangling Gate Definitions
# ============================================================================

# Set of two-qubit entangling gates (true quantum gates)
ENTANGLING_2Q_GATES = {
    "cx",      # CNOT
    "cz",      # Controlled-Z
    "cy",      # Controlled-Y
    "swap",    # SWAP
    "rxx",     # Rotation XX
    "ryy",     # Rotation YY
    "rzz",     # Rotation ZZ
    "iswap",   # iSWAP
    "dcx",     # Double CNOT
    "ecr",     # Echoed Cross Resonance (IBM)
    "crx",     # Controlled RX
    "cry",     # Controlled RY
    "crz",     # Controlled RZ
    "ch",      # Controlled Hadamard
    "csx",     # Controlled SX
    "cs",      # Controlled S
    "ct",      # Controlled T
}

# Set of all two-qubit gates (including non-entangling like barrier)
ALL_2Q_GATE_NAMES = ENTANGLING_2Q_GATES | {"barrier", "measure"}


# ============================================================================
# Helper Functions
# ============================================================================

def get_gate_histogram(circuit: QuantumCircuit) -> Dict[str, int]:
    """
    Get a histogram of all gates in the circuit.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping gate names to their counts.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> hist = get_gate_histogram(qc)
    >>> print(hist)  # {'h': 1, 'cx': 1}
    """
    return dict(circuit.count_ops())


def count_entangling_gates(circuit: QuantumCircuit) -> int:
    """
    Count only entangling two-qubit gates in the circuit.
    
    Entangling gates are those that can create entanglement between qubits,
    such as CNOT, CZ, RXX, RYY, RZZ, etc. Non-entangling gates like barrier
    are excluded.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze.
    
    Returns
    -------
    int
        Number of entangling two-qubit gates.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.cx(0, 1)
    >>> qc.barrier(0, 1)  # Not counted
    >>> count = count_entangling_gates(qc)
    >>> print(count)  # 1
    """
    histogram = get_gate_histogram(circuit)
    
    entangling_count = 0
    for gate_name, count in histogram.items():
        gate_name_lower = gate_name.lower()
        if gate_name_lower in ENTANGLING_2Q_GATES:
            entangling_count += count
    
    return entangling_count


def count_all_2q_gates(circuit: QuantumCircuit) -> int:
    """
    Count all two-qubit gates (entangling and non-entangling).
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze.
    
    Returns
    -------
    int
        Total number of two-qubit gates.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.cx(0, 1)
    >>> qc.barrier(0, 1)
    >>> count = count_all_2q_gates(qc)
    >>> print(count)  # 2
    """
    histogram = get_gate_histogram(circuit)
    
    total_2q = 0
    for gate_name, count in histogram.items():
        gate_name_lower = gate_name.lower()
        if gate_name_lower in ALL_2Q_GATE_NAMES:
            total_2q += count
        else:
            # Check if it's a two-qubit gate by examining instructions
            # Some custom gates might not be in our list
            for instruction in circuit.data:
                if instruction.operation.name.lower() == gate_name_lower:
                    if len(instruction.qubits) == 2:
                        total_2q += count
                        break
    
    return total_2q


def count_1q_gates(circuit: QuantumCircuit) -> int:
    """
    Count all single-qubit gates in the circuit.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze.
    
    Returns
    -------
    int
        Number of single-qubit gates.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.rx(0.5, 1)
    >>> count = count_1q_gates(qc)
    >>> print(count)  # 2
    """
    histogram = get_gate_histogram(circuit)
    
    one_qubit_count = 0
    for gate_name, count in histogram.items():
        gate_name_lower = gate_name.lower()
        # Single-qubit gates (common ones)
        if gate_name_lower in {
            "h", "x", "y", "z", "s", "t", "sdg", "tdg",
            "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
            "sx", "sxdg", "id", "i", "reset"
        }:
            one_qubit_count += count
        else:
            # Check if it's a single-qubit gate by examining instructions
            for instruction in circuit.data:
                if instruction.operation.name.lower() == gate_name_lower:
                    if len(instruction.qubits) == 1:
                        one_qubit_count += count
                        break
    
    return one_qubit_count


def maybe_get_backend_properties(backend: Optional[Any]) -> Dict[str, Any]:
    """
    Extract backend properties if available.
    
    Parameters
    ----------
    backend : optional
        Qiskit backend to inspect.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with backend properties:
        - name: backend name
        - coupling_map: coupling map if available
        - basis_gates: basis gates if available
        - num_qubits: number of qubits if available
    """
    properties = {
        "name": None,
        "coupling_map": None,
        "basis_gates": None,
        "num_qubits": None,
    }
    
    if backend is None:
        return properties
    
    try:
        # Try to get backend name
        if hasattr(backend, "name"):
            properties["name"] = backend.name
        elif hasattr(backend, "configuration"):
            config = backend.configuration()
            properties["name"] = getattr(config, "backend_name", str(backend))
        
        # Try to get coupling map
        if hasattr(backend, "configuration"):
            config = backend.configuration()
            if hasattr(config, "coupling_map"):
                properties["coupling_map"] = config.coupling_map
            if hasattr(config, "basis_gates"):
                properties["basis_gates"] = config.basis_gates
            if hasattr(config, "n_qubits"):
                properties["num_qubits"] = config.n_qubits
        
        # For newer Qiskit versions
        if hasattr(backend, "coupling_map"):
            properties["coupling_map"] = backend.coupling_map
        if hasattr(backend, "basis_gates"):
            properties["basis_gates"] = backend.basis_gates
        if hasattr(backend, "num_qubits"):
            properties["num_qubits"] = backend.num_qubits
        
    except Exception as e:
        warnings.warn(
            f"Could not extract all backend properties: {e}",
            UserWarning
        )
    
    return properties


def summarize_transpile(
    circuit: QuantumCircuit,
    backend: Optional[Any] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Summarize transpilation results with detailed statistics.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to transpile and analyze.
    backend : optional
        Qiskit backend for transpilation.
    verbose : bool, optional
        If True, print detailed information (default: False).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with transpilation summary:
        - circuit: transpiled circuit
        - n_2q: total two-qubit gate count
        - n_entangling: entangling two-qubit gate count
        - n_1q: one-qubit gate count
        - depth: circuit depth
        - histogram: gate histogram
        - backend_name: backend name if available
    """
    # Transpile if backend provided
    if backend is not None:
        try:
            transpiled = transpile(
                circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=42  # For reproducibility
            )
        except Exception as e:
            warnings.warn(
                f"Transpilation failed: {e}. Using original circuit.",
                UserWarning
            )
            transpiled = circuit
    else:
        transpiled = circuit
    
    # Get backend properties
    backend_props = maybe_get_backend_properties(backend)
    
    # Count gates
    n_2q = count_all_2q_gates(transpiled)
    n_entangling = count_entangling_gates(transpiled)
    n_1q = count_1q_gates(transpiled)
    depth = transpiled.depth()
    histogram = get_gate_histogram(transpiled)
    
    summary = {
        "circuit": transpiled,
        "n_2q": n_2q,
        "n_entangling": n_entangling,
        "n_1q": n_1q,
        "depth": depth,
        "histogram": histogram,
        "backend_name": backend_props["name"],
    }
    
    if verbose:
        print("=" * 60)
        print("Transpilation Summary")
        print("=" * 60)
        print(f"Backend: {backend_props['name'] or 'None (no transpilation)'}")
        if backend_props["coupling_map"]:
            print(f"Coupling map: {backend_props['coupling_map']}")
        if backend_props["basis_gates"]:
            print(f"Basis gates: {backend_props['basis_gates']}")
        print(f"Circuit depth: {depth}")
        print(f"One-qubit gates: {n_1q}")
        print(f"Two-qubit gates (total): {n_2q}")
        print(f"Two-qubit gates (entangling): {n_entangling}")
        print("\nGate histogram:")
        for gate, count in sorted(histogram.items()):
            print(f"  {gate}: {count}")
        print("=" * 60)
    
    return summary


# ============================================================================
# Main Transpilation Function (Enhanced)
# ============================================================================

def transpile_and_count_2q(
    circuit: QuantumCircuit,
    backend: Optional[Any] = None,
    coupling_map: Optional[Any] = None,
    basis_gates: Optional[list] = None,
    optimization_level: int = 1,
    verbose: bool = False,
    return_dict: bool = False
) -> Union[Tuple[QuantumCircuit, int], Dict[str, Any]]:
    """
    Transpile a circuit and count the number of two-qubit gates.
    
    This function handles device-aware compilation for hardware-aware compression
    experiments. It supports various Qiskit backends and provides detailed
    gate counting for compression metrics.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to transpile and count.
    backend : optional
        Qiskit backend to transpile for. Can be:
        - AerSimulator
        - FakeBackend (from qiskit.providers.fake_provider)
        - IBMQ backend
        - None (no transpilation)
    coupling_map : optional
        Manual coupling map to use if backend is None.
        List of qubit pairs, e.g., [[0, 1], [1, 2]].
    basis_gates : optional
        Manual basis gates list if backend is None.
        List of gate names, e.g., ['cx', 'u3', 'id'].
    optimization_level : int, optional
        Transpilation optimization level (0-3, default: 1).
    verbose : bool, optional
        If True, print detailed transpilation information (default: False).
    return_dict : bool, optional
        If True, return detailed dictionary instead of tuple (default: False).
        When False, maintains backward compatibility with original API.
    
    Returns
    -------
    Union[Tuple[QuantumCircuit, int], Dict[str, Any]]
        If return_dict=False (default):
            (transpiled_circuit, two_qubit_gate_count)
        If return_dict=True:
            Dictionary with keys:
            - circuit: transpiled circuit
            - n_2q: total two-qubit gate count
            - n_entangling: entangling two-qubit gate count
            - n_1q: one-qubit gate count
            - depth: circuit depth
            - histogram: gate histogram
    
    Notes
    -----
    The two-qubit gate count is the primary metric for compression experiments.
    This function:
    1. Transpiles the circuit to match hardware constraints (if backend provided)
    2. Counts entangling two-qubit gates (CNOT, CZ, RXX, RYY, RZZ, etc.)
    3. Provides detailed statistics for analysis
    
    The function maintains backward compatibility: by default it returns
    (circuit, count) as before, but can return a detailed dictionary if requested.
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> 
    >>> # Basic usage (backward compatible)
    >>> transpiled, count = transpile_and_count_2q(qc)
    >>> print(count)  # 1
    >>> 
    >>> # Detailed usage
    >>> result = transpile_and_count_2q(qc, return_dict=True)
    >>> print(result['n_entangling'])  # 1
    >>> print(result['n_1q'])  # 1
    >>> 
    >>> # With backend
    >>> from qiskit_aer import AerSimulator
    >>> backend = AerSimulator()
    >>> result = transpile_and_count_2q(qc, backend=backend, verbose=True)
    """
    # Validate backend or use manual coupling map
    transpile_kwargs = {}
    
    if backend is not None:
        # Validate backend
        if not isinstance(backend, (Backend, FakeBackend)):
            # Check if it's a string (backend name) or has backend-like attributes
            if not (hasattr(backend, "configuration") or hasattr(backend, "coupling_map")):
                raise ValueError(
                    f"backend must be a valid Qiskit backend, got {type(backend)}"
                )
        
        transpile_kwargs["backend"] = backend
    elif coupling_map is not None:
        # Use manual coupling map
        transpile_kwargs["coupling_map"] = coupling_map
        if basis_gates is not None:
            transpile_kwargs["basis_gates"] = basis_gates
    
    # Transpile circuit
    if transpile_kwargs:
        try:
            transpiled = transpile(
                circuit,
                optimization_level=optimization_level,
                seed_transpiler=42,  # For reproducibility
                **transpile_kwargs
            )
        except Exception as e:
            warnings.warn(
                f"Transpilation failed: {e}. Using original circuit.",
                UserWarning
            )
            transpiled = circuit
    else:
        # No transpilation
        transpiled = circuit
    
    # Count gates
    n_2q = count_all_2q_gates(transpiled)
    n_entangling = count_entangling_gates(transpiled)
    
    # Return format based on return_dict flag
    if return_dict:
        # Return detailed dictionary
        n_1q = count_1q_gates(transpiled)
        depth = transpiled.depth()
        histogram = get_gate_histogram(transpiled)
        backend_props = maybe_get_backend_properties(backend)
        
        result = {
            "circuit": transpiled,
            "n_2q": n_2q,
            "n_entangling": n_entangling,
            "n_1q": n_1q,
            "depth": depth,
            "histogram": histogram,
        }
        
        if verbose:
            summarize_transpile(circuit, backend=backend, verbose=True)
        
        return result
    else:
        # Return tuple for backward compatibility
        if verbose:
            # Still print verbose info if requested
            backend_props = maybe_get_backend_properties(backend)
            print("=" * 60)
            print("Transpilation Summary")
            print("=" * 60)
            print(f"Backend: {backend_props['name'] or 'None (no transpilation)'}")
            if backend_props["coupling_map"]:
                print(f"Coupling map: {backend_props['coupling_map']}")
            if backend_props["basis_gates"]:
                print(f"Basis gates: {backend_props['basis_gates']}")
            print(f"Circuit depth: {transpiled.depth()}")
            print(f"Two-qubit gates (entangling): {n_entangling}")
            print(f"Two-qubit gates (total): {n_2q}")
            print("=" * 60)
        
        return transpiled, n_entangling  # Return entangling count (more meaningful)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_transpiled_depth(circuit: QuantumCircuit, backend: Optional[Any] = None) -> int:
    """
    Get the depth of a transpiled circuit.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to transpile.
    backend : optional
        Backend for transpilation.
    
    Returns
    -------
    int
        Circuit depth after transpilation.
    """
    if backend is not None:
        transpiled = transpile(circuit, backend=backend, optimization_level=1)
    else:
        transpiled = circuit
    
    return transpiled.depth()


def compare_transpile_results(
    circuit: QuantumCircuit,
    backends: list,
    verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Compare transpilation results across multiple backends.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to transpile.
    backends : list
        List of backends to compare.
    verbose : bool, optional
        If True, print comparison table (default: False).
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping backend names to their transpilation results.
    """
    results = {}
    
    for backend in backends:
        backend_props = maybe_get_backend_properties(backend)
        backend_name = backend_props["name"] or str(backend)
        
        result = summarize_transpile(circuit, backend=backend, verbose=False)
        results[backend_name] = result
    
    if verbose:
        print("=" * 60)
        print("Backend Comparison")
        print("=" * 60)
        print(f"{'Backend':<20} {'Depth':<10} {'2Q Gates':<10} {'1Q Gates':<10}")
        print("-" * 60)
        for name, result in results.items():
            print(
                f"{name:<20} {result['depth']:<10} "
                f"{result['n_entangling']:<10} {result['n_1q']:<10}"
            )
        print("=" * 60)
    
    return results
