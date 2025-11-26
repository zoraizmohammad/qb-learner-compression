"""
transpile_utils.py

Utilities for transpiling quantum circuits and counting two-qubit gates.

This module handles device-aware compilation, measuring the post-transpile
two-qubit gate count which serves as the compression metric.
"""

from __future__ import annotations

from typing import Optional, Any
from qiskit import QuantumCircuit
from qiskit import transpile


def transpile_and_count_2q(
    circuit: QuantumCircuit,
    backend: Optional[Any] = None
) -> tuple[QuantumCircuit, int]:
    """
    Transpile a circuit and count the number of two-qubit gates.
    
    If a backend is provided, transpiles to that backend's coupling map.
    Otherwise, just counts gates in the original circuit.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to transpile and count.
    backend : optional
        Qiskit backend to transpile for. If None, just counts gates
        in the original circuit.
    
    Returns
    -------
    tuple[QuantumCircuit, int]
        (transpiled_circuit, two_qubit_gate_count)
    """
    if backend is not None:
        # Transpile to the backend's coupling map
        transpiled = transpile(circuit, backend=backend, optimization_level=1)
    else:
        # No backend, just use the original circuit
        transpiled = circuit
    
    # Count two-qubit gates (CNOT, RXX, RYY, RZZ, etc.)
    two_qubit_count = 0
    for instruction in transpiled.data:
        if len(instruction.qubits) == 2:
            two_qubit_count += 1
    
    return transpiled, two_qubit_count

