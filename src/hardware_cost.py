"""
hardware_cost.py

Real, runtime hardware-aware cost for the quantum Bayesian learner.

This is the genuine "kernel gate": given the ansatz structure (depth, qubit pairs,
entangler type, and the binary mask over Heisenberg interaction terms), we build the
exact Qiskit circuit and TRANSPILE it onto IBM's FakeManila coupling map, then count
the number of two-qubit operations that remain (including any SWAPs the transpiler
inserts to satisfy the coupling map). No hardcoded counts, no backend=None shortcut.

The cost depends only on the circuit STRUCTURE (which entanglers are active), not on
the rotation angle values, so we build with placeholder angles. Results are cached by
the structural key so repeated calls during training are cheap and reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile

# FakeManila: 5-qubit IBM device snapshot (linear coupling map 0-1-2-3-4).
try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2 as _FakeManila
except Exception:  # pragma: no cover - fallback for other qiskit distributions
    from qiskit.providers.fake_provider import FakeManilaV2 as _FakeManila

_BACKEND = _FakeManila()

# Two-qubit gate names we count as hardware 2q operations after transpilation.
_TWO_Q_BASIS = {"cx", "cz", "ecr", "swap", "iswap", "rzz", "rxx", "ryy"}


def build_cost_circuit(
    n_qubits: int,
    depth: int,
    pairs: List[Tuple[int, int]],
    mask: np.ndarray,
    entangler_type: str = "heisenberg",
) -> QuantumCircuit:
    """Build the (angle-agnostic) circuit whose 2q structure we will transpile.

    Mirrors qcore's ansatz: per layer, single-qubit rotations on every qubit, then
    for each active edge the active Heisenberg interaction terms (RXX, RYY, RZZ),
    selected individually by the per-term mask (shape (depth, n_edges, 3)). For the
    HEA variant the mask's first term flags the whole CX-RY-CX block.
    Single-qubit rotation angles are placeholders (they do not affect 2q counts).
    """
    mask = np.asarray(mask, dtype=int)
    qc = QuantumCircuit(n_qubits)
    a = 0.1  # placeholder angle; irrelevant to 2q gate counting
    for d in range(depth):
        for q in range(n_qubits):
            qc.rx(a, q)
            qc.rz(a, q)
        for k, (i, j) in enumerate(pairs):
            if entangler_type.lower() == "heisenberg":
                if mask[d, k, 0]:
                    qc.rxx(a, i, j)
                if mask[d, k, 1]:
                    qc.ryy(a, i, j)
                if mask[d, k, 2]:
                    qc.rzz(a, i, j)
            elif entangler_type.lower() == "hea":
                if mask[d, k, 0]:
                    qc.cx(i, j)
                    qc.ry(a, j)
                    qc.cx(i, j)
            else:
                raise ValueError(f"unknown entangler_type {entangler_type!r}")
    return qc


def _count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


@lru_cache(maxsize=4096)
def _cost_cached(key) -> int:
    n_qubits, depth, pairs, mask_bytes, mask_shape, entangler_type, opt_level, seed = key
    mask = np.frombuffer(mask_bytes, dtype=np.int64).reshape(mask_shape)
    qc = build_cost_circuit(n_qubits, depth, list(pairs), mask, entangler_type)
    tqc = transpile(
        qc,
        backend=_BACKEND,
        optimization_level=opt_level,
        seed_transpiler=seed,
    )
    return _count_2q(tqc)


def transpiled_2q_cost(
    n_qubits: int,
    depth: int,
    pairs: List[Tuple[int, int]],
    mask: np.ndarray,
    entangler_type: str = "heisenberg",
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> int:
    """Post-transpile two-qubit gate count on FakeManila (cached, deterministic)."""
    mask = np.asarray(mask, dtype=np.int64)
    key = (
        int(n_qubits),
        int(depth),
        tuple(tuple(p) for p in pairs),
        mask.tobytes(),
        mask.shape,
        str(entangler_type).lower(),
        int(optimization_level),
        int(seed_transpiler),
    )
    return _cost_cached(key)


if __name__ == "__main__":
    pairs = [(0, 1)]
    depth = 3
    # Sweep total active interaction terms 0..9 (depth*1edge*3 terms) for the
    # finer-grained Heisenberg frontier.
    print("Heisenberg per-term frontier (depth=3, 1 edge, 9 possible terms):")
    for active in range(9, -1, -1):
        flat = np.zeros(depth * 3, dtype=int)
        flat[:active] = 1
        m = flat.reshape(depth, 1, 3)
        n2q = transpiled_2q_cost(2, depth, pairs, m, "heisenberg")
        print(f"  active_terms={active}  N2q={n2q}")
