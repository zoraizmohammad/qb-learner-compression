"""
verify_qcore.py

Bit-for-bit verification that src/qcore.py's statevector simulator agrees with
Qiskit for a random circuit using the same gate set (RX, RZ, RXX, RYY, RZZ).
This guards against gate-convention / qubit-ordering bugs in the JAX core.
"""
import numpy as np
import jax.numpy as jnp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore


def build_qiskit(n, ops):
    qc = QuantumCircuit(n)
    for op in ops:
        kind = op[0]
        if kind == "rx":
            qc.rx(op[1], op[2])
        elif kind == "rz":
            qc.rz(op[1], op[2])
        elif kind == "rxx":
            qc.rxx(op[1], op[2], op[3])
        elif kind == "ryy":
            qc.ryy(op[1], op[2], op[3])
        elif kind == "rzz":
            qc.rzz(op[1], op[2], op[3])
    return qc


def build_jax(n, ops):
    state = jnp.zeros((2,) * n, dtype=jnp.complex128).at[(0,) * n].set(1.0 + 0j)
    for op in ops:
        kind = op[0]
        if kind == "rx":
            state = qcore._apply_1q(state, qcore._rx(op[1]), op[2], n)
        elif kind == "rz":
            state = qcore._apply_1q(state, qcore._rz(op[1]), op[2], n)
        elif kind == "rxx":
            state = qcore._apply_2q(state, qcore._rpp(op[1], qcore._X), op[2], op[3], n)
        elif kind == "ryy":
            state = qcore._apply_2q(state, qcore._rpp(op[1], qcore._Y), op[2], op[3], n)
        elif kind == "rzz":
            state = qcore._apply_2q(state, qcore._rpp(op[1], qcore._Z), op[2], op[3], n)
    return np.asarray(state).reshape(-1)


def main():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for trial in range(20):
        n = rng.choice([2, 3])
        ops = []
        for _ in range(rng.integers(5, 15)):
            t = float(rng.uniform(-np.pi, np.pi))
            k = rng.choice(["rx", "rz", "rxx", "ryy", "rzz"])
            if k in ("rx", "rz"):
                ops.append((k, t, int(rng.integers(0, n))))
            else:
                i = int(rng.integers(0, n)); j = i
                while j == i:
                    j = int(rng.integers(0, n))
                ops.append((k, t, i, j))
        sv_qiskit = Statevector(build_qiskit(n, ops)).data
        sv_jax = build_jax(n, ops)
        # Qiskit uses little-endian qubit ordering; our tensor uses qubit 0 as the
        # first tensor axis (big-endian). Compare via global-phase-invariant fidelity
        # after reindexing.
        # Reindex jax (axis q = qubit q, q=0 first) to qiskit little-endian:
        sv_jax_le = np.asarray(jnp.transpose(sv_jax.reshape((2,) * n),
                                             axes=list(range(n))[::-1])).reshape(-1)
        fid = np.abs(np.vdot(sv_jax_le, sv_qiskit))
        err = abs(1.0 - fid)
        max_err = max(max_err, err)
    print(f"max |1 - fidelity| over 20 random circuits = {max_err:.2e}")
    assert max_err < 1e-9, "qcore simulator disagrees with Qiskit!"
    print("PASS: qcore statevector matches Qiskit (up to global phase / endianness).")


if __name__ == "__main__":
    main()
