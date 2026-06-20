"""Tests for src/qcore.py: the exact differentiable quantum learner core."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src import qcore


# ---------------------------------------------------------------------------
# 1. Simulator matches Qiskit (mirrors scripts/verify_qcore.py)
# ---------------------------------------------------------------------------

def _build_qiskit(n, ops):
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


def _build_jax(n, ops):
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


def test_simulator_matches_qiskit():
    """qcore statevector == Qiskit Statevector (global-phase + endianness aware)."""
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(12):
        n = int(rng.choice([2, 3]))
        ops = []
        for _ in range(int(rng.integers(5, 15))):
            t = float(rng.uniform(-np.pi, np.pi))
            k = rng.choice(["rx", "rz", "rxx", "ryy", "rzz"])
            if k in ("rx", "rz"):
                ops.append((k, t, int(rng.integers(0, n))))
            else:
                i = int(rng.integers(0, n))
                j = i
                while j == i:
                    j = int(rng.integers(0, n))
                ops.append((k, t, i, j))

        sv_qiskit = Statevector(_build_qiskit(n, ops)).data
        sv_jax = _build_jax(n, ops)
        # qcore is big-endian (qubit 0 = first tensor axis); Qiskit is little-endian.
        sv_jax_le = np.asarray(
            jnp.transpose(sv_jax.reshape((2,) * n), axes=list(range(n))[::-1])
        ).reshape(-1)
        # global-phase-invariant comparison via fidelity
        fid = np.abs(np.vdot(sv_jax_le, sv_qiskit))
        max_err = max(max_err, abs(1.0 - fid))

    assert max_err < 1e-9, f"qcore disagrees with Qiskit: max |1-fid| = {max_err:.2e}"


# ---------------------------------------------------------------------------
# 2. Dynamic (runtime) mask == static discrete mask
# ---------------------------------------------------------------------------

def _small_cfg():
    return qcore.ModelConfig(
        n_qubits=2, depth=3, pairs=[(0, 1)], n_reupload=2,
        readout_paulis=("Z0", "Z1", "X0", "X1", "Z0Z1"),
    )


def test_dynamic_mask_equals_static():
    """make_dynamic_fns predict (runtime mask) == make_loss_and_grad predict (static)."""
    cfg = _small_cfg()
    params = qcore.init_params(cfg, seed=3)
    X = np.random.default_rng(7).uniform(0, 1, size=(11, 2))
    Xj = jnp.asarray(X)

    _, _, dyn_predict = qcore.make_dynamic_fns(cfg)

    rng = np.random.default_rng(123)
    shape = (cfg.depth, cfg.n_edges, 3)
    for _ in range(5):
        mask = rng.integers(0, 2, size=shape).astype(int)
        # static path: rebuilds compiled fns for this specific mask
        _, _, static_predict = qcore.make_loss_and_grad(cfg, mask)
        p_static = np.asarray(static_predict(params, Xj))
        # dynamic path: mask is a runtime float arg
        p_dyn = np.asarray(dyn_predict(params, Xj, jnp.asarray(mask.astype(float))))
        err = float(np.max(np.abs(p_static - p_dyn)))
        assert err < 1e-9, f"dynamic vs static mismatch (mask={mask.tolist()}): {err:.2e}"


# ---------------------------------------------------------------------------
# 3. Training reduces loss
# ---------------------------------------------------------------------------

def test_training_reduces_loss():
    """A handful of Adam steps strictly reduce training loss."""
    cfg = _small_cfg()
    params = qcore.init_params(cfg, seed=1)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(24, 2))
    # learnable target: simple linear-ish boundary
    y = (X[:, 0] + X[:, 1] > 1.0).astype(float)

    loss_fn, vg, _ = qcore.make_dynamic_fns(cfg)
    mask = jnp.asarray(qcore.full_mask(cfg).astype(float))
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    opt = qcore.Adam(params, lr=0.1)
    loss0 = float(loss_fn(params, Xj, yj, mask))
    for _ in range(25):
        _, g = vg(params, Xj, yj, mask)
        params = opt.step(params, g)
    loss1 = float(loss_fn(params, Xj, yj, mask))

    assert loss1 < loss0, f"loss did not decrease: {loss0:.5f} -> {loss1:.5f}"


# ---------------------------------------------------------------------------
# 4. full_mask shape / values
# ---------------------------------------------------------------------------

def test_full_mask_shape():
    cfg = _small_cfg()
    m = qcore.full_mask(cfg)
    assert m.shape == (cfg.depth, cfg.n_edges, 3)
    assert np.all(m == 1)
