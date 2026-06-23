"""Tests for the density-matrix quantum learner in ``src/qdensity.py``.

These verify the physical correctness of the mixed-state forward map (CPTP-ness,
genuine mixedness, the non-unital evidence channel), the statevector-limit
reduction, the static/dynamic mask equivalence, and that the canonical model
learns above chance. Kept tiny so the whole file runs in well under a minute.

``tests/conftest.py`` puts the repo root on sys.path.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

from src import qdensity
from src import qcore


def _density_cfg(prior="mixed", damping=True):
    return qdensity.DensityConfig(
        n_qubits=2,
        depth=4,
        pairs=[(0, 1)],
        n_reupload=2,
        readout_paulis=("Z0", "Z1", "X0", "X1"),
        prior=prior,
        damping=damping,
    )


# ---------------------------------------------------------------------------
# 1. The forward map produces a valid density operator (CPTP image).
# ---------------------------------------------------------------------------
def test_cptp_trace_and_positivity():
    cfg = _density_cfg(prior="mixed", damping=True)
    mask = qdensity.full_mask(cfg)
    rng = np.random.default_rng(0)
    for trial in range(4):
        params = qdensity.init_params(cfg, seed=trial)
        x = rng.uniform(0.0, 1.0, size=(cfg.n_qubits,))
        diag = qdensity.density_diagnostics(params, x, cfg, mask)
        assert abs(diag["trace"] - 1.0) < 1e-9, diag
        assert diag["hermiticity_err"] < 1e-9, diag
        assert diag["min_eig"] > -1e-9, diag  # positive semidefinite


# ---------------------------------------------------------------------------
# 2. The mixed prior yields a genuinely mixed state (purity < 1).
# ---------------------------------------------------------------------------
def test_state_is_genuinely_mixed():
    cfg = _density_cfg(prior="mixed", damping=True)
    mask = qdensity.full_mask(cfg)
    rng = np.random.default_rng(1)
    for trial in range(3):
        params = qdensity.init_params(cfg, seed=trial + 10)
        x = rng.uniform(0.0, 1.0, size=(cfg.n_qubits,))
        diag = qdensity.density_diagnostics(params, x, cfg, mask)
        assert diag["purity"] < 0.999, diag


# ---------------------------------------------------------------------------
# 3. With a pure prior, no damping, and the Luders filter disabled (alpha=0),
#    the density-matrix readout reduces exactly to the qcore statevector model.
# ---------------------------------------------------------------------------
def test_reduces_to_statevector():
    cfg = _density_cfg(prior="pure", damping=False)
    mask = qdensity.full_mask(cfg)
    mt = qdensity._mask_tuple(mask, cfg)

    # Matching qcore statevector config (same gate structure + readout).
    qcfg = qcore.ModelConfig(
        n_qubits=cfg.n_qubits,
        depth=cfg.depth,
        pairs=cfg.pairs,
        n_reupload=cfg.n_reupload,
        feature_scale=cfg.feature_scale,
        readout_paulis=cfg.readout_paulis,
    )
    readout_ops = qcore.build_readout_ops(qcfg)
    all_on = tuple(
        tuple((True, True, True) for _ in range(qcfg.n_edges))
        for _ in range(qcfg.depth)
    )

    rng = np.random.default_rng(2)
    for trial in range(3):
        dparams = qdensity.init_params(cfg, seed=trial + 3)
        # Disable the Luders filter: K_x = exp(0) = identity.
        dparams = dict(dparams)
        dparams["alpha"] = jnp.array(0.0)

        # Copy the shared unitary + readout params into a qcore param dict.
        qparams = qcore.init_params(qcfg, seed=trial + 3)
        qparams = dict(qparams)
        for k in ("sq", "ent", "w", "b"):
            qparams[k] = dparams[k]

        x = rng.uniform(0.0, 1.0, size=(cfg.n_qubits,))

        rho = qdensity.run_density(dparams, jnp.asarray(x), cfg, mt)
        dexp = np.asarray(qdensity._expectations(rho, readout_ops))

        state = qcore.run_state(qparams, jnp.asarray(x), qcfg, all_on)
        sexp = np.asarray(qcore._expectations(state, readout_ops, qcfg.n_qubits))

        assert np.max(np.abs(dexp - sexp)) < 1e-9, (dexp, sexp)


# ---------------------------------------------------------------------------
# 4. The non-unital evidence channel makes the readout depend on x even from
#    the maximally-mixed prior (a unitary alone would leave I/2^n invariant).
# ---------------------------------------------------------------------------
def test_non_unital_breaks_invariance():
    cfg = _density_cfg(prior="mixed", damping=True)
    mask = qdensity.full_mask(cfg)
    mt = qdensity._mask_tuple(mask, cfg)
    readout_ops = qcore.build_readout_ops(cfg)

    params = qdensity.init_params(cfg, seed=7)
    rng = np.random.default_rng(7)
    x1 = rng.uniform(0.0, 1.0, size=(cfg.n_qubits,))
    x2 = rng.uniform(0.0, 1.0, size=(cfg.n_qubits,))

    e1 = np.asarray(qdensity._expectations(
        qdensity.run_density(params, jnp.asarray(x1), cfg, mt), readout_ops))
    e2 = np.asarray(qdensity._expectations(
        qdensity.run_density(params, jnp.asarray(x2), cfg, mt), readout_ops))

    assert np.max(np.abs(e1 - e2)) > 1e-3, (e1, e2)


# ---------------------------------------------------------------------------
# 5. The dynamic (runtime gate_scale) path matches the static (recompiled)
#    path for arbitrary binary masks.
# ---------------------------------------------------------------------------
def test_dynamic_mask_equals_static():
    cfg = _density_cfg(prior="mixed", damping=True)
    params = qdensity.init_params(cfg, seed=5)

    rng = np.random.default_rng(5)
    X = rng.uniform(0.0, 1.0, size=(6, cfg.n_qubits))

    _, _, dyn_predict = qdensity.make_dynamic_fns(cfg)

    for trial in range(3):
        mask = rng.integers(0, 2, size=(cfg.depth, cfg.n_edges, 3))
        _, _, stat_predict = qdensity.make_loss_and_grad(cfg, mask)

        p_static = np.asarray(stat_predict(params, jnp.asarray(X)))
        mask_f = jnp.asarray(mask.astype(float))
        p_dynamic = np.asarray(dyn_predict(params, jnp.asarray(X), mask_f))

        assert np.max(np.abs(p_static - p_dynamic)) < 1e-9, (p_static, p_dynamic)


# ---------------------------------------------------------------------------
# 6. The canonical model learns the easy checkerboard well above chance.
# ---------------------------------------------------------------------------
def test_learns_above_chance():
    from src.data import get_checker_difficulty

    cfg = _density_cfg(prior="pure", damping=True)
    mask = qdensity.full_mask(cfg)

    X, y = get_checker_difficulty("easy", n=160, seed=30)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    Xtr = jnp.asarray(Xtr)
    ytr = jnp.asarray(ytr.astype(float))
    Xte = jnp.asarray(Xte)

    _, vg, predict = qdensity.make_loss_and_grad(cfg, mask)

    params = qdensity.init_params(cfg, seed=0)
    opt = qdensity.Adam(params, lr=0.05)
    for _ in range(400):
        _, grads = vg(params, Xtr, ytr)
        params = opt.step(params, grads)

    acc = qdensity.accuracy(predict, params, Xte, yte)
    assert acc > 0.75, acc
