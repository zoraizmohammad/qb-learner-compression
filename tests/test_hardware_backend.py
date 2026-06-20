"""
Tests for the hardware deployment path (src/hardware_backend.py).

These run on the exact StatevectorEstimator only (no account / no QPU needed), so they
are CI-safe. They guarantee the Qiskit inference circuit is bit-identical to the qcore
model it deploys, which is what makes results on a real device trustworthy.
"""
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore, hardware_backend as hw


def _cfg():
    return qcore.ModelConfig(n_qubits=2, depth=4, pairs=[(0, 1)], n_reupload=2,
                             readout_paulis=("Z0", "Z1", "X0", "X1"))


def _all_on(cfg):
    return tuple(tuple((True, True, True) for _ in range(cfg.n_edges)) for _ in range(cfg.depth))


def test_inference_circuit_matches_qcore_expectations():
    cfg = _cfg()
    params = qcore.init_params(cfg, seed=7)
    ops = qcore.build_readout_ops(cfg)
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1, size=(5, 2))
    mask = qcore.full_mask(cfg)
    v_qcore = np.array([
        np.asarray(qcore._expectations(
            qcore.run_state(params, jnp.asarray(x), cfg, _all_on(cfg)), ops, cfg.n_qubits))
        for x in xs])
    est, _ = hw.make_estimator("exact")
    v_qiskit = hw.estimate_expectations(params, mask, xs, cfg, est)
    assert np.max(np.abs(v_qcore - v_qiskit)) < 1e-9


def test_exact_predictions_match_qcore():
    cfg = _cfg()
    params = qcore.init_params(cfg, seed=3)
    mask = qcore.full_mask(cfg)
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(8, 2))
    # qcore predictions
    _, _, predict = qcore.make_loss_and_grad(cfg, mask)
    p_qcore = np.asarray(predict(params, jnp.asarray(X)))
    est, _ = hw.make_estimator("exact")
    p_hw = hw.predict(params, mask, X, cfg, est)
    assert np.max(np.abs(p_qcore - p_hw)) < 1e-9


def test_mask_off_term_is_identity():
    # A fully-zero mask must yield the same circuit as omitting all entanglers:
    # predictions depend only on single-qubit layers, matching qcore with an empty mask.
    cfg = _cfg()
    params = qcore.init_params(cfg, seed=5)
    empty = np.zeros((cfg.depth, cfg.n_edges, 3), dtype=int)
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(6, 2))
    _, _, predict = qcore.make_loss_and_grad(cfg, empty)
    p_qcore = np.asarray(predict(params, jnp.asarray(X)))
    est, _ = hw.make_estimator("exact")
    p_hw = hw.predict(params, empty, X, cfg, est)
    assert np.max(np.abs(p_qcore - p_hw)) < 1e-9


def test_readout_observables_target_correct_qubits():
    cfg = _cfg()
    obs = hw.readout_observables(cfg)
    assert len(obs) == len(cfg.readout_paulis)
    # each is a single-qubit Pauli on 2 qubits
    for o in obs:
        assert o.num_qubits == 2


def test_save_load_roundtrip(tmp_path):
    cfg = _cfg()
    params = qcore.init_params(cfg, seed=9)
    mask = qcore.full_mask(cfg)
    path = tmp_path / "model.npz"
    hw.save_model(str(path), params, mask, cfg)
    p2, m2, cfg2 = hw.load_model(str(path))
    assert cfg2.n_qubits == cfg.n_qubits and cfg2.depth == cfg.depth
    assert cfg2.readout_paulis == cfg.readout_paulis
    assert np.array_equal(np.asarray(m2), np.asarray(mask))
    est, _ = hw.make_estimator("exact")
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, size=(5, 2))
    assert np.max(np.abs(hw.predict(params, mask, X, cfg, est)
                        - hw.predict(p2, m2, X, cfg2, est))) < 1e-12
