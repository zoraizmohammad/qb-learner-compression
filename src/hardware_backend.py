"""
hardware_backend.py

Run the *trained* quantum Bayesian learner on real (or realistically-noisy) quantum
hardware. Training stays in exact simulation (src/qcore.py, fast analytic gradients);
this module deploys a trained model for inference/evaluation by:

  1. Building the Qiskit circuit that mirrors the qcore ansatz EXACTLY (data
     re-uploading + single-qubit rotations + masked Heisenberg entanglers).
  2. Estimating the readout observables (single-qubit Pauli expectations) on a chosen
     backend via the Estimator primitive.
  3. Applying the trained linear readout head (w, b) to those expectations to predict.

Three backends, selected by ``make_estimator``:
  * "exact"  -- StatevectorEstimator: noiseless, exact. Bit-identical to qcore (verified
                in tests), so the bridge from simulation to circuit is trustworthy.
  * "noisy"  -- AerSimulator.from_backend(FakeManila): the real FakeManila noise model,
                with finite shots. Free, requires no account, and is the recommended way
                to see how the simulated frontier holds up under device noise.
  * "ibm"    -- a real IBM Quantum device via QiskitRuntimeService (needs an API token;
                see docs/HARDWARE.md). Jobs queue and consume QPU time.

Because training is done in simulation and only inference runs on hardware, validating
the paper's frontier on a real device costs a handful of cheap circuit executions rather
than a full hardware-in-the-loop training run.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from . import qcore


# ============================================================================
# Circuit construction (mirrors qcore.run_state exactly)
# ============================================================================

def build_inference_circuit(params, mask, x, cfg: qcore.ModelConfig) -> QuantumCircuit:
    """Qiskit circuit reproducing qcore's forward map for one stimulus x.

    Same gate set, order, angles, qubit indexing, and (discrete) mask as qcore, so a
    model trained in qcore transfers unchanged. No measurement is appended; observables
    are supplied to the Estimator primitive.
    """
    n = cfg.n_qubits
    sq = np.asarray(params["sq"], dtype=float)
    ent = np.asarray(params["ent"], dtype=float)
    mask = np.asarray(mask, dtype=int)
    x = np.asarray(x, dtype=float)
    qc = QuantumCircuit(n)
    layers_per_reupload = max(1, cfg.depth // cfg.n_reupload)
    for d in range(cfg.depth):
        if d % layers_per_reupload == 0:
            for q in range(min(n, x.shape[0])):
                ang = cfg.feature_scale * x[q]
                qc.ry(ang, q)
                qc.rz(ang, q)
        for q in range(n):
            qc.rx(sq[d, q, 0], q)
            qc.rz(sq[d, q, 1], q)
        for k, (i, j) in enumerate(cfg.pairs):
            if mask[d, k, 0]:
                qc.rxx(ent[d, k, 0], i, j)
            if mask[d, k, 1]:
                qc.ryy(ent[d, k, 1], i, j)
            if mask[d, k, 2]:
                qc.rzz(ent[d, k, 2], i, j)
    return qc


def _parse_pauli(spec: str):
    """'Z0' -> ('Z',[0]); 'Z0Z1' -> ('ZZ',[0,1])."""
    label, qubits, k = "", [], 0
    while k < len(spec):
        p = spec[k]; k += 1; num = ""
        while k < len(spec) and spec[k].isdigit():
            num += spec[k]; k += 1
        label += p; qubits.append(int(num))
    return label, qubits


def readout_observables(cfg: qcore.ModelConfig) -> List[SparsePauliOp]:
    """Observables for cfg.readout_paulis, qubit-targeted (endian-safe)."""
    obs = []
    for spec in cfg.readout_paulis:
        label, qubits = _parse_pauli(spec)
        obs.append(SparsePauliOp.from_sparse_list([(label, qubits, 1.0)], num_qubits=cfg.n_qubits))
    return obs


# ============================================================================
# Estimator backends
# ============================================================================

def make_estimator(mode: str = "exact", shots: int = 4096, seed: int = 42,
                   ibm_backend: Optional[str] = None, optimization_level: int = 1):
    """Return (estimator, transpile_target) for a backend mode.

    transpile_target is a backend to pre-transpile circuits onto (or None for exact).
    """
    mode = mode.lower()
    if mode == "exact":
        from qiskit.primitives import StatevectorEstimator
        return StatevectorEstimator(seed=seed), None

    if mode == "noisy":
        from qiskit_aer import AerSimulator
        from qiskit.primitives import BackendEstimatorV2
        try:
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        except Exception:
            from qiskit.providers.fake_provider import FakeManilaV2
        sim = AerSimulator.from_backend(FakeManilaV2())
        est = BackendEstimatorV2(backend=sim)
        est.options.default_shots = shots
        return est, sim

    if mode == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
        service = QiskitRuntimeService()  # uses saved account / QISKIT_IBM_TOKEN
        backend = (service.backend(ibm_backend) if ibm_backend
                   else service.least_busy(operational=True, simulator=False))
        est = EstimatorV2(mode=backend)
        est.options.default_shots = shots
        return est, backend

    raise ValueError(f"unknown estimator mode {mode!r} (use 'exact', 'noisy', or 'ibm')")


# ============================================================================
# Inference and evaluation
# ============================================================================

def estimate_expectations(params, mask, X, cfg, estimator, transpile_target=None,
                          optimization_level: int = 1, seed_transpiler: int = 42) -> np.ndarray:
    """Estimate the readout expectations for every row of X. Shape (len(X), n_features)."""
    obs = readout_observables(cfg)
    circuits = [build_inference_circuit(params, mask, x, cfg) for x in np.asarray(X)]

    if transpile_target is not None:
        from qiskit import transpile
        circuits = transpile(circuits, backend=transpile_target,
                              optimization_level=optimization_level,
                              seed_transpiler=seed_transpiler)
        # map observables onto the transpiled layout of each circuit
        pubs = [(qc, [o.apply_layout(qc.layout) for o in obs]) for qc in circuits]
    else:
        pubs = [(qc, obs) for qc in circuits]

    job = estimator.run(pubs)
    res = job.result()
    feats = np.array([np.asarray(r.data.evs, dtype=float).reshape(-1) for r in res])
    return feats


def predict(params, mask, X, cfg, estimator, **kw) -> np.ndarray:
    feats = estimate_expectations(params, mask, X, cfg, estimator, **kw)
    w = np.asarray(params["w"], dtype=float)
    b = float(params["b"])
    logits = feats @ w + b
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate(params, mask, X, y, cfg, estimator, **kw) -> dict:
    p = predict(params, mask, X, cfg, estimator, **kw)
    yhat = (p >= 0.5).astype(int)
    y = np.asarray(y)
    return {"accuracy": float(np.mean(yhat == y)),
            "proba": p, "pred": yhat}


# ============================================================================
# Persisting trained models (train once in sim, deploy later on hardware)
# ============================================================================

def save_model(path, params, mask, cfg: qcore.ModelConfig) -> None:
    np.savez(
        path,
        sq=np.asarray(params["sq"]), ent=np.asarray(params["ent"]),
        w=np.asarray(params["w"]), b=np.asarray(params["b"]),
        gates=np.asarray(params.get("gates", np.zeros((cfg.depth, cfg.n_edges, 3)))),
        mask=np.asarray(mask, dtype=int),
        n_qubits=cfg.n_qubits, depth=cfg.depth, n_reupload=cfg.n_reupload,
        feature_scale=cfg.feature_scale,
        pairs=np.asarray(cfg.pairs), readout_paulis=np.asarray(cfg.readout_paulis),
    )


def load_model(path):
    """Return (params, mask, cfg) from a saved .npz model."""
    import jax.numpy as jnp
    d = np.load(path, allow_pickle=True)
    cfg = qcore.ModelConfig(
        n_qubits=int(d["n_qubits"]), depth=int(d["depth"]),
        pairs=[tuple(p) for p in d["pairs"]], n_reupload=int(d["n_reupload"]),
        feature_scale=float(d["feature_scale"]),
        readout_paulis=tuple(str(s) for s in d["readout_paulis"]),
    )
    params = {"sq": jnp.asarray(d["sq"]), "ent": jnp.asarray(d["ent"]),
              "w": jnp.asarray(d["w"]), "b": jnp.asarray(d["b"]),
              "gates": jnp.asarray(d["gates"])}
    return params, d["mask"], cfg
