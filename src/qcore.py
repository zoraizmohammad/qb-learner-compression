"""
qcore.py

Exact, differentiable quantum learner core for the quantum Bayesian learner.

Design goals (see HANDOFF.md, Phase 1):
- A genuinely trainable model that learns the Pothos-Chater boundary well above
  chance, with EXACT gradients (JAX autodiff -- no finite-difference noise).
- An honest, well-defined forward model:
    |0...0>  --[ data re-uploading: encode(x) then a trainable variational layer ]xL-->
    measure a set of Pauli expectations  -->  trainable linear readout  -->  sigmoid  -->  BCE.
- Heisenberg-type entanglers (RXX, RYY, RZZ) matching the paper's intended ansatz,
  gated by a binary mask over interaction terms (used for compression in later phases).
- The belief register starts in the pure state |0...0> (the "no evidence yet" prior).
  The variational + encoding map is the unitary belief-update channel; the readout
  yields the posterior class probability. This avoids the maximally-mixed-invariance
  problem documented in HANDOFF.md and keeps the model fully differentiable.

This module is the SINGLE source of truth for the learned model. Hardware cost
(post-transpile FakeManila two-qubit count) is computed separately in
``hardware_cost.py`` using Qiskit, so the two concerns stay decoupled.

The simulator's statevector is verified bit-for-bit against Qiskit in
``scripts/verify_qcore.py`` so there are no gate-convention bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # exact double precision, deterministic


# ============================================================================
# Gate matrices (Qiskit conventions)
# ============================================================================

def _rx(t):
    c = jnp.cos(t / 2.0)
    s = jnp.sin(t / 2.0)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex128)


def _ry(t):
    c = jnp.cos(t / 2.0)
    s = jnp.sin(t / 2.0)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)


def _rz(t):
    e = jnp.exp(-1j * t / 2.0)
    return jnp.array([[e, 0], [0, jnp.conj(e)]], dtype=jnp.complex128)


# Pauli matrices
_I = jnp.eye(2, dtype=jnp.complex128)
_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


def _rpp(t, P):
    """exp(-i t/2 * P⊗P) for a two-qubit Pauli-Pauli interaction (RXX/RYY/RZZ)."""
    PP = jnp.kron(P, P)
    c = jnp.cos(t / 2.0)
    s = jnp.sin(t / 2.0)
    return c * jnp.eye(4, dtype=jnp.complex128) - 1j * s * PP


# ============================================================================
# Statevector application via tensor contraction (supports any n_qubits)
# ============================================================================

def _apply_1q(state, U, q, n):
    """Apply 2x2 gate U to qubit q of an n-qubit statevector (shape (2,)*n)."""
    state = jnp.tensordot(U, state, axes=([1], [q]))      # new axis 0 = output qubit q
    state = jnp.moveaxis(state, 0, q)
    return state


def _apply_2q(state, U, i, j, n):
    """Apply 4x4 gate U to qubits (i, j). U indexed as (i_out, j_out, i_in, j_in)."""
    U4 = U.reshape(2, 2, 2, 2)
    state = jnp.tensordot(U4, state, axes=([2, 3], [i, j]))  # axes 0,1 = out i,j
    state = jnp.moveaxis(state, [0, 1], [i, j])
    return state


# ============================================================================
# Model configuration
# ============================================================================

@dataclass
class ModelConfig:
    n_qubits: int = 2
    depth: int = 3
    pairs: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 1)])
    n_reupload: int = 2          # how many times features are re-encoded
    feature_scale: float = np.pi  # features in [0,1] -> rotation angles in [0, pi]
    # Pauli observables used as readout features: list of (pauli_string) e.g. "Z0","X1","Z0Z1"
    readout_paulis: Tuple[str, ...] = ("Z0", "Z1", "X0", "X1", "Z0Z1")

    @property
    def n_edges(self) -> int:
        return len(self.pairs)


# ----------------------------------------------------------------------------
# Readout observable construction (full 2^n x 2^n Hermitian matrices)
# ----------------------------------------------------------------------------

def _pauli_op(spec: str, n: int):
    """Build a full operator from a spec like 'Z0', 'X1', 'Z0Z1'."""
    table = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    per_qubit = [None] * n
    k = 0
    while k < len(spec):
        p = spec[k]
        # read the integer index following the pauli letter
        k += 1
        num = ""
        while k < len(spec) and spec[k].isdigit():
            num += spec[k]
            k += 1
        per_qubit[int(num)] = table[p]
    op = jnp.array([[1.0 + 0j]])
    for q in range(n):
        op = jnp.kron(op, per_qubit[q] if per_qubit[q] is not None else _I)
    return op


def build_readout_ops(cfg: ModelConfig):
    return [_pauli_op(s, cfg.n_qubits) for s in cfg.readout_paulis]


# ============================================================================
# Parameter initialization
# ============================================================================

def full_mask(cfg: ModelConfig) -> np.ndarray:
    """All Heisenberg interaction terms active: shape (depth, n_edges, 3)."""
    return np.ones((cfg.depth, cfg.n_edges, 3), dtype=int)


def init_params(cfg: ModelConfig, seed: int = 0, scale: float = 0.3) -> Dict[str, jnp.ndarray]:
    """Trainable parameters: single-qubit rotations, Heisenberg angles, readout head."""
    rng = np.random.default_rng(seed)
    n_feat = len(cfg.readout_paulis)
    params = {
        # per layer, per qubit: (RX, RZ)
        "sq": jnp.array(rng.normal(0.0, scale, size=(cfg.depth, cfg.n_qubits, 2))),
        # per layer, per edge: (RXX, RYY, RZZ)
        "ent": jnp.array(rng.normal(0.0, scale, size=(cfg.depth, cfg.n_edges, 3))),
        # trainable readout head
        "w": jnp.array(rng.normal(0.0, 0.1, size=(n_feat,))),
        "b": jnp.array(0.0),
        # per-term mask gate logits (used only by the learned-mask path). Init high
        # (+3 -> sigmoid ~0.95) so compression starts from a (near) full circuit.
        "gates": jnp.full((cfg.depth, cfg.n_edges, 3), 3.0),
    }
    return params


# ============================================================================
# Forward pass
# ============================================================================

def _encode(state, x, cfg: ModelConfig):
    """Feature encoding: RY(scale*x_q) then RZ(scale*x_q) on each qubit."""
    n = cfg.n_qubits
    for q in range(min(n, x.shape[0])):
        ang = cfg.feature_scale * x[q]
        state = _apply_1q(state, _ry(ang), q, n)
        state = _apply_1q(state, _rz(ang), q, n)
    return state


def _variational_layer(state, sq_d, ent_d, mask_d, cfg: ModelConfig):
    """One trainable layer: single-qubit rotations + masked Heisenberg entanglers."""
    n = cfg.n_qubits
    for q in range(n):
        state = _apply_1q(state, _rx(sq_d[q, 0]), q, n)
        state = _apply_1q(state, _rz(sq_d[q, 1]), q, n)
    for k, (i, j) in enumerate(cfg.pairs):
        mk = mask_d[k]  # (XX_active, YY_active, ZZ_active) static python bools
        if mk[0]:
            state = _apply_2q(state, _rpp(ent_d[k, 0], _X), i, j, n)
        if mk[1]:
            state = _apply_2q(state, _rpp(ent_d[k, 1], _Y), i, j, n)
        if mk[2]:
            state = _apply_2q(state, _rpp(ent_d[k, 2], _Z), i, j, n)
    return state


def run_state(params, x, cfg: ModelConfig, mask, gate_scale=None):
    """Evolve |0...0> through re-uploading + variational layers; return statevector.

    If ``gate_scale`` (shape (depth, n_edges, 3) in [0,1]) is given, each Heisenberg
    interaction angle is multiplied by its scale -- a differentiable relaxation of the
    binary mask: scale->0 turns the term into identity, scale->1 keeps it fully. Used
    by the learned-mask compression path. When None, the discrete ``mask`` is used.
    """
    n = cfg.n_qubits
    state = jnp.zeros((2,) * n, dtype=jnp.complex128)
    state = state.at[(0,) * n].set(1.0 + 0j)  # |0...0>
    eff_ent = params["ent"] if gate_scale is None else params["ent"] * gate_scale
    # interleave data re-uploading with trainable layers
    layers_per_reupload = max(1, cfg.depth // cfg.n_reupload)
    for d in range(cfg.depth):
        if d % layers_per_reupload == 0:
            state = _encode(state, x, cfg)
        state = _variational_layer(state, params["sq"][d], eff_ent[d], mask[d], cfg)
    return state


def _expectations(state, readout_ops, n):
    psi = state.reshape(-1)
    feats = []
    for O in readout_ops:
        feats.append(jnp.real(jnp.vdot(psi, O @ psi)))
    return jnp.stack(feats)


def logit(params, x, cfg: ModelConfig, readout_ops, mask):
    state = run_state(params, x, cfg, mask)
    feats = _expectations(state, readout_ops, cfg.n_qubits)
    return jnp.dot(params["w"], feats) + params["b"]


def proba(params, x, cfg, readout_ops, mask):
    return jax.nn.sigmoid(logit(params, x, cfg, readout_ops, mask))


# ============================================================================
# Loss (mean binary cross-entropy) -- vectorized over the dataset
# ============================================================================

def _make_batched_logit(cfg: ModelConfig, readout_ops, mask):
    """Return a jitted function params,X -> logits over a batch."""
    def single(params, x):
        return logit(params, x, cfg, readout_ops, mask)
    batched = jax.vmap(single, in_axes=(None, 0))
    return batched


def make_loss_and_grad(cfg: ModelConfig, mask):
    """Build a (loss_fn, value_and_grad_fn) pair for given (static) config and mask."""
    readout_ops = build_readout_ops(cfg)
    # mask shape: (depth, n_edges, 3) of {0,1}; one flag per Heisenberg term (XX,YY,ZZ).
    marr = np.asarray(mask, dtype=int)
    mask = tuple(
        tuple(tuple(bool(v) for v in edge) for edge in layer)
        for layer in marr.tolist()
    )
    batched_logit = _make_batched_logit(cfg, readout_ops, mask)

    def loss_fn(params, X, y):
        logits = batched_logit(params, X)
        # numerically stable BCE with logits
        ce = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(ce)

    loss_fn_j = jax.jit(loss_fn)
    vg = jax.jit(jax.value_and_grad(loss_fn))

    def predict_proba(params, X):
        return jax.nn.sigmoid(batched_logit(params, X))

    return loss_fn_j, vg, jax.jit(predict_proba)


def make_learned_mask_loss_and_grad(cfg: ModelConfig, lam: float, cost_per_term: float = 2.0):
    """Hardware-aware loss with a LEARNED (continuously-relaxed) per-term mask.

    L = mean_BCE + lam * cost_per_term * sum(sigmoid(gates)).

    The penalty is the expected post-transpile two-qubit cost (each Heisenberg term
    costs ~`cost_per_term` CX on FakeManila). Larger `lam` drives the gate keep-probs
    toward 0, pruning interaction terms. Gradients are exact (JAX). For evaluation /
    reporting, binarize the gates (`binarize_gates`) and recompute the REAL FakeManila
    cost + the discrete-circuit accuracy -- never the relaxed surrogate.
    """
    readout_ops = build_readout_ops(cfg)
    all_on = tuple(tuple((True, True, True) for _ in range(cfg.n_edges)) for _ in range(cfg.depth))

    def single_logit(params, x):
        s = jax.nn.sigmoid(params["gates"])
        state = run_state(params, x, cfg, all_on, gate_scale=s)
        feats = _expectations(state, readout_ops, cfg.n_qubits)
        return jnp.dot(params["w"], feats) + params["b"]

    batched = jax.vmap(single_logit, in_axes=(None, 0))

    def loss_fn(params, X, y):
        logits = batched(params, X)
        ce = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        surrogate_cost = cost_per_term * jnp.sum(jax.nn.sigmoid(params["gates"]))
        return jnp.mean(ce) + lam * surrogate_cost

    return jax.jit(loss_fn), jax.jit(jax.value_and_grad(loss_fn))


def binarize_gates(params, threshold: float = 0.5) -> np.ndarray:
    """Return the discrete per-term mask (depth, n_edges, 3) implied by learned gates."""
    s = np.asarray(jax.nn.sigmoid(params["gates"]))
    return (s > threshold).astype(int)


def make_dynamic_fns(cfg: ModelConfig):
    """Build jitted fns where the mask is a RUNTIME argument (no per-mask recompile).

    The discrete mask (0/1 float array, shape (depth, n_edges, 3)) is passed as the
    entangler gate-scale: a term with mask=0 becomes RPP(0)=identity (exactly absent),
    mask=1 keeps it fully. This is bit-identical to the static discrete circuit but lets
    one compiled function evaluate/train ANY mask, which is dramatically faster for
    pruning sweeps. Returns (loss_fn, value_and_grad_fn, predict_proba_fn), each taking
    a trailing ``mask`` argument.
    """
    readout_ops = build_readout_ops(cfg)
    all_on = tuple(tuple((True, True, True) for _ in range(cfg.n_edges)) for _ in range(cfg.depth))

    def single_logit(params, x, mask):
        state = run_state(params, x, cfg, all_on, gate_scale=mask)
        feats = _expectations(state, readout_ops, cfg.n_qubits)
        return jnp.dot(params["w"], feats) + params["b"]

    batched = jax.vmap(single_logit, in_axes=(None, 0, None))

    def loss_fn(params, X, y, mask):
        logits = batched(params, X, mask)
        ce = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(ce)

    def predict_proba(params, X, mask):
        return jax.nn.sigmoid(batched(params, X, mask))

    return (jax.jit(loss_fn),
            jax.jit(jax.value_and_grad(loss_fn, argnums=0)),
            jax.jit(predict_proba))


# ============================================================================
# Adam optimizer over a pytree of params
# ============================================================================

class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        self.v = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        self.m = jax.tree_util.tree_map(lambda m, g: self.b1 * m + (1 - self.b1) * g, self.m, grads)
        self.v = jax.tree_util.tree_map(lambda v, g: self.b2 * v + (1 - self.b2) * g * g, self.v, grads)
        mhat = jax.tree_util.tree_map(lambda m: m / (1 - self.b1 ** self.t), self.m)
        vhat = jax.tree_util.tree_map(lambda v: v / (1 - self.b2 ** self.t), self.v)
        return jax.tree_util.tree_map(
            lambda p, m, v: p - self.lr * m / (jnp.sqrt(v) + self.eps), params, mhat, vhat
        )


# ============================================================================
# Convenience: accuracy
# ============================================================================

def accuracy(predict_proba_fn, params, X, y) -> float:
    p = np.asarray(predict_proba_fn(params, X))
    return float(np.mean((p >= 0.5).astype(int) == np.asarray(y)))
