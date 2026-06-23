"""
qdensity.py

The literal quantum Bayesian learner: a genuine mixed-state density-matrix model whose
belief is a density operator, whose prior is a real (mixed) state, and whose evidence
update is a NON-UNITAL quantum channel with trace renormalization -- the quantum analogue
of a Bayesian belief update. This implements, in code, the mechanism the paper describes
(Eq. for E_x), with no pure-state shortcut.

Per stimulus x, starting from the prior rho_0:
  for each re-uploading block:
    rho <- U_enc(x) rho U_enc(x)^dagger          # unitary feature encoding
    rho <- U_var(theta,m) rho U_var^dagger        # masked Heisenberg variational layer
    rho <- E_x(rho)                                # NON-UNITAL evidence channel:
             (a) Luders/POVM filter K_x rho K_x^dagger, then renormalize  (Bayesian update)
             (b) amplitude damping per qubit (a non-unital CPTP Kraus channel)
  read out posterior category probability from single-qubit Pauli expectations of rho.

The prior is the maximally mixed state I/2^n by default ("no information"): a unitary alone
leaves it invariant, so it is precisely the NON-UNITAL evidence channel that lets evidence
move the belief -- exactly the role evidence plays in Bayesian updating. Everything is a
differentiable dense matrix operation, so the model trains with exact JAX gradients.

Reuses gate matrices and the ModelConfig / readout operators from qcore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .qcore import (ModelConfig, _rx, _rz, _ry, _rpp, _X, _Y, _Z,
                    build_readout_ops)

_I2 = jnp.eye(2, dtype=jnp.complex128)


# ============================================================================
# Embedding single- and two-qubit operators into the full 2^n-dim space
# (qubit 0 is the first tensor factor, matching qcore's convention)
# ============================================================================

def _embed1(U2, q, n):
    full = U2 if q == 0 else _I2
    for k in range(1, n):
        full = jnp.kron(full, U2 if k == q else _I2)
    return full


def _embed2(U4, i, n):
    """Embed a 4x4 gate acting on adjacent qubits (i, i+1)."""
    left = jnp.eye(2 ** i, dtype=jnp.complex128)
    right = jnp.eye(2 ** (n - i - 2), dtype=jnp.complex128)
    return jnp.kron(jnp.kron(left, U4), right)


def _conj(rho, M):
    return M @ rho @ M.conj().T


# ============================================================================
# Configuration and parameters
# ============================================================================

@dataclass
class DensityConfig(ModelConfig):
    prior: str = "mixed"          # "mixed" (I/2^n) or "pure" (|0..0><0..0|)
    damping: bool = True          # include amplitude-damping (multi-Kraus, non-unital) channel


def prior_state(cfg: DensityConfig):
    dim = 2 ** cfg.n_qubits
    if cfg.prior == "pure":
        rho = jnp.zeros((dim, dim), dtype=jnp.complex128).at[0, 0].set(1.0)
    else:
        rho = jnp.eye(dim, dtype=jnp.complex128) / dim
    return rho


def init_params(cfg: DensityConfig, seed: int = 0, scale: float = 0.3) -> Dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    n_feat = len(cfg.readout_paulis)
    return {
        "sq": jnp.array(rng.normal(0.0, scale, size=(cfg.depth, cfg.n_qubits, 2))),
        "ent": jnp.array(rng.normal(0.0, scale, size=(cfg.depth, cfg.n_edges, 3))),
        # evidence Luders filter K_x = exp(-(alpha/2) sum_q g_q(x) Z_q), g_q(x)=ev[q,0]*x_q+ev[q,1]
        "ev": jnp.array(rng.normal(0.0, scale, size=(cfg.n_qubits, 2))),
        "alpha": jnp.array(1.0),
        # amplitude-damping rate gamma_q(x) = sigmoid(damp[q,0]*x_q + damp[q,1])
        "damp": jnp.array(rng.normal(0.0, scale, size=(cfg.n_qubits, 2))),
        "w": jnp.array(rng.normal(0.0, 0.1, size=(n_feat,))),
        "b": jnp.array(0.0),
    }


def full_mask(cfg: DensityConfig) -> np.ndarray:
    return np.ones((cfg.depth, cfg.n_edges, 3), dtype=int)


# ============================================================================
# Evidence channel: non-unital, with trace renormalization
# ============================================================================

def _evidence_channel(rho, x, params, cfg, Zq):
    n = cfg.n_qubits
    # (a) Luders / POVM filter -- a stimulus-dependent contraction, then renormalize.
    # K_x is diagonal: exp(-(alpha/2) sum_q g_q(x) Z_q). Non-trace-preserving => Bayesian renorm.
    g = params["ev"][:, 0] * x[:n] + params["ev"][:, 1]          # (n,)
    Hdiag = jnp.zeros((2 ** n,), dtype=jnp.float64)
    for q in range(n):
        Hdiag = Hdiag + g[q] * jnp.real(jnp.diagonal(Zq[q]))
    K = jnp.diag(jnp.exp(-0.5 * params["alpha"] * Hdiag)).astype(jnp.complex128)
    rho = _conj(rho, K)
    rho = rho / jnp.trace(rho)                                    # <-- Bayesian renormalization

    # (b) amplitude damping per qubit -- a genuine multi-Kraus, NON-UNITAL CPTP channel.
    if cfg.damping:
        for q in range(n):
            gamma = jax.nn.sigmoid(params["damp"][q, 0] * x[min(q, x.shape[0] - 1)] + params["damp"][q, 1])
            s = jnp.sqrt(jnp.clip(gamma, 0.0, 1.0))
            E0 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128) \
                 + jnp.sqrt(jnp.clip(1.0 - gamma, 0.0, 1.0)) * jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
            E1 = s * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128)
            E0f, E1f = _embed1(E0, q, n), _embed1(E1, q, n)
            rho = _conj(rho, E0f) + _conj(rho, E1f)
        rho = rho / jnp.trace(rho)
    return rho


# ============================================================================
# Forward map
# ============================================================================

def run_density(params, x, cfg: DensityConfig, mask, gate_scale=None):
    """Evolve the belief density matrix. If ``gate_scale`` (depth,n_edges,3 in [0,1]) is
    given, each Heisenberg angle is multiplied by its scale (a differentiable, recompile-free
    relaxation of the binary mask: scale 0 => identity); otherwise the static ``mask`` is used."""
    n = cfg.n_qubits
    Zq = [_embed1(_Z, q, n) for q in range(n)]
    ent = params["ent"] if gate_scale is None else params["ent"] * gate_scale
    rho = prior_state(cfg)
    layers_per_reupload = max(1, cfg.depth // cfg.n_reupload)
    for d in range(cfg.depth):
        if d % layers_per_reupload == 0:
            for q in range(min(n, x.shape[0])):
                ang = cfg.feature_scale * x[q]
                rho = _conj(rho, _embed1(_ry(ang), q, n))
                rho = _conj(rho, _embed1(_rz(ang), q, n))
        # variational unitary layer
        for q in range(n):
            rho = _conj(rho, _embed1(_rx(params["sq"][d, q, 0]), q, n))
            rho = _conj(rho, _embed1(_rz(params["sq"][d, q, 1]), q, n))
        for k, (i, j) in enumerate(cfg.pairs):
            if gate_scale is not None:
                rho = _conj(rho, _embed2(_rpp(ent[d, k, 0], _X), i, n))
                rho = _conj(rho, _embed2(_rpp(ent[d, k, 1], _Y), i, n))
                rho = _conj(rho, _embed2(_rpp(ent[d, k, 2], _Z), i, n))
            else:
                mk = mask[d][k]
                if mk[0]:
                    rho = _conj(rho, _embed2(_rpp(ent[d, k, 0], _X), i, n))
                if mk[1]:
                    rho = _conj(rho, _embed2(_rpp(ent[d, k, 1], _Y), i, n))
                if mk[2]:
                    rho = _conj(rho, _embed2(_rpp(ent[d, k, 2], _Z), i, n))
        # non-unital evidence update
        rho = _evidence_channel(rho, x, params, cfg, Zq)
    return rho


def _expectations(rho, readout_ops):
    return jnp.stack([jnp.real(jnp.trace(rho @ O)) for O in readout_ops])


def logit(params, x, cfg, readout_ops, mask):
    rho = run_density(params, x, cfg, mask)
    feats = _expectations(rho, readout_ops)
    return jnp.dot(params["w"], feats) + params["b"]


# ============================================================================
# Loss / gradients / optimizer (runtime mask via static tuple)
# ============================================================================

def _mask_tuple(mask, cfg):
    m = np.asarray(mask, dtype=int)
    return tuple(tuple(tuple(bool(v) for v in edge) for edge in layer) for layer in m.tolist())


def make_loss_and_grad(cfg: DensityConfig, mask):
    readout_ops = build_readout_ops(cfg)
    mt = _mask_tuple(mask, cfg)

    def single(params, x):
        return logit(params, x, cfg, readout_ops, mt)

    batched = jax.vmap(single, in_axes=(None, 0))

    def loss_fn(params, X, y):
        logits = batched(params, X)
        ce = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(ce)

    def predict(params, X):
        return jax.nn.sigmoid(batched(params, X))

    return jax.jit(loss_fn), jax.jit(jax.value_and_grad(loss_fn)), jax.jit(predict)


def make_dynamic_fns(cfg: DensityConfig):
    """Jitted fns with the mask as a RUNTIME argument (no per-mask recompile), via the
    entangler gate-scale. Bit-identical to the static discrete circuit for binary masks."""
    readout_ops = build_readout_ops(cfg)

    def single(params, x, mask):
        rho = run_density(params, x, cfg, None, gate_scale=mask)
        feats = _expectations(rho, readout_ops)
        return jnp.dot(params["w"], feats) + params["b"]

    batched = jax.vmap(single, in_axes=(None, 0, None))

    def loss_fn(params, X, y, mask):
        logits = batched(params, X, mask)
        ce = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(ce)

    def predict(params, X, mask):
        return jax.nn.sigmoid(batched(params, X, mask))

    return (jax.jit(loss_fn), jax.jit(jax.value_and_grad(loss_fn, argnums=0)), jax.jit(predict))


class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = jax.tree_util.tree_map(jnp.zeros_like, params)
        self.v = jax.tree_util.tree_map(jnp.zeros_like, params)
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        self.m = jax.tree_util.tree_map(lambda m, g: self.b1 * m + (1 - self.b1) * g, self.m, grads)
        self.v = jax.tree_util.tree_map(lambda v, g: self.b2 * v + (1 - self.b2) * g * g, self.v, grads)
        mh = jax.tree_util.tree_map(lambda m: m / (1 - self.b1 ** self.t), self.m)
        vh = jax.tree_util.tree_map(lambda v: v / (1 - self.b2 ** self.t), self.v)
        return jax.tree_util.tree_map(lambda p, m, v: p - self.lr * m / (jnp.sqrt(v) + self.eps),
                                      params, mh, vh)


def accuracy(predict_fn, params, X, y) -> float:
    p = np.asarray(predict_fn(params, X))
    return float(np.mean((p >= 0.5).astype(int) == np.asarray(y)))


# ============================================================================
# Diagnostics: trace, purity, positivity (for tests)
# ============================================================================

def density_diagnostics(params, x, cfg, mask):
    rho = np.asarray(run_density(params, jnp.asarray(x), cfg, _mask_tuple(mask, cfg)))
    tr = float(np.real(np.trace(rho)))
    herm = float(np.max(np.abs(rho - rho.conj().T)))
    eig = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    purity = float(np.real(np.trace(rho @ rho)))
    return {"trace": tr, "hermiticity_err": herm, "min_eig": float(eig.min()), "purity": purity}
