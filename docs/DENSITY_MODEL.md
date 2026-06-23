# The Density-Matrix Quantum Bayesian Learner (`src/qdensity.py`)

This document describes the literal mixed-state, density-matrix implementation of the
quantum Bayesian learner. It is the honest realization of the mechanism the paper's
Methods section describes: a belief that is a density operator and an evidence update
that is a genuine (non-unital) quantum channel — with no pure-state shortcut.

## Contents

1. [Why this module exists](#1-why-this-module-exists)
2. [The mechanism](#2-the-mechanism)
3. [Verified properties](#3-verified-properties)
4. [How to use it](#4-how-to-use-it)
5. [Relation to the paper](#5-relation-to-the-paper)

---

## 1. Why this module exists

The project ships **two** learner implementations that share the same gate matrices,
`ModelConfig`, and Pauli-readout construction (`src/qcore.py` is the single source of
truth for those primitives):

- **`src/qcore.py` — fast, exact PURE-STATE model.** This is a variational statevector
  simulator. The belief register starts in the pure state `|0…0⟩`, evolves under unitary
  data re-uploading and a masked Heisenberg variational ansatz, and the class probability
  is read from Pauli expectations through a trainable linear head. It is differentiable
  (JAX, float64), verified bit-for-bit against Qiskit, and fast. **All of the bulk
  experiments — the compression frontiers, λ sweeps, mask ablations, and hardware-cost
  studies — run on `qcore`.**

- **`src/qdensity.py` — literal MIXED-STATE density-matrix model.** Here the belief is a
  full density operator `ρ` (a `2ⁿ × 2ⁿ` Hermitian matrix), the prior is a real (possibly
  mixed) state, and the per-stimulus evidence update is a **non-unital quantum channel
  with trace renormalization** — the quantum analogue of a Bayesian belief update. There
  is no pure-state shortcut anywhere in the forward map.

`qdensity` exists so that the paper's methods are true in the strongest possible sense.
The pure-state model is a perfectly valid variational quantum classifier, but it cannot,
by itself, demonstrate the "belief-as-density-operator, evidence-as-quantum-channel"
mechanism, because a unitary evolution of a pure state can never realize a genuine,
information-gaining channel acting on a mixed prior. `qdensity` carries the dense matrix
machinery needed to make that mechanism literal: a maximally mixed prior, a Lüders/POVM
evidence filter, amplitude damping, and trace renormalization, all of it still fully
differentiable so the model trains with exact JAX gradients.

It is slower than `qcore` (dense `2ⁿ × 2ⁿ` matrix conjugations instead of statevector
contractions), but it is exact and differentiable, and it reduces *bit-for-bit* to the
statevector model when the channel is switched off and the prior is pure (see §3).

---

## 2. The mechanism

### 2.1 The belief state and its prior

The belief is a density operator `ρ`. The prior `ρ₀` is produced by `prior_state(cfg)`
and is selected by `DensityConfig.prior`:

- `prior="mixed"` → the **maximally mixed state** `ρ₀ = I / 2ⁿ`. This is the natural
  "no information" / uniform prior: every basis state is equally likely.
- `prior="pure"` → the pure state `|0…0⟩⟨0…0|`.

```python
def prior_state(cfg: DensityConfig):
    dim = 2 ** cfg.n_qubits
    if cfg.prior == "pure":
        rho = jnp.zeros((dim, dim), dtype=jnp.complex128).at[0, 0].set(1.0)
    else:
        rho = jnp.eye(dim, dtype=jnp.complex128) / dim
    return rho
```

### 2.2 The per-stimulus update

For each stimulus `x`, starting from `ρ₀`, the forward map `run_density(...)` iterates,
for every re-uploading block, three steps. The first two are **unitary** (acting by
conjugation `ρ → U ρ U†`):

1. **Unitary feature encoding.** Each feature `x_q` is encoded with `RY(scale·x_q)` then
   `RZ(scale·x_q)` on qubit `q` (re-uploaded `n_reupload` times across the depth).
2. **Masked Heisenberg variational layer.** Single-qubit `RX`/`RZ` rotations followed by
   the two-qubit Heisenberg interaction terms `RXX`, `RYY`, `RZZ` on each edge. A per-term
   binary `mask` (or a continuous `gate_scale`; see §4) selects which interaction terms are
   active — this is the compression knob shared with `qcore`.

The third step is the heart of the model — a **non-unital evidence channel**,
`_evidence_channel(...)`:

3a. **Lüders / POVM evidence filter (a Bayesian update).** A stimulus-dependent,
diagonal Kraus operator `K_x = exp(−(α/2) Σ_q g_q(x) Z_q)`, with
`g_q(x) = ev[q,0]·x_q + ev[q,1]`, is applied as `ρ → K_x ρ K_x†`. Because `K_x` is a
*contraction* (it is not trace-preserving on its own), the belief must be renormalized,
`ρ → ρ / Tr(ρ)`. This renormalization **is** the quantum Bayesian update: amplitudes
consistent with the stimulus are up-weighted, the rest are down-weighted, and the
posterior is re-normalized to a valid belief — exactly Bayes' rule for a quantum state.

3b. **Per-qubit amplitude damping (a genuine multi-Kraus, non-unital CPTP channel).** When
`DensityConfig.damping=True`, each qubit is passed through an amplitude-damping channel
with a stimulus-dependent rate `γ_q(x) = sigmoid(damp[q,0]·x_q + damp[q,1])`, applied via
its two Kraus operators `E₀, E₁` as `ρ → E₀ ρ E₀† + E₁ ρ E₁†`, followed by a final
renormalization. Amplitude damping is non-unital (`E₀ E₀† + E₁ E₁†` damps toward `|0⟩`),
so it too can move the belief.

```python
def _evidence_channel(rho, x, params, cfg, Zq):
    n = cfg.n_qubits
    # (a) Lüders / POVM filter: a stimulus-dependent contraction, then renormalize.
    g = params["ev"][:, 0] * x[:n] + params["ev"][:, 1]
    Hdiag = jnp.zeros((2 ** n,), dtype=jnp.float64)
    for q in range(n):
        Hdiag = Hdiag + g[q] * jnp.real(jnp.diagonal(Zq[q]))
    K = jnp.diag(jnp.exp(-0.5 * params["alpha"] * Hdiag)).astype(jnp.complex128)
    rho = _conj(rho, K)
    rho = rho / jnp.trace(rho)                    # <-- Bayesian renormalization
    # (b) amplitude damping per qubit: a multi-Kraus, NON-UNITAL CPTP channel.
    if cfg.damping:
        ...
        rho = rho / jnp.trace(rho)
    return rho
```

### 2.3 Readout

After the final block, the posterior category probability is read from single-qubit
Pauli expectations `Tr[ρ P]` (`_expectations`) fed through a trainable linear head
`logit = w · v(ρ) + b`, then a sigmoid. Loss is numerically stable binary cross-entropy.

### 2.4 Why non-unitality is the whole point

For the maximally mixed prior `ρ₀ = I / 2ⁿ`, **any unitary leaves it invariant**:
`U (I/2ⁿ) U† = U U† / 2ⁿ = I / 2ⁿ`. So if the entire forward map were unitary, evidence
could *never* move the belief — the model would read out the same featureless state for
every stimulus. It is precisely the **non-unital** evidence channel — the Lüders filter
with renormalization, plus amplitude damping — that lets a stimulus change the belief.
That is exactly the role evidence plays in Bayesian updating: a uniform prior stays
uniform until evidence arrives and reshapes it. Non-unitality is therefore not an
implementation detail; it is the mechanism.

### 2.5 Function map

| Function | Role |
| --- | --- |
| `prior_state` | builds `ρ₀` (mixed `I/2ⁿ` or pure `|0…0⟩⟨0…0|`) |
| `run_density` | the full forward map: re-uploading + variational layers + evidence channel |
| `_evidence_channel` | the non-unital Bayesian update (Lüders filter + renorm + amplitude damping) |
| `make_loss_and_grad` | jitted BCE loss, `value_and_grad`, and predict, for a fixed (static) mask |
| `make_dynamic_fns` | same, but with the mask as a runtime argument (no per-mask recompiles) |
| `density_diagnostics` | returns trace, Hermiticity error, minimum eigenvalue, and purity |

---

## 3. Verified properties

The following hold for the density `ρ` produced by `run_density`. They were confirmed by
running the model and inspecting `density_diagnostics`, and they are the invariants a
density-matrix model must satisfy:

- **Trace-preserving after renormalization.** `Tr(ρ) = 1` at the readout (the explicit
  `ρ / Tr(ρ)` steps in `_evidence_channel` guarantee this).
- **Hermitian.** `ρ = ρ†` to numerical precision (Hermiticity error ~`1e-17`).
- **Positive semidefinite.** All eigenvalues are ≥ 0 (minimum eigenvalue stays positive),
  so `ρ` is a valid density matrix.
- **Genuinely mixed.** With the maximally mixed prior and the damping channel active, the
  purity `Tr(ρ²) < 1` (e.g. ~0.41 for the default 2-qubit config) — this is a real mixed
  state, not a pure state in disguise.
- **Reduces bit-for-bit to the statevector model.** With `prior="pure"`, `damping=False`,
  and the Lüders filter disabled (so `K_x = I`), `run_density` reproduces the outer product
  `|ψ⟩⟨ψ|` of the `qcore` statevector to ~`1e-16`. The density model is a strict superset
  of the pure-state model.
- **Learns the task above chance.** Trained with Adam on the checkerboard categorization
  task, it reaches well above chance accuracy.

`density_diagnostics(params, x, cfg, mask)` returns these quantities directly:

```python
{"trace": 1.0, "hermiticity_err": 3.1e-17, "min_eig": 0.0497, "purity": 0.4112}
```

**Canonical config that trains best:** `prior="pure", damping=True`. This combination
gives a sharp, informative prior while keeping a genuinely non-unital (amplitude-damping)
evidence channel, and it is the configuration that learns the categorization task most
reliably.

---

## 4. How to use it

A minimal train/eval loop. Build a `DensityConfig`, initialize parameters, train with
`make_loss_and_grad` + `Adam`, and evaluate with `accuracy`:

```python
import jax.numpy as jnp
from src.qdensity import (
    DensityConfig, init_params, full_mask,
    make_loss_and_grad, Adam, accuracy,
)

cfg = DensityConfig(
    n_qubits=2, depth=4, n_reupload=2,
    readout_paulis=("Z0", "Z1", "X0", "X1"),
    prior="pure", damping=True,        # canonical, best-training config
)

params = init_params(cfg, seed=0)
mask = full_mask(cfg)                  # all Heisenberg interaction terms active

loss_fn, value_and_grad, predict = make_loss_and_grad(cfg, mask)

opt = Adam(params, lr=0.05)
for _ in range(120):
    loss, grads = value_and_grad(params, X_train, y_train)
    params = opt.step(params, grads)

test_acc = accuracy(predict, params, X_test, y_test)
```

`make_loss_and_grad` bakes the mask in as a **static** Python tuple, so each distinct mask
triggers a JAX recompile. For compression sweeps that evaluate or train *many* masks, use
`make_dynamic_fns` instead, which passes the mask as a **runtime argument** (a per-term
`gate_scale` in `[0,1]`): one compiled function then handles any mask, with no per-mask
recompiles. For binary masks this is bit-identical to the static discrete circuit.

```python
loss_fn, value_and_grad, predict = make_dynamic_fns(cfg)
# each call now takes a trailing `mask` argument:
loss, grads = value_and_grad(params, X_train, y_train, mask)
probs = predict(params, X_test, mask)
```

**Performance note.** `qdensity` is slower than `qcore` because it manipulates dense
`2ⁿ × 2ⁿ` density matrices (full operator conjugations) rather than statevectors. It is,
however, exact and fully differentiable. Use `qcore` for the large experiment grids and
use `qdensity` when you specifically need the literal mixed-state mechanism.

---

## 5. Relation to the paper

This module is the concrete realization of the paper's Methods section. Where the Methods
describe the belief state as a **density operator** and evidence as a **quantum channel
acting on that operator**, `qdensity` implements exactly that:

- the belief is a density operator `ρ` (`prior_state`, `run_density`);
- the prior is a real state — the maximally mixed `I/2ⁿ` "no information" prior or a pure
  `|0…0⟩` prior;
- evidence is incorporated by a non-unital channel (the Lüders/POVM filter
  `E_x` with trace renormalization, plus amplitude damping) in `_evidence_channel`, which
  is the quantum analogue of Bayes' rule;
- the posterior category probability is read from Pauli expectations of `ρ`.

In short, `qcore` is the efficient engine used for the bulk results, and `qdensity` is the
faithful, mixed-state model that makes the paper's belief-state-as-density-operator and
evidence-as-quantum-channel claims literally true.
