# Quantum Bayesian Learner with Hardware-Aware Circuit Compression

A hybrid quantum–classical project studying how a **hardware-aware quantum
Bayesian learner** trades classification accuracy against entangling-circuit
complexity under realistic NISQ hardware constraints. The learner performs
Pothos–Chater–style similarity categorization on a 2-feature toy task; we then
**compress its entangling structure** and measure how accuracy degrades against a
**real, post-transpile two-qubit gate count on IBM's FakeManila** backend.

**Reference:** Pothos, E. M., & Chater, N. (2002). A simplicity principle in
unsupervised human categorization. *Cognitive Psychology*, 45, 45–85.

> **Note on this README.** The project was overhauled into a "v2" pipeline. The
> learning core was rebuilt from scratch as an exact, differentiable JAX
> statevector learner (`src/qcore.py`), the hardware cost is now a genuine
> FakeManila transpile (`src/hardware_cost.py`), and all experiments run through
> `src/run_v2.py` writing to `results_v2/`. This README documents that current
> pipeline. (Earlier finite-difference code under `src/learner.py`,
> `src/train_*.py`, etc. is superseded and kept only for reference; do not use
> it.)

---

## Table of Contents

1. [Overview & Purpose](#1-overview--purpose)
2. [Environment Setup](#2-environment-setup)
3. [Architecture (file by file)](#3-architecture-file-by-file)
4. [How to Reproduce](#4-how-to-reproduce)
5. [Interpreting the Results](#5-interpreting-the-results)
6. [The Paper](#6-the-paper)
7. [Running on Real Hardware](#7-running-on-real-hardware)

---

## 1. Overview & Purpose

The thesis is unchanged from the project's inception: build a small **quantum
Bayesian learner** that does Pothos–Chater–style categorization, and study **how
far its entangling structure can be compressed before accuracy collapses**, using
a *real* hardware cost rather than an idealized gate count.

Concretely, the v2 pipeline provides:

1. **An exact, differentiable learner.** A 2-qubit JAX statevector model that
   starts from the pure prior `|0…0⟩`, re-uploads the input features, applies
   trainable **Heisenberg** entanglers (`RXX·RYY·RZZ`), reads out trainable
   single-qubit Pauli expectations, and is trained with **exact autodiff
   gradients** (no finite differences). It genuinely learns the task.
2. **A real hardware-aware cost.** The number of two-qubit operations `N₂q`
   **after transpiling the circuit onto IBM's `FakeManila` coupling map**
   (including any SWAPs the transpiler inserts), computed at runtime, not
   hardcoded.
3. **A compression study.** A per-interaction-term binary mask over the
   entanglers, swept by **greedy structured pruning**, traces an
   accuracy-vs-`N₂q` Pareto frontier. A **matched random-mask ablation** shows
   that the *learned* (greedy) sparsity beats unstructured sparsity at equal gate
   budget.
4. **A difficulty knob.** A checkerboard categorization concept whose spatial
   frequency controls task complexity (easy/medium/hard), producing a *family* of
   frontiers.

---

## 2. Environment Setup

The project targets **Python 3.14** with a local virtual environment.

```bash
cd qb-learner-compression
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Why the dependency stack changed

The original repo pinned `qiskit==1.1.0`, **which does not build on Python 3.14**
(the only interpreter available on the development machine). `requirements.txt`
was therefore re-pinned to a current, tested stack:

| Package | Version | Role |
|---|---|---|
| `qiskit` | 2.4.2 | circuit construction + transpile |
| `qiskit-aer` | 0.17.2 | simulator backend |
| `qiskit-ibm-runtime` | 0.47.0 | provides `FakeManilaV2` (hardware snapshot) |
| `jax`, `jaxlib` | 0.10.2 | exact-autodiff statevector learner |
| `numpy` | 2.4.6 | arrays |
| `scipy` | 1.17.1 | numerics |
| `scikit-learn` | 1.9.0 | stratified train/test split |
| `pandas` | 3.0.3 | result logging / aggregation |
| `matplotlib` | 3.11.0 | figures |
| `tqdm`, `pyyaml` | — | misc utilities |

The **learned model runs entirely in JAX** (exact double-precision autodiff);
**Qiskit is used only for the FakeManila post-transpile hardware cost**, so the
two concerns stay decoupled.

---

## 3. Architecture (file by file)

### `src/qcore.py` — the learner core (single source of truth)

An **exact, differentiable JAX statevector** quantum learner. This file replaces
the old finite-difference / frozen-readout learner entirely.

- **Forward model.** Starts in the pure prior `|0…0⟩` (the honest "no evidence
  yet" state — this avoids the maximally-mixed-invariance problem of the old
  code). It then interleaves **data re-uploading** (`n_reupload`: re-encode the
  features `RY(πxq)·RZ(πxq)` per qubit) with trainable variational layers.
- **Entanglers.** Each layer applies single-qubit `RX·RZ` rotations followed by
  **Heisenberg** two-qubit interactions `RXX·RYY·RZZ` on each edge in
  `cfg.pairs`. Every interaction term is individually gated by a **per-term binary
  mask** of shape `(depth, n_edges, 3)`.
- **Readout.** A **trainable** linear head over Pauli expectations:
  `logit = w · ⟨P⟩ + b`, with `w, b` optimized. The default readout observables
  are configurable (`readout_paulis`); the experiment runner uses single-qubit
  Paulis only so that **entanglement is the only thing that can carry feature
  interaction information**.
- **Gradients & optimizer.** `make_loss_and_grad` builds a jitted numerically
  stable BCE loss and an **exact** `jax.value_and_grad`. A small pytree `Adam`
  optimizer (`qcore.Adam`) trains all parameters jointly.
- **`make_dynamic_fns`** — builds jitted loss/grad/predict functions where the
  **mask is a runtime argument** (a term with `mask=0` becomes `RPP(0)=identity`,
  bit-identical to omitting it). This lets one compiled function evaluate or train
  *any* mask with **no per-mask JAX recompiles** — essential for fast pruning
  sweeps.
- **`make_learned_mask_loss_and_grad`** — a continuous-relaxation "learned mask"
  with a differentiable `λ·N₂q` penalty. **Kept for reference but not used: it is
  unstable** (Adam drives all keep-probabilities to 0 before the circuit learns to
  use the gates). Greedy structured pruning is used instead.

The simulator is verified bit-for-bit against Qiskit (`scripts/verify_qcore.py`).

### `src/qdensity.py` — the literal mixed-state density-matrix learner

The principled realization of the quantum-Bayesian mechanism with **no pure-state
shortcut**: the belief is a genuine **density operator** `ρ`, the prior is the
**maximally mixed** state `I/2ⁿ` (a uniform "no information" belief), and the evidence
update is a **non-unital quantum channel with trace renormalization** — a
stimulus-dependent Lüders/POVM filter `Kₓ ρ Kₓ†` then `ρ/Tr(·)` (the Bayesian update),
plus per-qubit amplitude damping (a multi-Kraus, non-unital CPTP map). The readout is
`Tr[ρP]` through the same trainable head. It is exact, differentiable (dense JAX matrix
ops), reduces bit-for-bit to `qcore` when the channel is off, and is verified mixed
(purity < 1), Hermitian, PSD, trace-one. Used to confirm the frontier and learned-vs-random
results hold for the principled model (`src/run_density.py`, `tests/test_qdensity.py`).
**Full write-up in [`docs/DENSITY_MODEL.md`](docs/DENSITY_MODEL.md).**

### `src/hardware_cost.py` — genuine FakeManila two-qubit cost

Computes the real, post-transpile hardware cost:

- Instantiates `FakeManilaV2` (a 5-qubit IBM device snapshot, linear coupling map
  `0–1–2–3–4`).
- `build_cost_circuit(...)` rebuilds the **exact Qiskit circuit mirroring the
  qcore ansatz** (Heisenberg `RXX/RYY/RZZ`, selected by the per-term mask; HEA
  `CX-RY-CX` variant also supported). Angles are placeholders since the 2q **count
  depends only on circuit structure**, not angle values.
- `transpiled_2q_cost(...)` transpiles onto FakeManila and counts all two-qubit
  operations **including SWAPs** inserted to satisfy the coupling map. Results are
  **cached by a structural key and deterministic** (fixed `seed_transpiler`), so
  repeated calls during pruning are cheap and reproducible. The count is provably
  mask-dependent (more active terms ⇒ higher `N₂q`).

### `src/data.py` — datasets with a difficulty knob

- **`get_pothos_chater_checker(freq, …)`** — the checkerboard categorization task:
  `label = (⌊freq·x₀⌋ + ⌊freq·x₁⌋) mod 2`. Higher `freq` ⇒ finer cells ⇒ a more
  intricate boundary that needs more entangling capacity. This is the
  **difficulty knob**, in the spirit of Pothos–Chater local-similarity grouping.
- **`CHECKER_DIFFICULTY`** maps `easy / medium / hard → freq 2 / 3 / 4`;
  `get_checker_difficulty(level, …)` is the convenience accessor used by the
  runner. (Empirically, `freq=4` gives a clear *graded* decline with a visible
  knee rather than an all-or-nothing cliff.)
- **Legacy generators** retained for reference: the original Gaussian-cluster
  `get_pothos_chater_{small,medium,large}` plus the XOR/parity generators
  (`get_pothos_chater_xor`, `get_pothos_chater_parity`) and their difficulty maps.

### `src/run_v2.py` — the experiment runner (Phase 3/4)

The canonical pipeline. Uses `qcore` for learning and `hardware_cost` for the
cost, with a **fixed canonical config** (2 qubits, depth 4, one edge,
`n_reupload=2`, single-qubit readout `("Z0","Z1","X0","X1")`, Heisenberg
entanglers).

- **Full baseline** — trains the full circuit (warm-up).
- **Greedy frontier** (`greedy_trajectory`) — repeatedly drops the single
  interaction term whose removal least increases training CE, **warm-start
  retraining** at each budget, walking the mask from full down to no entanglers.
  One trajectory = the accuracy-vs-`N₂q` frontier.
- **λ operating points** — for each `λ`, picks the trajectory point minimizing
  `train_CE + λ·N₂q`.
- **Matched random-mask ablation** — at each pruned budget, trains several random
  masks with the same number of active terms (connectivity-matched control).
- **Logging** — writes per-difficulty CSVs to
  `results_v2/<difficulty>/{frontier,lambda_sweep,mask_ablation}.csv`,
  incrementally per seed. All reported metrics use the **discrete deployed
  circuit** and the **real FakeManila `N₂q`**.

```bash
python -m src.run_v2 --difficulty hard --seeds 0 1 2 3 4
python -m src.run_v2 --all --seeds 0 1 2 3 4    # easy, medium, hard
```

### `scripts/` — verification, smoke checks, figures, design probes

- **`verify_qcore.py`** — checks the qcore statevector against Qiskit over 20
  random circuits (max `|1 − fidelity|` ≈ 4e-16, accounting for global phase /
  endianness). Run this first to confirm the simulator is correct.
- **`train_quick.py`** — the Phase-1 Definition-of-Done check: trains the
  diagnostic baseline (`λ=0`, full mask) and confirms it learns well above chance
  (reaches train/test ≈ 1.0 on the well-separated large dataset).
- **`make_figures_v2.py`** — aggregates the `results_v2/<difficulty>/*.csv` logs
  into the paper's figures (`results_v2/figures/`) and **LaTeX tables**
  (`results_v2/tables/`). Every reported number is computed here from logged runs
  — nothing is hand-entered.
- **`probe_*.py`** (`probe_checker.py`, `probe_knee.py`, `probe_knee3.py`,
  `probe_concentric.py`) — dataset-design probes used to choose the checkerboard
  task as the difficulty knob (XOR/parity gave a cliff, concentric gave no knee,
  checkerboard `freq=4` gave a graded frontier).

### `tests/` — pytest suite

`tests/test_qcore.py` covers: simulator-vs-Qiskit agreement, the dynamic
(runtime) mask matching the static discrete mask bit-for-bit, training reducing
the loss, and `full_mask` shape/values. Run:

```bash
python -m pytest tests/ -q
```

---

## 4. How to Reproduce

Copy-pasteable, start to finish:

```bash
# 1. Environment
cd qb-learner-compression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify the simulator is correct (qcore vs Qiskit)
python scripts/verify_qcore.py

# 3. (optional) sanity-check that the baseline learns + run the test suite
python scripts/train_quick.py
python -m pytest tests/ -q

# 4. Run the full sweep: greedy frontier + lambda sweep + mask ablation,
#    across all difficulties (easy/medium/hard), 5 seeds each
python -m src.run_v2 --all --seeds 0 1 2 3 4

# 5. Aggregate logs into figures + LaTeX tables
python scripts/make_figures_v2.py
```

Outputs land in:

- `results_v2/<difficulty>/{frontier,lambda_sweep,mask_ablation}.csv` — raw logs
- `results_v2/figures/` — `frontier_family.png`, `frontier_ce.png`,
  `mask_ablation.png`
- `results_v2/tables/` — `frontier.tex`, `lambda_sweep.tex`, `mask_ablation.tex`

### One-shot reproduction

`scripts/reproduce_paper.sh` runs the whole simulation pipeline end to end (sanity
check → pure-state sweeps → density-matrix sweeps → noise model → classical baselines
→ paired statistics → all figures). Add `--hardware` to also re-run the `ibm_fez`
device experiments (needs IBM credentials).

```bash
bash scripts/reproduce_paper.sh            # all simulation results + figures
bash scripts/reproduce_paper.sh --hardware # also the ibm_fez device runs
```

### Paper artifact → command map

Every figure and table in `paper/conference_101719.tex` is produced by the commands
below; raw numbers live next to the figures so the PDF, repo, and logs stay in lockstep.

| Paper artifact | Produced by | Data |
|---|---|---|
| Fig. 1 (frontier family) | `src.run_v2` → `make_figures_v2.py` | `results_v2/<diff>/frontier.csv` |
| Tab. I (λ sweep) | `src.run_v2` | `results_v2/hard/lambda_sweep.csv` |
| Fig. 2 + Tab. II (learned vs random) | `src.run_v2` → `make_figures_v2.py` | `results_v2/hard/mask_ablation.csv` |
| Fig. 3 (noise robustness) | `scripts/noise_robustness.py` | `results_v2/hardware/noise_robustness.csv` |
| Fig. 4 (device frontier) | `scripts/hardware_frontier.py` | `results_v2/hardware/ibm_fez_frontier.json` |
| Abstract/§IV-E endpoints | `scripts/run_on_hardware.py --mode ibm` | `results_v2/hardware/ibm_fez_results.json` |
| Fig. 5, 6 + Tab. III (density) | `src.run_density` → `make_figures_density.py` | `results_v2_density/<diff>/` |
| Tab. V (classical baselines, App. B) | `scripts/classical_baselines.py` | `results_v2/classical_baselines.csv` |
| §IV-C paired statistic | `scripts/paired_stats.py hard` | `results_v2/hard/mask_ablation.csv` |

Device-run provenance (backend, date, shots, transpiler settings) is recorded in
`results_v2/hardware/RUN_METADATA.md` and mirrored in the paper appendix.

---

## 5. Interpreting the Results

The specific numbers come from the run and depend on seeds, so they are not
hardcoded here. Qualitatively, you should expect:

- **An accuracy-vs-`N₂q` frontier with a knee.** As the greedy trajectory removes
  entangling terms, accuracy stays high for a while, then **degrades gradually**
  past a knee — a smooth trade-off, not a cliff (this is why the checkerboard task
  was chosen).
- **Entanglement is necessary.** At `N₂q = 0` (all entanglers removed) the model
  **collapses to roughly chance accuracy** — the single-qubit-only readout cannot
  represent the feature interaction the boundary requires without entanglers.
- **Learned masks beat random masks at matched budget.** At equal `N₂q`, the
  greedy-pruned (structured) masks achieve higher accuracy than connectivity-
  matched random masks, showing that *which* entanglers are kept matters, not just
  how many.
- **A family of frontiers by difficulty.** The checkerboard frequency
  (`easy/medium/hard` = `freq 2/3/4`) shifts the whole frontier: harder tasks need
  more entangling capacity to reach the same accuracy.

---

## 6. The Paper

The write-up lives in `paper/` (`paper/conference_101719.tex`, IEEEtran). It is
intended to be **regenerated from `results_v2/`**: the figures and tables emitted
by `scripts/make_figures_v2.py` are the authoritative source, so no quantitative
claim is hand-typed. Per the project plan, the paper is finalized **last**, after
the `results_v2` sweeps are complete.

---

## 7. Running on Real Hardware

The model **trains in exact simulation** and is **deployed for inference on hardware**.
`src/hardware_backend.py` builds a Qiskit circuit that is bit-identical to the trained
`qcore` model (verified to `< 1e-9` in `tests/test_hardware_backend.py`), estimates the
readout observables on a chosen backend, and applies the trained head. Three backends:

- `exact` — `StatevectorEstimator` (noiseless; equals the simulator).
- `noisy` — `AerSimulator.from_backend(FakeManila)`, the real device **noise model** (free).
- `ibm` — a real IBM Quantum device via `QiskitRuntimeService` (needs an API token).

```bash
# free: validate under the FakeManila noise model (full vs compressed circuit)
python scripts/run_on_hardware.py --difficulty hard --mode noisy --shots 4096
# how the whole frontier holds up under device noise (5 seeds -> figure + CSV)
python scripts/noise_robustness.py
# real IBM hardware (after setting up an account — see docs/HARDWARE.md)
python scripts/run_on_hardware.py --difficulty hard --mode ibm --n-eval 20
```

Across five seeds, accuracy under the noise model tracks the noiseless frontier within a
few percent at every two-qubit budget, so the compression savings are real net of noise.

**Validated on a real device.** Running on the physical IBM processor `ibm_fez`
(156 qubits), the full circuit (`N2q=24`) collapses to near chance (**0.60**) while the
greedy-compressed circuit (`N2q=6`) holds at **0.825** — on real hardware, structured
compression is not just free but *necessary*. The full on-device frontier
(`scripts/hardware_frontier.py`) makes this vivid: device accuracy falls *monotonically*
with `N2q` (0.81 at `N2q=2,6` → 0.56 at `N2q=18,24`), the opposite of the flat simulated
frontier, because extra two-qubit gates accumulate noise. Numbers in
`results_v2/hardware/`; figures `hardware_frontier.png` / `hardware_real.png` (regenerate
with `scripts/hardware_frontier.py` / `scripts/plot_hardware.py`). First, sanity-check your
connection:
```bash
python scripts/check_ibm_connection.py        # lists devices + least busy
```
**Full connection instructions (IBM Quantum Platform token/instance setup, costs,
troubleshooting) are in [`docs/HARDWARE.md`](docs/HARDWARE.md).**

---

## Contact

Mohammad Zoraiz — mz248@duke.edu
