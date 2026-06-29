# POSTER MASTER SPEC — single source of truth

Build a **36 in wide × 48 in tall PORTRAIT** academic research poster, 3 columns,
Duke-branded. Every number below is taken verbatim from the final paper
(`conference_101719.tex`). DO NOT invent or alter numbers. Keep all text concise and
scannable (poster, not paper). Target ~700–900 words total in body.

================================================================================
## DESIGN SYSTEM
================================================================================
- Orientation/size: 36" W × 48" H, portrait. 3 equal columns.
- Color palette (Duke):
  - Duke Navy  #012169  → header band, section-header bars, table header row
  - Duke Royal #00539B  → secondary accents, links, rules
  - Amber/Gold #E89923  → ONE highlight only: the hardware-reversal callout box border/accent
  - Box tint   #EEF2F8  → very light blue-gray fill for content boxes
  - Text       #1A1A1A on white; white text on navy
- Fonts: sans-serif throughout (Helvetica/Arial). Title bold. Section headers bold white on navy bars.
- Body text must read from ~5 ft: body ≥ 24pt-equivalent, section heads larger, title largest.
- Layout: navy HEADER BAND across full width at top (~5 in tall) holding title + author + Duke logo.
  Then 3 columns. Thin footer OR place refs/QR/ack at bottom of column 3.
- Section headers: short navy bar with white text, optional small number (1–8) for reading flow.
- Use the 3 result figures large; they are the stars. Balance ~45% visual.

## ASSETS (absolute paths)
- Duke/ECE logo (WHITE knockout, put on navy header): assets/duke_ece_logo.png
- Repo QR (navy on white): assets/repo_qr.png
- Figures (matplotlib, 1000×720 px, 200 dpi):
  - figures/fig_frontier_family.png   (Result 1 — capacity boundary)
  - figures/fig_mask_ablation.png     (Result 2 — structure beats count)
  - figures/fig_hardware_frontier.png (Result 3 — real-hardware reversal; CENTERPIECE)
  - figures/fig_noise_robustness.png  (optional, robustness)
  - figures/fig_density_frontier.png  (optional, robustness)

================================================================================
## HEADER
================================================================================
TITLE: Hardware-Aware Quantum Bayesian Learner Compression:
       Linking Human Belief Updating to NISQ Circuit Complexity
AUTHOR: Mohammad Zoraiz
AFFIL: Duke University · Pratt School of Engineering · Electrical Engineering, Physics & Computer Science · Durham, NC
CONTACT: mz248@duke.edu
VENUE TAG (small): IEEE Quantum Week — QCE26 Poster
LOGO: Duke/ECE lockup (white) at right of header band.

================================================================================
## COLUMN 1
================================================================================

### [1] Motivation — Two fields, one question
- Bayesian models cast human learning as **belief updating**: a prior over hypotheses,
  revised by evidence. **Quantum cognition** represents beliefs as density operators in
  Hilbert space, with evidence acting through **quantum channels**.
- On **NISQ** hardware, **two-qubit gates dominate the error budget**, so removing them
  without breaking the task is a central engineering goal.
- **Central question:** *How much of a quantum Bayesian learner's entangling structure can
  we remove before it can no longer represent the task — and what happens on real hardware?*

### [2] The Quantum Bayesian Learner  (SCHEMATIC HERE)
- Belief = state ρ on a **2-qubit** register; prior ρ₀ = "no evidence yet."
- A stimulus x updates the belief via a **CPTP evidence channel**
  𝓔ₓ(ρ) = Σₖ Eₓₖ ρ Eₓₖ†.
- Update circuit = **hardware-efficient ansatz**: data re-uploading + per-layer single-qubit
  Rₓ, R_z + **Heisenberg entanglers** U_ij = exp[−i(α XᵢXⱼ + β YᵢYⱼ + γ ZᵢZⱼ)]
  as native R_XX, R_YY, R_ZZ. Depth **D = 4**, 2 re-uploads.
- **Readout: single-qubit Pauli expectations** v = (⟨Z₀⟩,⟨Z₁⟩,⟨X₀⟩,⟨X₁⟩) → trainable head
  ŷ = σ(w·v + b). Single-qubit-only readout means **entanglement is the ONLY way to carry
  feature interaction** — so the two-qubit budget literally bounds the learner's capacity.
- Two realizations, same conclusions: a fast **pure-state** model and a literal
  **mixed-state density matrix** (maximally mixed prior I/2ⁿ, genuinely **non-unital**
  Lüders/POVM evidence filter + amplitude damping, Bayesian trace renormalization).

SCHEMATIC to draw (left→right pipeline):
  [checkerboard stimulus x] → [Prior ρ₀] → [ Re-upload RY(πx)·RZ(πx) | Rx,Rz | masked
  Heisenberg R_XX·R_YY·R_ZZ on edge ] repeated ×D=4 → [single-qubit Pauli readout] →
  [trainable head σ(w·v+b) → ŷ]. Show a small 12-cell MASK grid (some cells ON/OFF) gating
  the entanglers, and an arrow "transpile → IBM FakeManila → N₂q" coming off the entanglers.

### [3] Method — mask, real cost, greedy compression
- **Per-term binary mask** m_{ℓ,p} ∈ {0,1} over p ∈ {XX,YY,ZZ} in each of D=4 layers ⇒
  **12 maskable terms** (R_PP(0)=I when off).
- **Hardware-aware cost N₂q** = real **post-transpile two-qubit gate count** on IBM
  **FakeManila** (5-qubit, includes SWAPs), computed by actually invoking Qiskit's
  transpiler. Full circuit ⇒ **N₂q = 24** (each active Heisenberg term = 2 CNOTs).
- **Objective:** L(θ,m) = (1/N) Σ BCE(yₖ, ŷ) + λ·N₂q(m).
- **Greedy structured pruning:** from the full circuit, repeatedly drop the interaction term
  whose removal least increases training cross-entropy, **warm-start retrain** → traces the
  accuracy-vs-N₂q **frontier** from 24 down to 0.
- Exact **JAX autodiff** (no parameter-shift / finite differences); statevector verified
  **bit-for-bit vs Qiskit** (|1 − fidelity| ≈ 4×10⁻¹⁶).

### Task — controllable difficulty knob
- **Checkerboard** categorization: y = (⌊f·x₀⌋ + ⌊f·x₁⌋) mod 2, stimuli in [0,1]².
- Difficulty f: **f=2 easy / f=3 medium / f=4 hard**. 200 stimuli, 70/30 split, balanced
  ⇒ **chance = 0.5**. Local similarity, but the rule depends on the **joint** features ⇒
  needs the interaction only entanglement can supply.

================================================================================
## COLUMN 2  (RESULTS)
================================================================================

### [4] Result 1 — The capacity boundary   [fig_frontier_family.png, large]
- Above chance at all nonzero budgets (easy/medium ≈ **0.95–0.97**, hard ≈ **0.87–0.93**);
  frontier **nearly flat** → most entangling structure is **redundant**.
- At **N₂q = 0** all difficulties **collapse toward chance** (0.52 easy, 0.64 medium,
  0.49 hard) → some entanglement is **necessary**, but far less than the full circuit.
- Hard task (not saturated): accuracy **peaks at intermediate cost** (**0.927 @ N₂q=6**)
  and is **lower for the full circuit** (**0.873 @ N₂q=24**) → moderate compression acts as
  a capacity-control **regularizer**.

### [5] Result 2 — Structure beats count   [fig_mask_ablation.png]
- Learned (greedy) masks vs **random masks at matched N₂q**.
- Learned wins at **every** budget by **~6–12 points**. Per seed: learned beats random in
  **49 / 55** paired (seed,budget) comparisons, mean advantage **+0.083**, one-sided
  Wilcoxon **p < 10⁻⁷**.
- ⇒ **Which** entanglers you keep matters, not just how many.

TABLE (hard task — learned vs random, mean acc, 5 seeds):
| N₂q | Learned | Random | Δ |
| 2 | 0.847 | 0.742 | +0.104 |
| 6 | 0.927 | 0.814 | +0.112 |
| 10 | 0.913 | 0.794 | +0.119 |
| 14 | 0.917 | 0.826 | +0.091 |
| 18 | 0.900 | 0.809 | +0.091 |
| 22 | 0.897 | 0.836 | +0.061 |

(Optional compact λ-sweep table if space — hard task: λ=0 → N₂q 9.6, acc 0.923;
λ=0.04 → N₂q 3.2, acc 0.933; λ=0.15 → N₂q 1.6, acc 0.823. "Sheds ~2/3 of 2-qubit gates
for free, >90% at modest cost.")

================================================================================
## COLUMN 3
================================================================================

### [6] Result 3 — On REAL hardware, compression becomes NECESSARY
        [fig_hardware_frontier.png — CENTERPIECE, largest; amber-accent callout box]
- Trained circuits run on physical IBM **ibm_fez** (156-qubit Heron), hard task, 5 device
  budgets, **16 test stimuli @ 2048 shots** (endpoints: 40 stimuli @ 4096 shots).
- In simulation the frontier is flat; **on hardware it falls monotonically** with N₂q:
  **0.81 @ N₂q=2,6 → 0.75 @ 12 → collapses to 0.56** (near chance) **@ N₂q=18,24.**
- Endpoint confirmation: **full circuit 0.60** (sim 0.95) vs **compressed 0.83** (sim 0.93).
- **Takeaway (callout):** On real NISQ hardware, hardware-aware compression of the quantum
  Bayesian learner is not merely economical but **necessary** — surplus entangling capacity
  is actively harmful, and task-aware pruning is what makes the learner usable at all.

### [7] Robustness — the result is not an artifact
- **Device noise model** (FakeManila, 4096 shots, 5 seeds): noisy accuracy tracks the
  noiseless frontier within a few % at every budget → savings genuine net of noise.
- **Density-matrix model** (genuine mixed state, non-unital channel): reproduces every
  finding — flat frontier, collapse at N₂q=0, hard peak **0.88 @ 6 vs 0.87 @ 24**, learned >
  random (mean **+0.043**). Conclusions hold for the principled belief-update mechanism.
- **Classical baselines** (identical splits, 5 seeds): logistic regression at chance on
  every level (confirms the joint-feature rule needs entanglement); RBF-SVM & MLP fall
  monotonically easy→hard. Best compressed QBL: **easy 0.970, medium 0.973, hard 0.927**.

### [8] Key takeaways
- Most entanglement is **redundant** — sheds ~**2/3** of two-qubit gates for **free**,
  **>90%** at modest cost.
- Entanglement is **necessary** — removing all of it collapses the learner to chance:
  a concrete **capacity boundary**.
- **Structure beats count** — learned masks beat random at every matched budget.
- **On real hardware, compression is what makes the learner work at all.**

### Future work (1 line)
Multi-qubit belief registers & higher-dim stimuli; richer non-unital evidence channels;
hardware-aware objectives weighting depth, coherence & native fidelity; feed measured device
error rates back into the compression objective.

### References (key — abbreviated)
[1] Busemeyer & Bruza, *Quantum Models of Cognition and Decision*, 2012.
[2] Pothos & Chater, "A simplicity principle in unsupervised human categorization," *Cogn. Psych.*, 2002.
[3] Kandala et al., "Hardware-efficient VQE…," *Nature*, 2017.
[4] Preskill, "Quantum computing in the NISQ era and beyond," *Quantum*, 2018.
[5] Cerezo et al., "Variational quantum algorithms," *Nat. Rev. Phys.*, 2021.
[6] Sim et al., "Adaptive pruning-based optimization of PQCs," *Quantum Sci. Technol.*, 2021.

### Acknowledgments + links
- Acknowledgments: "Device runs used IBM Quantum services. The author thanks Duke University's
  Pratt School of Engineering." (NO external funding claimed — solo author.)
- QR (assets/repo_qr.png) → github.com/zoraizmohammad/qb-learner-compression
- Contact: Mohammad Zoraiz · mz248@duke.edu

================================================================================
## QUALITY BAR
================================================================================
- All numbers EXACTLY as above. Title/author exact. Chance line = 0.5.
- No placeholder/lorem text. No PhDPosters watermark. Spell-check.
- Compiles/renders clean at 36×48 portrait; no overflow, no overlapping shapes,
  consistent column widths and gutters, generous whitespace.
- Render to PNG and visually confirm before declaring done.
