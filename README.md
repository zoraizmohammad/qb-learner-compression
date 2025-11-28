# Quantum Bayesian Learner with Hardware-Aware Circuit Compression

A hybrid quantum–classical project exploring how a quantum Bayesian learner's performance depends on variational circuit complexity under realistic NISQ hardware constraints.

**Reference:** Pothos, E. M., & Chater, N. (2002). A simplicity principle in unsupervised human categorization. *Cognitive Psychology*, 45, 45–85.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Experimental Pipeline](#experimental-pipeline)
3. [How to Run Everything](#how-to-run-everything)
4. [Output Structure](#output-structure)
5. [Notebooks](#notebooks)
6. [Expected Results](#expected-results)
7. [Reproduction Steps for Paper Figures](#reproduction-steps-for-paper-figures)

---

## Repository Structure

### Directory Tree

```
qb-learner-compression/
│
├── src/                          # Core source code
│   ├── data.py                   # Dataset generation (Pothos-Chater style)
│   ├── channels.py               # Quantum evidence channels (Kraus operators)
│   ├── ansatz.py                 # Variational circuit construction
│   ├── learner.py                # Core learning logic & loss computation
│   ├── train_baseline.py         # Baseline training script
│   ├── train_compressed.py       # Compressed training with pruning
│   ├── evaluate_model.py         # Model evaluation & comparison
│   ├── run_experiment.py         # Single experiment launcher
│   ├── run_all_experiments.py    # Batch experiment runner
│   ├── transpile_utils.py        # Gate counting & transpilation
│   ├── plots.py                  # Visualization utilities
│   └── logging_utils.py          # Logging & file I/O utilities
│
├── experiments/                  # YAML experiment configurations
│   ├── baseline_pothos_large.yaml
│   ├── compressed_pothos_large.yaml
│   └── ... (various configs)
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── sanity_checks.ipynb       # Component verification
│   ├── visualize_channels.ipynb  # Channel visualization
│   └── visualize_ansatz.ipynb    # Circuit structure visualization
│
├── results/                      # All experiment outputs
│   ├── logs/                     # Training logs (CSV)
│   ├── runs/                     # Individual run directories
│   ├── experiments_<timestamp>/  # Batch experiment results
│   │   ├── experiments/          # Individual experiment outputs
│   │   └── summary/              # Aggregated summaries
│   └── figures/                  # Generated plots
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## File-by-File Documentation

### Core Source Files (`src/`)

#### `src/data.py`
**Purpose:** Generates Pothos-Chater-style toy categorization datasets.

**What it does:**
- Creates three dataset sizes: small (8 points), medium (20 points), large (40 points)
- Each dataset has two Gaussian clusters in 2D feature space
- Features are normalized to [0, 1] for quantum channel encoding

**Key Functions:**
- `get_pothos_chater_small()` → (X, y) with 8 samples
- `get_pothos_chater_medium()` → (X, y) with 20 samples
- `get_pothos_chater_large()` → (X, y) with 40 samples (recommended for paper)
- `get_toy_dataset(name)` → Generic entry point

**Integration:**
- Called by training scripts (`train_baseline.py`, `train_compressed.py`)
- Used in notebooks for visualization
- Automatically loaded via YAML configs in `run_experiment.py`

**Pipeline Position:** First step in experimental pipeline (data generation)

---

#### `src/channels.py`
**Purpose:** Implements quantum evidence channels (Kraus operators) that update the learner's belief state.

**What it does:**
- Builds Kraus operators from stimulus features
- Maps feature values to channel strength parameters
- Supports multiple channel types: amplitude damping, phase damping, rotation, projective update

**Key Functions:**
- `evidence_amplitude_damping(x, strength)` → Amplitude-damping channel
- `evidence_projective_update(x, strength)` → Bayesian-like projective update (default)
- `build_evidence_channel(x, kind, strength)` → Unified channel selector
- `compute_strength_from_stimulus(x, method)` → Feature-to-strength mapping
- `apply_channel_to_density_matrix(rho, channel)` → Apply channel to state

**How it works:**
1. Takes stimulus features `x` (1D array)
2. Computes channel strength from features (default: mean of features)
3. Builds Kraus operators based on strength
4. Returns Qiskit `Kraus` channel object

**Integration:**
- Called by `learner.py` in `apply_evidence_channel()`
- Used in notebooks for visualization
- Channel strength affects learning dynamics (higher = stronger evidence)

**Pipeline Position:** Applied during forward pass (after ansatz, before prediction)

---

#### `src/ansatz.py`
**Purpose:** Constructs parameterized variational quantum circuits with masked entangling gates.

**What it does:**
- Builds layered circuits: single-qubit rotations (RX, RZ) + entangling blocks
- Supports two entangler types:
  - **HEA-style:** CX, RY, CX (stronger entanglement)
  - **Heisenberg:** RXX, RYY, RZZ (original)
- Uses binary masks to selectively enable/disable entanglers (compression mechanism)

**Key Functions:**
- `build_ansatz(n_qubits, depth, theta, mask, pairs)` → Constructs circuit
- `init_random_theta(n_qubits, depth, n_edges, scale)` → Random parameter initialization
- `init_full_mask(depth, n_edges)` → All gates active (baseline)
- `init_sparse_mask(depth, n_edges, sparsity)` → Random sparse mask
- `count_parameters(theta, mask, n_qubits)` → Counts active parameters
- `get_default_pairs(n_qubits)` → Linear chain connectivity [(0,1), (1,2), ...]

**Parameter Structure:**
- `theta`: shape `(depth, max(n_qubits, n_edges), 5)`
  - `theta[d, q, 0]`: RX angle for qubit `q` at depth `d`
  - `theta[d, q, 1]`: RZ angle for qubit `q` at depth `d`
  - `theta[d, k, 2:5]`: Entangling parameters for edge `k` at depth `d`
- `mask`: shape `(depth, n_edges)` binary array
  - `mask[d, k] = 1` → entangler on `pairs[k]` at depth `d` is active
  - `mask[d, k] = 0` → entangler is pruned (compressed mode)

**Integration:**
- Called by `learner.py` in `forward_loss()` to build circuit
- Used by `train_baseline.py` and `train_compressed.py` for training
- Mask is pruned in `train_compressed.py` via `greedy_prune()`

**Pipeline Position:** Circuit construction (before transpilation and loss computation)

---

#### `src/learner.py`
**Purpose:** Core learning logic: forward pass, loss computation, and prediction.

**What it does:**
- Implements the quantum Bayesian learning workflow:
  1. Initialize belief state (slightly biased toward |0...0⟩)
  2. Apply feature encoding (RY, RZ rotations)
  3. Apply ansatz (unitary transformation)
  4. Apply evidence channel (belief update)
  5. Make prediction (Z expectation + sigmoid)
  6. Compute loss (binary cross-entropy + λ × gate count)

**Key Functions:**
- `forward_loss(theta, mask, X, y, lam, ...)` → Computes hybrid loss
  - Returns: `{"total_loss", "ce_loss", "two_q_cost", "avg_pred", "preds"}`
- `predict_label(rho, readout_alpha)` → Class probability from quantum state
- `predict_hard(rho, threshold)` → Hard class label (0 or 1)
- `init_belief(n_qubits)` → Initial belief state (not maximally mixed)
- `apply_evidence_channel(rho, x, strength, qargs)` → Apply channel to state
- `apply_unitary(rho, qc)` → Apply circuit to density matrix

**Loss Function:**
```
Loss = CE_loss + λ × two_qubit_gate_count
```
- `CE_loss`: Binary cross-entropy between predictions and labels
- `two_qubit_gate_count`: Post-transpile entangling gate count (compression metric)
- `λ`: Regularization strength (controls accuracy vs. complexity trade-off)

**Integration:**
- Called by training scripts in optimization loop
- Used by `evaluate_model.py` for evaluation
- `transpile_utils.py` provides gate counting

**Pipeline Position:** Core of training loop (forward pass + loss computation)

---

#### `src/train_baseline.py`
**Purpose:** Trains baseline model with all entangling gates active (no compression).

**What it does:**
- Optimizes only continuous parameters `theta` (mask fixed at all 1s)
- Uses finite-difference gradient descent or Adam optimizer
- Saves training history, parameters, and plots

**Key Functions:**
- `main(n_qubits, depth, n_iterations, lr, lam, ...)` → Main training function
- `finite_diff_gradient(loss_fn, theta, h)` → Gradient computation
- `AdamOptimizer` → Adam optimizer class

**Outputs:**
- `results/runs/baseline_<timestamp>/`
  - `training_history.csv` → Loss, accuracy, gate count per iteration
  - `params_final.npz` → Final `theta` parameters
  - `loss_history.npz` → Arrays for plotting
  - `final_metrics.json` → Final accuracy, loss, gate count
  - `figures/` → Training curves, Pareto plots

**Integration:**
- Called by `run_experiment.py` when `mode: "baseline"`
- Called by `run_all_experiments.py` for batch runs
- Uses `learner.py` for forward pass, `transpile_utils.py` for gate counting

**Pipeline Position:** Training execution (baseline mode)

**Usage:**
```bash
python -m src.train_baseline --iterations 100 --n_qubits 2 --depth 3 --lr 0.01 --lam 0.1
```

---

#### `src/train_compressed.py`
**Purpose:** Trains compressed model with greedy pruning of entangling gates.

**What it does:**
- Starts with all entanglers active (full mask)
- Alternates between:
  1. Optimizing `theta` via gradient descent
  2. Pruning entanglers that don't significantly impact loss (every `prune_every` iterations)
- Pruning criterion: disable gate if loss increase ≤ `tolerance`

**Key Functions:**
- `main(n_qubits, depth, n_iterations, lr, lam, prune_every, tolerance, ...)` → Main training
- `greedy_prune(theta, mask, X, y, tolerance, ...)` → Pruning logic
- `compute_mask_sparsity(mask)` → Fraction of active entanglers

**Pruning Algorithm:**
1. For each active entangler `(d, k)`:
   - Temporarily set `mask[d, k] = 0`
   - Compute loss with gate disabled
   - If `loss_increase ≤ tolerance` AND `ce_loss_increase ≤ tolerance`:
     - Permanently disable gate
     - Recompute baseline loss
2. Continue until no more gates can be pruned

**Outputs:**
- `results/runs/compressed_<timestamp>/`
  - `training_history.csv` → Includes `mask_sparsity` column
  - `params_final.npz` → Final `theta` and `mask`
  - `mask_history.npz` → Mask evolution over training
  - `figures/` → Includes mask heatmaps

**Integration:**
- Called by `run_experiment.py` when `mode: "compressed"`
- Called by `run_all_experiments.py` for batch runs
- Uses `learner.py` for forward pass, `transpile_utils.py` for gate counting

**Pipeline Position:** Training execution (compressed mode)

**Usage:**
```bash
python -m src.train_compressed --iterations 100 --prune_every 20 --tolerance 0.01
```

---

#### `src/evaluate_model.py`
**Purpose:** Evaluates trained models and generates comparison plots.

**What it does:**
- Loads baseline and compressed models from `results/`
- Recomputes accuracy by re-running predictions
- Generates comparison plots (loss curves, mask visualizations, Pareto fronts)

**Key Functions:**
- `load_training_logs(results_dir)` → Load CSV logs
- `load_compressed_model(results_dir)` → Load `theta` and `mask`
- `compute_accuracy_from_model(theta, mask, ...)` → Re-run predictions
- `generate_comparison_plots(...)` → Create visualization plots

**Outputs:**
- `results/figures/eval_*.png` → Comparison plots

**Integration:**
- Called manually after training
- Uses `learner.py` for predictions, `plots.py` for visualization

**Pipeline Position:** Post-training evaluation

**Usage:**
```bash
python -m src.evaluate_model
```

---

#### `src/run_experiment.py`
**Purpose:** One-click launcher for single experiments from YAML configs.

**What it does:**
- Loads YAML configuration file
- Validates config structure
- Dispatches to `train_baseline.py` or `train_compressed.py` based on `mode`

**Integration:**
- Called manually or by scripts
- Uses `train_baseline.py` or `train_compressed.py` for actual training

**Pipeline Position:** Experiment orchestration

**Usage:**
```bash
python -m src.run_experiment --config experiments/baseline_pothos_large.yaml
```

---

#### `src/run_all_experiments.py`
**Purpose:** Batch runner for multiple experiments with aggregation.

**What it does:**
- Discovers all YAML configs in `experiments/`
- Runs each experiment sequentially
- Aggregates results into summary CSV and comparison plots
- Supports dry-run, limiting, tagging, and skip-failed modes

**Key Functions:**
- `discover_configs(experiments_dir)` → Find all YAML files
- `normalize_config(config)` → Apply defaults and validate
- `run_single_experiment(config, ...)` → Run one experiment
- `aggregate_results(all_summaries, summary_dir)` → Generate summaries

**Outputs:**
- `results/experiments_<tag>_<timestamp>/`
  - `experiments/` → Individual experiment outputs
  - `summary/` → Aggregated CSV and plots
    - `all_experiments_summary.csv` → Table of all results
    - `accuracy_vs_gatecount.png` → Scatter plot
    - `baseline_vs_compressed.png` → Bar chart comparison
    - `experiment_methods_comparison.png` → Method comparison

**Integration:**
- Called manually for batch runs
- Uses `train_baseline.py` and `train_compressed.py` for training
- Uses `plots.py` for aggregation plots

**Pipeline Position:** Batch experiment orchestration

**Usage:**
```bash
# Run all experiments
python -m src.run_all_experiments

# Dry run (list configs)
python -m src.run_all_experiments --dry-run

# Run with tag
python -m src.run_all_experiments --tag paper_results

# Limit to first 2 experiments
python -m src.run_all_experiments --limit 2

# Skip failed experiments
python -m src.run_all_experiments --skip_failed
```

---

#### `src/transpile_utils.py`
**Purpose:** Transpiles circuits and counts two-qubit gates (compression metric).

**What it does:**
- Transpiles circuits to match hardware constraints (coupling map, basis gates)
- Counts entangling two-qubit gates (CNOT, CZ, RXX, RYY, RZZ, etc.)
- Provides detailed gate statistics

**Key Functions:**
- `transpile_and_count_2q(circuit, backend, ...)` → Main function
  - Returns: `(transpiled_circuit, two_qubit_gate_count)`
- `count_entangling_gates(circuit)` → Count only entangling gates
- `count_all_2q_gates(circuit)` → Count all two-qubit gates
- `summarize_transpile(circuit, backend, verbose)` → Detailed statistics

**Why This Matters:**
- Two-qubit gate count is the **compression metric**
- Post-transpile count reflects hardware reality (not just circuit structure)
- Lower gate count = better compression = lower hardware cost

**Integration:**
- Called by `learner.py` in `forward_loss()` to compute gate count
- Used by training scripts for loss computation
- Cached by mask hash in `learner.py` for efficiency

**Pipeline Position:** Gate counting (during loss computation)

---

#### `src/plots.py`
**Purpose:** Visualization utilities for training curves, masks, and comparisons.

**Key Functions:**
- `plot_all_curves_from_history(history, prefix, output_dir)` → Training curves
- `plot_mask_heatmap(mask, title, fname, output_dir)` → Mask visualization
- `plot_pareto_from_runs(runs, output_dir, fname)` → Pareto front plots
- `compare_methods_bar(method_results, fname, output_dir)` → Bar chart comparison

**Integration:**
- Called by training scripts to save plots
- Called by `run_all_experiments.py` for aggregation plots
- Called by `evaluate_model.py` for comparison plots

---

#### `src/logging_utils.py`
**Purpose:** Utilities for saving training logs, parameters, and metrics.

**Key Functions:**
- `save_training_history(run_dir, df_history)` → Save CSV log
- `save_parameters(run_dir, theta, mask)` → Save NPZ files
- `save_final_metrics(run_dir, metrics)` → Save JSON metrics
- `timestamp_str()` → Generate timestamp strings

**Integration:**
- Called by training scripts to save outputs
- Used throughout pipeline for file I/O

---

### Experiment Configuration Files (`experiments/`)

#### YAML Structure

Each YAML file defines a single experiment:

```yaml
experiment_name: "baseline_pothos_large"
mode: "baseline"  # or "compressed"

n_qubits: 2
depth: 3
n_iterations: 150
lr: 0.05
lam: 0.01
dataset_name: "pothos_chater_large"
channel_strength: 0.6
seed: 42

# Compressed-specific (only if mode == "compressed")
prune_every: 120
tolerance: 0.01
```

**Available Configs:**
- `baseline_pothos_large.yaml` → Baseline with large dataset
- `compressed_pothos_large.yaml` → Compressed with large dataset
- Various variants (different hyperparameters)

**Integration:**
- Loaded by `run_experiment.py` for single runs
- Discovered by `run_all_experiments.py` for batch runs

---

## Experimental Pipeline

### 1. Data Generation

**Module:** `src/data.py`

**How it works:**
1. Defines two category prototypes in 2D space:
   - Category A: centered at `[0.23, 0.27]`
   - Category B: centered at `[0.77, 0.73]`
2. Samples points around each prototype using Gaussian noise:
   - Small: 4 points per category (std=0.07)
   - Medium: 10 points per category (std=0.10)
   - Large: 20 points per category (std=0.11) ← **Recommended for paper**
3. Features are clipped to [0, 1] for quantum channel encoding

**Recommended Dataset:**
- **Large (40 points)** is recommended for the paper
- Provides sufficient data for meaningful learning while remaining computationally tractable

**Usage:**
```python
from src.data import get_toy_dataset
X, y = get_toy_dataset("pothos_chater_large")
# X: shape (40, 2), y: shape (40,)
```

---

### 2. Model Construction

**Module:** `src/ansatz.py`

**How it works:**
1. **Single-qubit rotations:** Each layer applies RX and RZ to all qubits
2. **Entangling blocks:** HEA-style (CX, RY, CX) or Heisenberg (RXX, RYY, RZZ)
3. **Masking:** Binary mask controls which entanglers are active
   - `mask[d, k] = 1` → entangler on edge `k` at depth `d` is active
   - `mask[d, k] = 0` → entangler is pruned (compressed mode)

**Circuit Structure:**
```
Layer 0: [RX, RZ] on all qubits → Entanglers (masked) → [RX, RZ] on all qubits
Layer 1: [RX, RZ] on all qubits → Entanglers (masked) → [RX, RZ] on all qubits
Layer 2: [RX, RZ] on all qubits → Entanglers (masked) → [RX, RZ] on all qubits
```

**Compression Mechanism:**
- Mask starts as all 1s (all gates active)
- During training, mask is pruned (gates set to 0)
- Pruned gates are not included in circuit → lower gate count

**Usage:**
```python
from src.ansatz import build_ansatz, init_random_theta, init_full_mask, get_default_pairs

n_qubits = 2
depth = 3
pairs = get_default_pairs(n_qubits)  # [(0, 1)]
theta = init_random_theta(n_qubits, depth, len(pairs))
mask = init_full_mask(depth, len(pairs))

qc = build_ansatz(n_qubits, depth, theta, mask, pairs)
```

---

### 3. Evidence Channels

**Module:** `src/channels.py`

**How it works:**
1. Takes stimulus features `x` (1D array, e.g., `[0.3, 0.7]`)
2. Computes channel strength from features:
   - Default method: `strength = strength_max * (0.5 + 0.5 * (mean(x) - 0.5))`
   - Maps feature mean to strength in range `[0.25 * max, 0.75 * max]`
3. Builds Kraus operators based on strength:
   - **Amplitude damping:** Energy dissipation (default in older code)
   - **Projective update:** Bayesian-like soft measurement (default in current code)
4. Returns Qiskit `Kraus` channel

**Channel Strength Effects:**
- Higher strength → stronger evidence update → faster learning
- Lower strength → weaker update → slower learning
- Default: `strength_max = 0.4` (configurable via YAML)

**Usage:**
```python
from src.channels import build_evidence_channel
from qiskit.quantum_info import DensityMatrix

x = np.array([0.3, 0.7])
channel = build_evidence_channel(x, kind="projective", strength=0.4)
rho = DensityMatrix([[0.5, 0.0], [0.0, 0.5]])
rho_new = rho.evolve(channel)
```

---

### 4. Learning

**Module:** `src/learner.py`

**How it works:**
1. **Forward Pass (per sample):**
   ```
   rho_0 = init_belief(n_qubits)           # Initial state (biased toward |0...0⟩)
   rho_1 = apply_feature_encoding(rho_0, x) # RY, RZ rotations from features
   rho_2 = apply_unitary(rho_1, ansatz)     # Variational circuit
   rho_3 = apply_evidence_channel(rho_2, x) # Evidence update
   prob = predict_label(rho_3)              # Z expectation + sigmoid
   ```
2. **Loss Computation:**
   ```
   CE_loss = mean(-log(p_i) if y_i==1 else -log(1-p_i))
   gate_count = transpile_and_count_2q(ansatz)[1]
   total_loss = CE_loss + λ * gate_count
   ```
3. **Optimization:**
   - Finite-difference gradient: `grad = (loss(theta+h) - loss(theta)) / h`
   - Parameter update: `theta = theta - lr * grad`
   - For compressed mode: periodically prune mask

**Pruning (Compressed Mode):**
- Every `prune_every` iterations:
  1. For each active entangler:
     - Temporarily disable it
     - Compute loss increase
     - If increase ≤ `tolerance`: permanently disable
  2. Continue until no more gates can be pruned

**Prediction:**
- Uses multi-qubit Z and X expectations:
  ```
  logit = α * (0.6*<Z_0> + 0.4*<Z_1> + 0.3*<X_0> + 0.2*<X_1>)
  prob = sigmoid(logit)
  ```
- `readout_alpha` controls nonlinearity (default: 4.0)

**Usage:**
```python
from src.learner import forward_loss

result = forward_loss(
    theta=theta,
    mask=mask,
    X=X_train,
    y=y_train,
    lam=0.1,
    n_qubits=2,
    depth=3,
    channel_strength=0.4
)
# result["total_loss"], result["ce_loss"], result["two_q_cost"], result["preds"]
```

---

### 5. Training Scripts

#### Baseline Training (`src/train_baseline.py`)

**Workflow:**
1. Load dataset
2. Initialize `theta` (random) and `mask` (all 1s)
3. For each iteration:
   - Compute loss via `forward_loss()`
   - Compute gradient via finite differences
   - Update `theta` via gradient descent
   - Log metrics (loss, accuracy, gate count)
4. Save outputs (history, parameters, plots)

**Key Parameters:**
- `n_iterations`: Number of training steps (default: 100)
- `lr`: Learning rate (default: 0.01)
- `lam`: Regularization strength (default: 0.1)
- `optimizer_type`: "finite_diff" or "adam"

**Outputs:**
- `results/runs/baseline_<timestamp>/training_history.csv`
- `results/runs/baseline_<timestamp>/params_final.npz`
- `results/runs/baseline_<timestamp>/figures/`

---

#### Compressed Training (`src/train_compressed.py`)

**Workflow:**
1. Load dataset
2. Initialize `theta` (random) and `mask` (all 1s)
3. For each iteration:
   - Compute loss via `forward_loss()`
   - Compute gradient via finite differences
   - Update `theta` via gradient descent
   - **Every `prune_every` iterations:** Run `greedy_prune()`
   - Log metrics (loss, accuracy, gate count, mask sparsity)
4. Save outputs (history, parameters, mask history, plots)

**Key Parameters:**
- `prune_every`: Pruning frequency (default: 20)
- `tolerance`: Maximum loss increase allowed when pruning (default: 0.01)

**Outputs:**
- `results/runs/compressed_<timestamp>/training_history.csv` (includes `mask_sparsity`)
- `results/runs/compressed_<timestamp>/params_final.npz` (includes `mask`)
- `results/runs/compressed_<timestamp>/mask_history.npz`
- `results/runs/compressed_<timestamp>/figures/` (includes mask heatmaps)

---

### 6. Evaluation

**Module:** `src/evaluate_model.py`

**What it does:**
1. Loads training logs from `results/logs/`
2. Loads model parameters from `results/`
3. Recomputes accuracy by re-running predictions
4. Generates comparison plots:
   - Loss curves (baseline vs compressed)
   - Mask sparsity over time
   - Mask heatmaps
   - Pareto front (CE loss vs gate count)

**Usage:**
```bash
python -m src.evaluate_model
```

**Outputs:**
- `results/figures/eval_loss_comparison.png`
- `results/figures/eval_compressed_mask_heatmap.png`
- `results/figures/eval_pareto_ce_loss_vs_cost.png`

---

### 7. Transpilation

**Module:** `src/transpile_utils.py`

**How it works:**
1. Takes circuit from `build_ansatz()`
2. Transpiles to match hardware constraints (if backend provided):
   - Coupling map (which qubit pairs can have gates)
   - Basis gates (available gate set)
3. Counts entangling two-qubit gates in transpiled circuit
4. Returns count as compression metric

**Why Post-Transpile Count:**
- Pre-transpile count doesn't reflect hardware reality
- Hardware may require SWAP gates to route qubits
- Post-transpile count = actual hardware cost

**Usage:**
```python
from src.transpile_utils import transpile_and_count_2q

transpiled, count = transpile_and_count_2q(circuit, backend=None)
# count: number of two-qubit gates
```

---

## How to Run Everything

### Environment Setup

```bash
# Clone repository (if applicable)
cd qb-learner-compression

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `qiskit==1.1.0` - Quantum circuit framework
- `qiskit-aer==0.13.2` - Quantum simulators
- `numpy`, `scipy` - Numerical computing
- `matplotlib`, `pandas` - Visualization and data
- `tqdm` - Progress bars
- `pyyaml` - Configuration files

---

### Running One Experiment

**From YAML config:**
```bash
python -m src.run_experiment --config experiments/baseline_pothos_large.yaml
```

**Direct training (baseline):**
```bash
python -m src.train_baseline \
    --n_qubits 2 \
    --depth 3 \
    --iterations 100 \
    --lr 0.01 \
    --lam 0.1 \
    --dataset pothos_chater_large
```

**Direct training (compressed):**
```bash
python -m src.train_compressed \
    --n_qubits 2 \
    --depth 3 \
    --iterations 100 \
    --lr 0.01 \
    --lam 0.1 \
    --prune_every 20 \
    --tolerance 0.01 \
    --dataset pothos_chater_large
```

---

### Running All Experiments

**Basic usage:**
```bash
python -m src.run_all_experiments
```

**With options:**
```bash
# Dry run (list configs without running)
python -m src.run_all_experiments --dry-run

# Add tag to output directory
python -m src.run_all_experiments --tag final_large

# Limit to first 2 experiments
python -m src.run_all_experiments --limit 2

# Skip failed experiments and continue
python -m src.run_all_experiments --skip_failed
```

**Output:**
- `results/experiments_<tag>_<timestamp>/`
  - Individual experiment outputs in `experiments/`
  - Aggregated summary in `summary/`

---

### Evaluating a Trained Model

```bash
python -m src.evaluate_model
```

**Prerequisites:**
- Baseline model: `results/baseline_final_theta.npy` or `results/logs/baseline_log.csv`
- Compressed model: `results/compressed_final.npz` or `results/logs/compressed_log.csv`

**Output:**
- Comparison plots in `results/figures/eval_*.png`

---

## Output Structure

### Individual Run Directory

**Location:** `results/runs/<run_name>/`

**Contents:**
- `config.json` → Exact configuration used (saved automatically)
- `training_history.csv` → Per-iteration metrics:
  - `iteration`: Training step
  - `loss`: Total loss (CE + λ × gate count)
  - `ce_loss`: Cross-entropy loss only
  - `accuracy`: Classification accuracy
  - `two_qubit_count`: Post-transpile gate count
  - `mask_sparsity`: (compressed only) Fraction of active entanglers
- `params_final.npz` → Final parameters:
  - `theta`: Final parameter tensor
  - `mask`: (compressed only) Final mask
- `loss_history.npz` → Arrays for plotting:
  - `loss`: Total loss array
  - `ce_loss`: CE loss array
  - `gate_cost`: Gate count array
- `mask_history.npz` → (compressed only) Mask evolution:
  - Shape: `(n_iterations, depth, n_edges)`
- `final_metrics.json` → Final metrics:
  - `final_loss`, `final_ce_loss`, `final_accuracy`
  - `final_two_qubit_count`
  - `final_mask_sparsity` (compressed only)
- `figures/` → Generated plots:
  - `*_loss.png` → Loss curves
  - `*_accuracy.png` → Accuracy curves
  - `*_gate_cost.png` → Gate count curves
  - `*_mask_heatmap.png` → (compressed only) Mask visualization
  - `*_pareto.png` → Pareto front plots

---

### Batch Experiment Summary

**Location:** `results/experiments_<tag>_<timestamp>/summary/`

**Contents:**
- `all_experiments_summary.csv` → Table of all experiments:
  - Columns: `experiment_name`, `mode`, `final_loss`, `final_accuracy`, `two_qubit_count`, `mask_sparsity`, etc.
- `accuracy_vs_gatecount.png` → Scatter plot (accuracy vs gate count)
- `baseline_vs_compressed.png` → Bar chart comparison
- `experiment_methods_comparison.png` → Method comparison bar chart

---

### Legacy Output Locations

For backward compatibility, some outputs are also saved to:
- `results/logs/baseline_log.csv` → Baseline training log
- `results/logs/compressed_log.csv` → Compressed training log
- `results/baseline_final_theta.npy` → Baseline final parameters
- `results/compressed_final.npz` → Compressed final parameters
- `results/figures/` → Legacy figure location

---

## Notebooks

### `notebooks/sanity_checks.ipynb`

**Purpose:** Comprehensive verification of all system components.

**What it tests:**
1. **Environment:** Imports and version checks
2. **Dataset:** Loading and visualization
3. **Evidence Channels:** Kraus operator construction, CPTP properties
4. **Density Matrix Updates:** Channel application, trace preservation
5. **Ansatz Construction:** Circuit building, parameter counting
6. **Forward Pass:** End-to-end loss computation
7. **Transpilation:** Gate counting accuracy

**How to run:**
```bash
# Start Jupyter
jupyter notebook notebooks/sanity_checks.ipynb
```

**Why it matters:**
- Ensures all components work correctly before running experiments
- Useful for debugging when things go wrong
- Verifies mathematical properties (CPTP, unitarity, etc.)

---

### `notebooks/visualize_channels.ipynb`

**Purpose:** Visualize quantum evidence channels and their effects.

**What it shows:**
1. **Kraus Operators:** Structure and properties
2. **Channel Strength:** How features map to strength parameters
3. **State Evolution:** How channels update density matrices
4. **Damping Effects:** Amplitude and phase damping visualization

**How to run:**
```bash
jupyter notebook notebooks/visualize_channels.ipynb
```

**Why it matters:**
- Understands how evidence channels encode stimulus information
- Visualizes the belief update mechanism
- Useful for paper figures showing channel effects

---

### `notebooks/visualize_ansatz.ipynb`

**Purpose:** Visualize variational circuit structure and mask effects.

**What it shows:**
1. **Circuit Structure:** Gate decomposition, depth, connectivity
2. **Mask Visualization:** Which entanglers are active/inactive
3. **Parameter Sensitivity:** How parameters affect circuit output
4. **Transpilation Effects:** How hardware constraints change circuits

**How to run:**
```bash
jupyter notebook notebooks/visualize_ansatz.ipynb
```

**Why it matters:**
- Understands circuit architecture
- Visualizes compression (mask sparsity)
- Useful for paper figures showing circuit structure

---

## Expected Results

### Baseline Model

**Expected Accuracy:** ≈ 0.5

**Why:**
- Toy dataset is symmetric (two balanced clusters)
- Linear decision boundary on qubit-0 leads to 50/50 classification
- Nonlinear readout (multi-qubit Z/X expectations) helps but may not achieve perfect separation

**Expected Gate Count:** 6–9 two-qubit gates

**Why:**
- Depth 3, 1 edge (2 qubits) → 3 entanglers initially
- Transpilation may add SWAP gates → 6–9 total

**Expected Loss:**
- CE loss: ~1.5–2.0 (depends on dataset and hyperparameters)
- Total loss: CE loss + λ × gate count

---

### Compressed Model

**Expected Accuracy:** 0.0–0.5 (may decrease from baseline)

**Why:**
- Pruning removes entanglers → circuit becomes less expressive
- May reduce to pure single-qubit classifier → lower accuracy
- **This is expected:** goal is compression, not accuracy improvement

**Expected Gate Count:** 0–3 two-qubit gates

**Why:**
- Pruning removes unnecessary entanglers
- Ideally reduces to 0–3 gates (from 6–9)
- Compression ratio: up to 100% (if all gates pruned)

**Expected Mask Sparsity:** 0.0–0.5 (50–100% of gates pruned)

**Why:**
- Greedy pruning removes gates that don't significantly impact loss
- Tolerance controls how aggressive pruning is

---

### Paper Goals

**This project demonstrates compression, not accuracy improvement.**

- **Goal 1:** Show that circuits can be compressed (gate count reduction)
- **Goal 2:** Show trade-off between accuracy and gate count
- **Goal 3:** Demonstrate hardware-aware optimization (post-transpile counts)

**Expected Findings:**
- Compressed models achieve lower gate counts
- Accuracy may decrease (acceptable trade-off)
- Compression ratio: 50–100% (depending on tolerance)

---

## Reproduction Steps for Paper Figures

### Generate All Results

**Run batch experiments:**
```bash
python -m src.run_all_experiments --tag paper_results
```

**This will:**
1. Run all experiments in `experiments/`
2. Save results to `results/experiments_paper_results_<timestamp>/`
3. Generate summary CSV and comparison plots

**Output location:**
- `results/experiments_paper_results_<timestamp>/summary/all_experiments_summary.csv`
- `results/experiments_paper_results_<timestamp>/summary/accuracy_vs_gatecount.png`
- `results/experiments_paper_results_<timestamp>/summary/baseline_vs_compressed.png`

---

### Generate Individual Run Plots

**For a specific run:**
```bash
# Run experiment
python -m src.run_experiment --config experiments/baseline_pothos_large.yaml

# Plots are automatically generated in:
# results/runs/baseline_<timestamp>/figures/
```

**Available plots:**
- `baseline_loss.png` → Total loss curve
- `baseline_ce_loss.png` → CE loss curve
- `baseline_accuracy.png` → Accuracy curve
- `baseline_gate_cost.png` → Gate count curve
- `baseline_pareto.png` → Pareto front (if applicable)

---

### Regenerate Summary Plots

**After running experiments:**
```bash
# Evaluate models (generates comparison plots)
python -m src.evaluate_model

# Or regenerate summary from existing results
python -m src.run_all_experiments --tag regenerate_summary
```

---

### Paper Figure Checklist

1. **Accuracy vs Gate Count Scatter:**
   - Source: `results/experiments_<tag>_<timestamp>/summary/accuracy_vs_gatecount.png`
   - Shows trade-off between accuracy and compression

2. **Baseline vs Compressed Comparison:**
   - Source: `results/experiments_<tag>_<timestamp>/summary/baseline_vs_compressed.png`
   - Bar chart comparing average metrics

3. **Training Curves:**
   - Source: `results/runs/<run_name>/figures/*_loss.png`, `*_accuracy.png`
   - Shows training dynamics

4. **Mask Visualization:**
   - Source: `results/runs/compressed_<timestamp>/figures/compressed_mask_heatmap.png`
   - Shows which gates were pruned

5. **Pareto Front:**
   - Source: `results/runs/<run_name>/figures/*_pareto.png`
   - Shows accuracy vs gate count trade-off

---

## Additional Notes

### Hyperparameter Tuning

**Key hyperparameters:**
- `lam` (λ): Controls accuracy vs. complexity trade-off
  - Higher λ → more compression, lower accuracy
  - Lower λ → less compression, higher accuracy
- `tolerance`: Pruning aggressiveness (compressed only)
  - Higher tolerance → more gates pruned
  - Lower tolerance → fewer gates pruned
- `channel_strength`: Evidence channel strength
  - Higher → stronger evidence updates
  - Lower → weaker evidence updates
- `readout_alpha`: Prediction nonlinearity
  - Higher → sharper predictions
  - Lower → softer predictions

### Troubleshooting

**Low accuracy (< 0.3):**
- Check if predictions are inverted (try flipping sign in `predict_label`)
- Increase `readout_alpha` for stronger nonlinearity
- Increase `channel_strength` for stronger evidence updates

**No compression (mask stays full):**
- Decrease `tolerance` (allows more aggressive pruning)
- Increase `lam` (stronger penalty on gate count)
- Check if `prune_every` is too large (pruning happens too infrequently)

**Training instability:**
- Decrease learning rate `lr`
- Use Adam optimizer instead of finite differences
- Check gradient norms (should not be too small or too large)

---

## Citation

If you use this code in your research, please cite:

```
Pothos, E. M., & Chater, N. (2002). A simplicity principle in unsupervised human categorization. 
Cognitive Psychology, 45, 45–85.
```

---

## Contact

Mohammad Zoraiz 
mz248@duke.edu