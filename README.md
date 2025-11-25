# Quantum Bayesian Learner with Hardware-Aware Circuit Compression

A hybrid quantum–classical project exploring how a quantum Bayesian learner’s performance depends on variational circuit complexity under realistic NISQ hardware constraints.

## Repository Structure

This project is organized into modular components so that the belief-update logic, variational circuit construction, transpilation, optimization, and experiments can be developed and debugged independently. 

The structure is:

```
qb-learner-compression/
│
├── src/
│   ├── data.py
│   ├── channels.py
│   ├── ansatz.py
│   ├── learner.py
│   ├── transpile_utils.py
│   ├── train.py
│   ├── plots.py
│   └── utils.py
│
├── experiments/
│   ├── config_example.yaml
│   └── run_experiment.py
│
├── notebooks/
│   ├── sanity_checks.ipynb
│   └── pareto_visualization.ipynb
│
├── results/
│   ├── logs/
│   └── figures/
│
├── README.md
├── requirements.txt
└── LICENSE
```
---

## File-by-File Overview

### **src/data.py**

Defines the toy dataset used for the quantum Bayesian learning task.

* Generates small classification datasets (e.g., 2D points).
* Normalizes/encodes stimuli into the form needed by the evidence channels.

### **src/channels.py**

Implements quantum channels (Kraus maps) that act as Bayesian “evidence updates.”

* Defines Kraus operators (E_{x,k}) as functions of the stimulus features.
* Provides utilities to apply channels to density matrices using Qiskit’s `Kraus` class.

### **src/ansatz.py**

Builds the parameterized variational circuit used for belief updating.

* Implements Heisenberg-style entangling blocks (`XX`, `YY`, `ZZ`).
* Supports binary **mask tensors** to selectively remove entanglers for compression.
* Constructs circuits respecting an explicit hardware coupling map.

### **src/learner.py**

Defines how predictions and losses are computed.

* Runs a full belief-update step using the ansatz + channel.
* Computes binary cross-entropy classification loss.
* Wraps the forward pass: `(θ, mask) → loss, prediction`.

### **src/transpile_utils.py**

Handles device-aware compilation.

* Transpiles circuits to simulated IBM backends.
* Measures **post-transpile two-qubit gate count**, which is the compression metric.
* Allows switching between FakeManila, FakePerth, etc.

### **src/train.py**

Core optimization logic.

* Optimization of circuit parameters (θ).
* Periodic pruning of mask (m) based on loss tolerance.
* Stores training logs (loss, CNOT count, accuracy).
* Exposes functions to train:

  * Uncompressed baseline
  * Compiler-only baseline
  * Hardware-aware compressed model

### **src/plots.py**

Visualizations for results.

* Accuracy vs. CNOT count (Pareto curve).
* Training loss curves.
* Mask structure heatmaps.
* Saves files to `results/figures/`.

### **src/utils.py**

Miscellaneous helpers.

* Seeding
* Parameter initialization
* File I/O utilities

---

## Experiment Files

### **experiments/config_example.yaml**

Template specifying hyperparameters for a training run:

* Circuit depth
* Learning rate
* Mask pruning frequency
* λ (two-qubit penalty)
* Backend name
* Dataset choice

### **experiments/run_experiment.py**

Runs an experiment using a config file.

* Loads config
* Calls training routine
* Saves results to `results/logs/`
* Optionally generates plots

---

## Jupyter Notebooks

### **notebooks/sanity_checks.ipynb**

Used early for debugging:

* Verify Kraus maps behave correctly
* Verify ansatz produces unitary
* Test forward pass on a single datapoint

### **notebooks/pareto_visualization.ipynb**

* Loads experiment logs
* Produces Pareto curves
* Compares compressed vs. uncompressed vs. compiler-only

---

## Results Directory

Holds all saved output.

### **results/logs/**

Training logs (`.csv` or `.json`) from each experiment:

* accuracy
* loss
* CNOT count
* mask configuration snapshots

### **results/figures/**

Auto-generated figures for the paper:

* circuit gate count plot
* classification boundaries
* Pareto curves

---

## requirements.txt

Specifies Python dependencies:

```
qiskit
qiskit-aer
numpy
scipy
matplotlib
pandas
pyyaml
tqdm
```
---
