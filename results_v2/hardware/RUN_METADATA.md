# Physical-device run provenance (`ibm_fez`)

All real-hardware numbers in the paper come from the two runs below. Raw logs are in
this directory (`ibm_fez_run.txt`, `ibm_fez_run1_16pts.txt`) and the machine-readable
results in `ibm_fez_results.json` (endpoints) and `ibm_fez_frontier.json` (frontier).

| Field | Frontier run | Endpoint run |
|---|---|---|
| Backend | `ibm_fez` (156-qubit IBM Heron) | `ibm_fez` (156-qubit IBM Heron) |
| Date | 2026-06-20 | 2026-06-20 |
| Plan / instance | open | open |
| Eval points | 16 | 40 |
| Shots | 2048 | 4096 |
| Budgets (N₂q) | 2, 6, 12, 18, 24 | 24 (full), 6 (compressed) |
| Difficulty | hard (freq 4) | hard (freq 4) |
| Seed | 0 | 0 |
| Transpiler optimization level | 1 | 1 |
| `seed_transpiler` | 42 | 42 |
| Primitive | `EstimatorV2` (default options) | `EstimatorV2` (default options) |
| Error mitigation | primitive defaults only (no explicit resilience configured) | same |
| Job IDs | not retained; see raw logs | not retained; see raw logs |

Notes:
- The trained model deployed on hardware is bit-identical to the simulator model
  (`src/hardware_backend.py` reproduces `src/qcore.py` to < 1e-9); `model_hard.npz`
  holds the deployed weights.
- The two-qubit count `N₂q` is the real post-transpile count on `FakeManila`
  (`src/hardware_cost.py`), used as the cost axis; the hardware accuracies are measured
  on `ibm_fez`.
- Reproduce with `scripts/hardware_frontier.py` (frontier) and
  `scripts/run_on_hardware.py --mode ibm` (endpoints); both require IBM credentials in
  `~/.qiskit/qiskit-ibm.json` (never committed).
