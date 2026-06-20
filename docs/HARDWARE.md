# Running on real (and realistically-noisy) quantum hardware

This project **trains in exact simulation** (fast, analytic gradients) and then
**deploys the trained model for inference on hardware**. Because only inference runs on
the device, validating the results costs a handful of cheap circuit executions instead
of a full hardware-in-the-loop training run.

The bridge from simulation to circuit is exact: `src/hardware_backend.py` builds a Qiskit
circuit that is **bit-identical** to the `qcore` model (verified to `< 1e-9` in
`tests/test_hardware_backend.py`), so a model trained in simulation transfers unchanged.

There are three backends, selected with `--mode`:

| mode    | what it is                                   | needs an account? | cost |
|---------|----------------------------------------------|-------------------|------|
| `exact` | `StatevectorEstimator` (noiseless, exact)    | no                | free |
| `noisy` | `AerSimulator.from_backend(FakeManila)` — the **real FakeManila noise model**, finite shots | no | free |
| `ibm`   | a real IBM Quantum device via `QiskitRuntimeService` | **yes** | metered QPU time |

---

## 1. Start free: the FakeManila noise model (recommended first step)

No account needed. This runs the trained circuits under IBM FakeManila's real noise model:

```bash
. .venv/bin/activate
# full vs greedy-compressed circuit, evaluated exactly and under device noise
python scripts/run_on_hardware.py --difficulty hard --mode noisy --shots 4096
# how the whole frontier holds up under noise (5 seeds -> figure + CSV)
python scripts/noise_robustness.py
```

`scripts/noise_robustness.py` writes `results_v2/figures/noise_robustness.png` and
`results_v2/hardware/noise_robustness.csv`. The takeaway (5 seeds): accuracy under the
noise model **tracks the noiseless frontier within a few percent at every two-qubit
budget**, so the compression savings are real net of device noise.

---

## 2. Connect to a real IBM Quantum device

IBM quantum hardware is accessed through the **IBM Quantum Platform** (on IBM Cloud).
The legacy `ibm_quantum` channel was retired; this project uses the current
`ibm_quantum_platform` channel via `qiskit-ibm-runtime` (installed, v0.47).

### 2.1 Get credentials
1. Create / sign in to an account at <https://quantum.cloud.ibm.com>.
2. Create an **instance** (the free **Open plan** is enough) and copy:
   - your **API token**, and
   - your **instance CRN** (looks like `crn:v1:bluemix:public:quantum-computing:...`).

### 2.2 Save the account once (writes `~/.qiskit/qiskit-ibm.json`)
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="<YOUR_API_TOKEN>",
    instance="<YOUR_INSTANCE_CRN>",   # optional if your account has a default
    set_as_default=True,
    overwrite=True,
)
```
Run that once in `python` (tip: type `! python` in this session to run it inline). After
that, `QiskitRuntimeService()` — which `--mode ibm` calls — picks the account up
automatically. Alternatively, set the token in the environment without saving a file:
```bash
export QISKIT_IBM_TOKEN="<YOUR_API_TOKEN>"
export QISKIT_IBM_INSTANCE="<YOUR_INSTANCE_CRN>"
```

### 2.3 Verify the connection
One command lists your operational real devices (with queue depths) and the least-busy one:
```bash
python scripts/check_ibm_connection.py
```
or inline:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
svc = QiskitRuntimeService()
print([b.name for b in svc.backends(operational=True, simulator=False)])
print("least busy:", svc.least_busy(operational=True, simulator=False).name)
```

### 2.4 Run inference on hardware
```bash
# evaluate on the least-busy real device; cap QPU cost to 20 test points
python scripts/run_on_hardware.py --difficulty hard --mode ibm --n-eval 20 --shots 4096

# or target a specific backend
python scripts/run_on_hardware.py --mode ibm --ibm-backend ibm_brisbane --n-eval 20
```

The script trains in simulation, then prints exact accuracy alongside on-device accuracy
for both the full and a greedy-compressed circuit, so you can see the two-qubit cost drop
(`N2q`) while accuracy is retained.

**Train once, deploy later** (so you don't retrain for every device run):
```bash
python scripts/run_on_hardware.py --difficulty hard --mode noisy \
    --save results_v2/hardware/model_hard.npz
python scripts/run_on_hardware.py --load results_v2/hardware/model_hard.npz --mode ibm --n-eval 20
```

---

## 3. Cost, queueing, and good practice

- **QPU time is metered.** The Open plan gives a limited monthly allowance; each evaluated
  stimulus is one circuit with four observables, so keep `--n-eval` small (10–30) when on
  real hardware. Use `--mode noisy` for unlimited iteration.
- **Jobs queue.** `least_busy` picks the shortest queue, but a run can still wait minutes.
- **The compressed circuit is the point.** It has fewer two-qubit gates after transpilation
  (lower `N2q`), so it is cheaper and accumulates less error on a real device — exactly the
  trade-off the paper studies.

---

## 4. Troubleshooting

- **`could not initialize 'ibm' backend` / auth errors** — re-run the `save_account` snippet
  with a valid token *and* instance CRN; confirm with the `svc.backends()` check in §2.3.
- **Account file location** — credentials live in `~/.qiskit/qiskit-ibm.json`. Delete it to
  reset, then `save_account` again.
- **API churn** — IBM has changed channel names over time. This project targets
  `qiskit-ibm-runtime==0.47` with `channel="ibm_quantum_platform"`. If you upgrade the
  package and the channel name changes, update the single call in
  `src/hardware_backend.py::make_estimator`.
- **Observable layout after transpilation** — handled automatically:
  `estimate_expectations` transpiles each circuit to the target and remaps observables with
  `SparsePauliOp.apply_layout`, so SWAP insertion does not corrupt the readout.
