#!/usr/bin/env bash
# Reproduce every figure and table in the paper from scratch.
#
# Maps each paper artifact to the exact command that produces it. Run from the repo
# root inside the project venv (see requirements.txt; Python 3.14, qiskit 2.4.2,
# qiskit-aer 0.17.2, qiskit-ibm-runtime 0.47.0, jax 0.10.2, scikit-learn 1.9.0).
#
# Usage:
#   bash scripts/reproduce_paper.sh           # simulation results (no IBM account needed)
#   bash scripts/reproduce_paper.sh --hardware # also re-run the ibm_fez device experiments
#
# The simulation pipeline is fully deterministic (fixed seeds) and self-contained.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/6] Sanity: learner matches Qiskit statevector =="
python scripts/verify_qcore.py

echo "== [2/6] Pure-state sweeps (frontier, lambda, mask ablation) for all difficulties =="
echo "   -> results_v2/{easy,medium,hard}/{frontier,lambda_sweep,mask_ablation}.csv"
echo "   -> feeds Fig.1 (frontier_family), Tab.I (lambda), Fig.2 + Tab.II (mask_ablation)"
python -m src.run_v2 --all --seeds 0 1 2 3 4

echo "== [3/6] Density-matrix (mixed-state) sweeps =="
echo "   -> results_v2_density/...  feeds Fig.5, Fig.6, Tab.III"
python -m src.run_density --all --seeds 0 1 2

echo "== [4/6] Noise-model robustness (FakeManila, 4096 shots) =="
echo "   -> results_v2/hardware/noise_robustness.csv  feeds Fig.3"
python scripts/noise_robustness.py

echo "== [5/6] Classical baselines + paired statistics (appendix) =="
python scripts/classical_baselines.py
python scripts/paired_stats.py hard

echo "== [6/6] Render all figures =="
python scripts/make_figures_v2.py
python scripts/make_figures_density.py
python scripts/plot_hardware.py

if [[ "${1:-}" == "--hardware" ]]; then
  echo "== [hardware] ibm_fez device runs (requires IBM credentials) =="
  echo "   -> results_v2/hardware/ibm_fez_frontier.json (Fig.4) + ibm_fez_results.json"
  python scripts/hardware_frontier.py
  python scripts/run_on_hardware.py --difficulty hard --mode ibm --n-eval 40 --shots 4096
fi

echo "== done. Figures in paper/, results in results_v2/ and results_v2_density/ =="
