"""
plot_hardware.py

Render the real-device result as a grouped bar chart from a saved JSON record, so the
figure is reproducible from logged numbers (no hand-drawing). Reads
results_v2/hardware/ibm_fez_results.json -- a list of records:
  {"model": "full", "n2q": 24, "acc_exact": 0.x, "acc_ibm": 0.y,
   "backend": "ibm_fez", "n_eval": N, "shots": S}
and writes results_v2/figures/hardware_real.png (+ a copy into paper/).
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results_v2/hardware/ibm_fez_results.json"
FIG = ROOT / "results_v2/figures/hardware_real.png"


def main():
    recs = json.loads(JSON.read_text())
    recs = sorted(recs, key=lambda r: -r["n2q"])  # full (high N2q) first
    labels = [f'{r["model"]}\n($N_{{2q}}={r["n2q"]}$)' for r in recs]
    exact = [r["acc_exact"] for r in recs]
    dev = [r["acc_ibm"] for r in recs]
    backend = recs[0].get("backend", "ibm")
    n_eval = recs[0].get("n_eval", "?")
    shots = recs[0].get("shots", "?")

    x = np.arange(len(recs)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x - w / 2, exact, w, label="exact (noiseless)", color="#2c7fb8")
    ax.bar(x + w / 2, dev, w, label=f"{backend} (real device)", color="#d95f0e")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.text(len(recs) - 0.5, 0.505, "chance", fontsize=7, color="gray", va="bottom", ha="right")
    for xi, v in zip(x - w / 2, exact):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, dev):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0.3, 1.0)
    ax.set_title(f"Real-device accuracy ({backend}, {n_eval} points, {shots} shots)", fontsize=9)
    ax.legend(loc="lower center", framealpha=0.95); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200); plt.close(fig)
    shutil.copy(FIG, ROOT / "paper/fig_hardware_real.png")
    print(f"wrote {FIG} and paper/fig_hardware_real.png")


if __name__ == "__main__":
    main()
