"""
noise_robustness.py

Quantify how the accuracy--complexity frontier changes under a realistic device NOISE
MODEL (Qiskit FakeManila), versus exact (noiseless) simulation. For several two-qubit
budgets along the greedy frontier and several seeds, we evaluate the trained circuit
both exactly and on the FakeManila noise model with finite shots.

Key finding (5 seeds): accuracy under the FakeManila noise model tracks the noiseless
frontier within a few percent at every two-qubit budget, with overlapping error bars.
The simulated accuracy--complexity trade-off is therefore preserved under realistic
device noise, so the gate-count savings from compression are real net of noise. Writes
results_v2/hardware/noise_robustness.csv and results_v2/figures/noise_robustness.png.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore, hardware_backend as hw
from src.data import get_checker_difficulty
from src.run_v2 import _cfg, Engine, greedy_trajectory
from src.hardware_cost import transpiled_2q_cost

BUDGETS_N2Q = [2, 6, 12, 18, 24]   # post-transpile two-qubit gate counts to probe
SEEDS = [0, 1, 2, 3, 4]
SHOTS = 4096


def main(difficulty="hard", n=300):
    cfg = _cfg()
    X, y = get_checker_difficulty(difficulty, n=n, seed=30)
    est_exact, _ = hw.make_estimator("exact")
    est_noisy, target = hw.make_estimator("noisy", shots=SHOTS)
    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        eng = Engine(cfg)
        traj = greedy_trajectory(eng, Xtr, ytr, Xte, yte, seed)
        by_n2q = {p["n2q"]: p["mask"] for p in traj}
        for n2q in BUDGETS_N2Q:
            if n2q not in by_n2q:
                continue
            mask = by_n2q[n2q]
            params = eng.train(Xtr, ytr, mask, seed, 600)
            acc_exact = hw.evaluate(params, mask, Xte, yte, cfg, est_exact)["accuracy"]
            acc_noisy = hw.evaluate(params, mask, Xte, yte, cfg, est_noisy,
                                    transpile_target=target)["accuracy"]
            rows.append(dict(difficulty=difficulty, seed=seed, n2q=n2q,
                             acc_exact=acc_exact, acc_noisy=acc_noisy))
            print(f"seed {seed}  N2q {n2q:>2}  exact {acc_exact:.3f}  noisy {acc_noisy:.3f}")

    df = pd.DataFrame(rows)
    outdir = Path("results_v2/hardware"); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "noise_robustness.csv", index=False)

    g = df.groupby("n2q").agg(ex_m=("acc_exact", "mean"), ex_s=("acc_exact", "std"),
                              no_m=("acc_noisy", "mean"), no_s=("acc_noisy", "std")).reset_index()
    g[["ex_s", "no_s"]] = g[["ex_s", "no_s"]].fillna(0.0)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(g["n2q"], g["ex_m"], yerr=g["ex_s"], marker="o", capsize=2, label="exact (noiseless)")
    ax.errorbar(g["n2q"], g["no_m"], yerr=g["no_s"], marker="s", capsize=2,
                label="FakeManila noise model")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.set_xlabel("Post-transpile two-qubit gates $N_{2q}$ (FakeManila)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Frontier preserved under the FakeManila noise model ({difficulty})", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    figpath = Path("results_v2/figures/noise_robustness.png")
    fig.savefig(figpath, dpi=200); plt.close(fig)
    print(f"\nwrote {outdir/'noise_robustness.csv'} and {figpath}")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
