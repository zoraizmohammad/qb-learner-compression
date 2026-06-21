"""
hardware_frontier.py

Map the accuracy--complexity frontier on a real IBM device: for several two-qubit
budgets along the greedy trajectory, evaluate the trained circuit both exactly and on
hardware. Writes results_v2/hardware/ibm_fez_frontier.json and a frontier figure.

Budget-conscious by default (16 points, 2048 shots, 5 budgets ~ 150 QPU-seconds).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore, hardware_backend as hw
from src.data import get_checker_difficulty
from src.run_v2 import _cfg, Engine, greedy_trajectory, N_QUBITS, DEPTH, PAIRS, ENTANGLER
from src.hardware_cost import transpiled_2q_cost

BUDGETS = [2, 6, 12, 18, 24]
EVAL_POINTS = 16
SHOTS = 2048
BACKEND = "ibm_fez"
SEED = 0


def main(difficulty="hard", n=160):
    cfg = _cfg()
    X, y = get_checker_difficulty(difficulty, n=n, seed=30)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xte, yte = Xte[:EVAL_POINTS], yte[:EVAL_POINTS]

    eng = Engine(cfg)
    print("training + greedy trajectory in simulation ...")
    traj = greedy_trajectory(eng, Xtr, ytr, Xte, yte, SEED)
    by_n2q = {p["n2q"]: p["mask"] for p in traj}

    est_exact, _ = hw.make_estimator("exact")
    print(f"connecting to {BACKEND} ...")
    est_ibm, target = hw.make_estimator("ibm", shots=SHOTS, ibm_backend=BACKEND)

    rows = []
    for n2q in BUDGETS:
        if n2q not in by_n2q:
            continue
        mask = by_n2q[n2q]
        params = eng.train(Xtr, ytr, mask, SEED, 600)
        acc_exact = hw.evaluate(params, mask, Xte, yte, cfg, est_exact)["accuracy"]
        acc_ibm = hw.evaluate(params, mask, Xte, yte, cfg, est_ibm, transpile_target=target)["accuracy"]
        rows.append(dict(n2q=int(n2q), acc_exact=acc_exact, acc_ibm=acc_ibm,
                         backend=BACKEND, n_eval=EVAL_POINTS, shots=SHOTS,
                         difficulty=difficulty, seed=SEED))
        print(f"N2q={n2q:>2}  exact={acc_exact:.3f}  {BACKEND}={acc_ibm:.3f}")

    outdir = Path("results_v2/hardware"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ibm_fez_frontier.json").write_text(json.dumps(rows, indent=2))

    rows = sorted(rows, key=lambda r: r["n2q"])
    n2qs = [r["n2q"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(n2qs, [r["acc_exact"] for r in rows], "o-", label="exact (noiseless)")
    ax.plot(n2qs, [r["acc_ibm"] for r in rows], "s-", label=f"{BACKEND} (real device)")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.text(max(n2qs), 0.505, "chance", fontsize=7, color="gray", va="bottom", ha="right")
    ax.set_xlabel("Post-transpile two-qubit gates $N_{2q}$ (FakeManila)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy--complexity frontier on real hardware ({BACKEND})", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    figp = Path("results_v2/figures/hardware_frontier.png")
    fig.savefig(figp, dpi=200); plt.close(fig)
    import shutil
    shutil.copy(figp, "paper/fig_hardware_frontier.png")
    print(f"\nwrote {outdir/'ibm_fez_frontier.json'}, {figp}, paper/fig_hardware_frontier.png")


if __name__ == "__main__":
    main()
