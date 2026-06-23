"""
make_figures_density.py

Aggregate the density-matrix model's results (results_v2_density/) into figures for the
paper: the accuracy--complexity frontier family and the learned-vs-random mask ablation.
Every number comes from the logged runs. Outputs into results_v2_density/figures/ and
copies the two paper figures into paper/.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2_density"
FIGS = RES / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
DIFFS = ["easy", "medium", "hard"]
COLORS = {"easy": "#2c7fb8", "medium": "#d95f0e", "hard": "#756bb1"}


def frontier_family():
    present = [d for d in DIFFS if (RES / d / "frontier.csv").exists()]
    if not present:
        print("no density frontier.csv yet"); return
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for diff in present:
        df = pd.read_csv(RES / diff / "frontier.csv")
        g = df.groupby("n2q").agg(m=("test_acc", "mean"), s=("test_acc", "std")).reset_index()
        g["s"] = g["s"].fillna(0.0)
        ax.errorbar(g["n2q"], g["m"], yerr=g["s"], marker="o", capsize=2,
                    color=COLORS.get(diff), label=diff)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.text(24, 0.505, "chance", fontsize=7, color="gray", va="bottom", ha="right")
    ax.axvspan(-0.5, 6, color="gold", alpha=0.10, lw=0)
    ax.set_xlabel("Post-transpile two-qubit gates $N_{2q}$ (FakeManila)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Density-matrix learner: accuracy--complexity frontier", fontsize=9)
    ax.legend(title="difficulty"); ax.grid(alpha=0.3); fig.tight_layout()
    out = FIGS / "density_frontier_family.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    shutil.copy(out, ROOT / "paper/fig_density_frontier.png")
    print(f"wrote {out} + paper/fig_density_frontier.png")


def ablation(diff="hard"):
    p = RES / diff / "mask_ablation.csv"
    if not p.exists():
        print(f"no density ablation for {diff}"); return
    df = pd.read_csv(p)
    g = df.groupby(["n2q", "mode"]).agg(m=("test_acc", "mean"), s=("test_acc", "std")).reset_index()
    g["s"] = g["s"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for mode, mk in [("learned", "o"), ("random", "s")]:
        sub = g[g["mode"] == mode].sort_values("n2q")
        ax.errorbar(sub["n2q"], sub["m"], yerr=sub["s"], marker=mk, capsize=2, label=mode)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.set_xlabel("Post-transpile two-qubit gates $N_{2q}$ (FakeManila)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Density-matrix learner: learned vs random masks ({diff})", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    out = FIGS / "density_mask_ablation.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    shutil.copy(out, ROOT / "paper/fig_density_ablation.png")
    print(f"wrote {out} + paper/fig_density_ablation.png")


if __name__ == "__main__":
    frontier_family()
    ablation("hard")
    print("done.")
