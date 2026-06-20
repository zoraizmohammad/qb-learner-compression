"""
make_figures_v2.py

Aggregate results_v2/<difficulty>/{frontier,lambda_sweep,mask_ablation}.csv into the
paper's figures and LaTeX tables. Every reported number is computed here from the
logged runs -- nothing is hand-entered.

Outputs (under results_v2/):
  figures/frontier_family.png      accuracy vs real FakeManila N2q, easy/medium/hard
  figures/frontier_ce.png          CE vs N2q (the smoother Pareto view)
  figures/mask_ablation.png        learned vs random masks by N2q (hard difficulty)
  tables/frontier.tex              mean+-std accuracy/CE per N2q per difficulty
  tables/lambda_sweep.tex          lambda -> operating point (hard difficulty)
  tables/mask_ablation.tex         learned vs random mean+-std by budget

Run after the sweeps finish:
  python scripts/make_figures_v2.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2"
FIGS = RES / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
TABS = RES / "tables"; TABS.mkdir(parents=True, exist_ok=True)
DIFFS = ["easy", "medium", "hard"]
COLORS = {"easy": "#2c7fb8", "medium": "#d95f0e", "hard": "#756bb1"}


def _load(diff, name):
    p = RES / diff / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def frontier_family():
    present = [d for d in DIFFS if (RES / d / "frontier.csv").exists()]
    if not present:
        print("no frontier.csv found yet; skipping frontier plots")
        return
    fig1, ax1 = plt.subplots(figsize=(5.0, 3.6))
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.6))
    table_rows = []
    for diff in present:
        df = _load(diff, "frontier")
        g = df.groupby("n2q").agg(acc_m=("test_acc", "mean"), acc_s=("test_acc", "std"),
                                  ce_m=("test_ce", "mean"), ce_s=("test_ce", "std")).reset_index()
        g["acc_s"] = g["acc_s"].fillna(0.0); g["ce_s"] = g["ce_s"].fillna(0.0)
        ax1.errorbar(g["n2q"], g["acc_m"], yerr=g["acc_s"], marker="o", capsize=2,
                     color=COLORS.get(diff), label=diff)
        ax2.errorbar(g["n2q"], g["ce_m"], yerr=g["ce_s"], marker="o", capsize=2,
                     color=COLORS.get(diff), label=diff)
        for _, r in g.iterrows():
            table_rows.append((diff, int(r["n2q"]), r["acc_m"], r["acc_s"], r["ce_m"], r["ce_s"]))
    for ax, ylab, fig, fn, title in [
            (ax1, "Test accuracy", fig1, "frontier_family.png",
             "Accuracy--complexity frontier"),
            (ax2, "Test cross-entropy", fig2, "frontier_ce.png",
             "Cross-entropy vs two-qubit cost")]:
        if ylab.endswith("accuracy"):
            ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
            ax.text(0.5, 0.505, "chance", fontsize=7, color="gray", va="bottom")
            # highlight the knee region (low two-qubit budget, where accuracy turns over)
            ax.axvspan(-0.5, 6, color="gold", alpha=0.10, lw=0)
            ax.text(2.7, ax.get_ylim()[0] + 0.04, "knee", fontsize=8,
                    color="#8a6d00", ha="center")
        ax.set_xlabel("Post-transpile two-qubit gates $N_{2q}$ (FakeManila)")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
        ax.legend(title="difficulty"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGS / fn, dpi=200); plt.close(fig)
    # LaTeX table
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Difficulty & $N_{2q}$ & Acc & Acc$_\sigma$ & CE & CE$_\sigma$ \\", r"\midrule"]
    for diff, n2q, am, as_, cm, cs in table_rows:
        lines.append(f"{diff} & {n2q} & {am:.3f} & {as_:.3f} & {cm:.3f} & {cs:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "frontier.tex").write_text("\n".join(lines))
    print(f"wrote {FIGS/'frontier_family.png'}, {FIGS/'frontier_ce.png'}, {TABS/'frontier.tex'}")


def ablation(diff="hard"):
    df = _load(diff, "mask_ablation")
    if df is None:
        print(f"no mask_ablation.csv for {diff}; skipping"); return
    g = df.groupby(["n2q", "mode"]).agg(acc_m=("test_acc", "mean"),
                                        acc_s=("test_acc", "std")).reset_index()
    g["acc_s"] = g["acc_s"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for mode, mk in [("learned", "o"), ("random", "s")]:
        sub = g[g["mode"] == mode].sort_values("n2q")
        ax.errorbar(sub["n2q"], sub["acc_m"], yerr=sub["acc_s"], marker=mk, capsize=2, label=mode)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.set_xlabel("Post-transpile two-qubit gates $N_{2q}$ (FakeManila)")
    ax.set_ylabel("Test accuracy"); ax.set_title(f"Learned vs random masks ({diff})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGS / "mask_ablation.png", dpi=200); plt.close(fig)
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"$N_{2q}$ & Learned Acc & Random Acc & $\Delta$ \\", r"\midrule"]
    for n2q in sorted(g["n2q"].unique()):
        le = g[(g.n2q == n2q) & (g["mode"] == "learned")]["acc_m"]
        ra = g[(g.n2q == n2q) & (g["mode"] == "random")]["acc_m"]
        if len(le) and len(ra):
            lines.append(f"{int(n2q)} & {le.iloc[0]:.3f} & {ra.iloc[0]:.3f} & {le.iloc[0]-ra.iloc[0]:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "mask_ablation.tex").write_text("\n".join(lines))
    print(f"wrote {FIGS/'mask_ablation.png'}, {TABS/'mask_ablation.tex'}")


def lambda_table(diff="hard"):
    df = _load(diff, "lambda_sweep")
    if df is None:
        print(f"no lambda_sweep.csv for {diff}; skipping"); return
    g = df.groupby("lam").agg(n2q=("n2q", "mean"), spars=("sparsity", "mean"),
                              acc=("test_acc", "mean"), ce=("test_ce", "mean")).reset_index()
    lines = [r"\begin{tabular}{ccccc}", r"\toprule",
             r"$\lambda$ & $N_{2q}$ & $s$ & Acc & CE \\", r"\midrule"]
    for _, r in g.iterrows():
        lines.append(f"{r['lam']:.3f} & {r['n2q']:.1f} & {r['spars']:.2f} & {r['acc']:.3f} & {r['ce']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "lambda_sweep.tex").write_text("\n".join(lines))
    print(f"wrote {TABS/'lambda_sweep.tex'}")
    print(g.to_string(index=False))


if __name__ == "__main__":
    frontier_family()
    ablation("hard")
    lambda_table("hard")
    print("done.")
