"""
Paired statistical support for the learned-vs-random mask claim.

For the hard task, at each (seed, budget) we pair the greedy/learned mask accuracy
against the mean accuracy of the matched random masks (same number of active terms).
We report:
  - number of paired (seed, budget) comparisons in which learned >= random and > random,
  - mean paired advantage +/- std,
  - a one-sided Wilcoxon signed-rank test (learned > random).

Reads results_v2/<difficulty>/mask_ablation.csv. No model re-runs.
Run: python scripts/paired_stats.py [difficulty]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load(difficulty: str):
    path = Path("results_v2") / difficulty / "mask_ablation.csv"
    learned = {}          # (seed, budget) -> acc
    random = defaultdict(list)  # (seed, budget) -> [acc, ...]
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), int(row["budget"]))
            acc = float(row["test_acc"])
            if row["mode"] == "learned":
                learned[key] = acc
            else:
                random[key].append(acc)
    return learned, random


def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "hard"
    learned, random = load(difficulty)
    keys = sorted(k for k in learned if k in random)
    L = np.array([learned[k] for k in keys])
    R = np.array([np.mean(random[k]) for k in keys])
    d = L - R

    n = len(keys)
    n_ge = int(np.sum(d >= -1e-12))
    n_gt = int(np.sum(d > 1e-12))
    stat, p = wilcoxon(L, R, alternative="greater", zero_method="wilcox")

    print(f"difficulty={difficulty}")
    print(f"paired (seed,budget) comparisons: {n}")
    print(f"learned >= random: {n_ge}/{n}   learned > random: {n_gt}/{n}")
    print(f"mean paired advantage: {d.mean():+.4f} +/- {d.std():.4f}")
    print(f"Wilcoxon signed-rank (one-sided, learned>random): W={stat:.1f}, p={p:.2e}")


if __name__ == "__main__":
    main()
