"""
Classical baselines on the exact checkerboard task used for the quantum learner.

Purpose: contextualize the easy/medium/hard difficulty levels with standard
classical classifiers, so "difficulty" is anchored to task structure rather than
defined only by the quantum model's behaviour. We are NOT trying to beat these
baselines; the contribution is the hardware-aware compression frontier. The point
is to show the task is meaningful and that the difficulty knob (checkerboard
frequency) behaves sensibly for off-the-shelf models too.

The data, sample size, and train/test split are bit-identical to src/run_v2.py:
    X, y = get_checker_difficulty(level, n=200, seed=30)
    for seed in 0..4: train_test_split(test_size=0.3, random_state=seed, stratify=y)
Accuracy is averaged over the same five seeds.

Run (from repo root, with a venv that has numpy/scikit-learn):
    python scripts/classical_baselines.py
Writes results_v2/classical_baselines.csv and prints a summary table.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_checker_difficulty, CHECKER_DIFFICULTY  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N = 200
DATA_SEED = 30


def make_models():
    """Standard, lightly-tuned classical classifiers (scaled features)."""
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RBF-SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=4000, random_state=0),
        ),
    }


def evaluate(level: str):
    X, y = get_checker_difficulty(level, n=N, seed=DATA_SEED)
    accs = {name: [] for name in make_models()}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            accs[name].append(float(model.score(Xte, yte)))
    return {name: (float(np.mean(v)), float(np.std(v))) for name, v in accs.items()}


def main():
    outdir = Path("results_v2")
    outdir.mkdir(exist_ok=True)
    rows = []
    print(f"{'difficulty':<8} " + "  ".join(f"{n:>14}" for n in make_models()))
    for level in CHECKER_DIFFICULTY:
        res = evaluate(level)
        rows.append((level, res))
        cells = "  ".join(f"{m:.3f}+/-{s:.3f}" for m, s in res.values())
        print(f"{level:<8} {cells}")

    out = outdir / "classical_baselines.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "model", "mean_acc", "std_acc", "seeds", "n", "data_seed"])
        for level, res in rows:
            for name, (m, s) in res.items():
                w.writerow([level, name, f"{m:.6f}", f"{s:.6f}", len(SEEDS), N, DATA_SEED])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
