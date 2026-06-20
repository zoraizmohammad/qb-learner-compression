"""
run_v2.py

Phase 3/4 experiment runner for the hardware-aware quantum Bayesian learner.

Uses the exact differentiable core (src/qcore.py) for learning and the real
FakeManila transpile (src/hardware_cost.py) for the two-qubit cost.

Compression mechanism: GREEDY STRUCTURED PRUNING (robust; matches the paper's
"greedy prune"). Train the full circuit, then repeatedly remove the single Heisenberg
interaction term whose removal least increases training CE, retraining the continuous
parameters at each step. This greedy trajectory traces the accuracy-vs-N2q frontier
from the full circuit down to no entanglers. The hardware-aware objective
L = CE + lam * N2q selects an operating point on this trajectory for each lam.

Ablation: at pruned budgets we also train RANDOM masks with the same number of active
terms (connectivity-matched control). Learned (greedy) masks should beat random ones.

Performance: the mask is a RUNTIME argument (qcore.make_dynamic_fns), so evaluating /
training any mask reuses one compiled function (no per-mask JAX recompiles). Results
are written incrementally per seed, so a partial run is never lost.

All reported accuracy/CE/N2q use the DISCRETE deployed circuit (binary mask) and the
real FakeManila transpiled count.

CLI:
  python -m src.run_v2 --difficulty hard --seeds 0 1 2 3 4
  python -m src.run_v2 --all --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import get_checker_difficulty, CHECKER_DIFFICULTY
from . import qcore
from .hardware_cost import transpiled_2q_cost

# ---- canonical experimental configuration (validated to give a graded frontier) ----
N_QUBITS = 2
DEPTH = 4
PAIRS = [(0, 1)]
N_REUPLOAD = 2
READOUT = ("Z0", "Z1", "X0", "X1")   # single-qubit only -> entanglement carries info
ENTANGLER = "heisenberg"
DEFAULT_LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15)
N_RANDOM_PER_BUDGET = 3              # random masks per budget for the ablation
WARMUP_ITERS = 600
RETRAIN_ITERS = 300
LR = 0.05


def _cfg():
    return qcore.ModelConfig(n_qubits=N_QUBITS, depth=DEPTH, pairs=PAIRS,
                             n_reupload=N_REUPLOAD, readout_paulis=READOUT)


def _real_n2q(mask) -> int:
    return transpiled_2q_cost(N_QUBITS, DEPTH, PAIRS, np.asarray(mask, dtype=int), ENTANGLER)


class Engine:
    """Holds the one-time-compiled dynamic-mask functions for a config."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss, self.vg, self.predict = qcore.make_dynamic_fns(cfg)

    def train(self, Xtr, ytr, mask, seed, n_iters, warm=None):
        params = qcore.init_params(self.cfg, seed=seed) if warm is None else warm
        opt = qcore.Adam(params, lr=LR)
        Xj, yj = jnp.asarray(Xtr), jnp.asarray(ytr.astype(float))
        mj = jnp.asarray(np.asarray(mask, dtype=float))
        for _ in range(n_iters):
            _, g = self.vg(params, Xj, yj, mj)
            params = opt.step(params, g)
        return params

    def eval(self, params, mask, Xte, yte):
        mj = jnp.asarray(np.asarray(mask, dtype=float))
        p = np.asarray(self.predict(params, jnp.asarray(Xte), mj))
        acc = float(np.mean((p >= 0.5).astype(int) == np.asarray(yte)))
        ce = float(self.loss(params, jnp.asarray(Xte), jnp.asarray(yte.astype(float)), mj))
        return acc, ce

    def train_ce(self, params, Xtr, ytr, mask):
        mj = jnp.asarray(np.asarray(mask, dtype=float))
        return float(self.loss(params, jnp.asarray(Xtr), jnp.asarray(ytr.astype(float)), mj))


def greedy_trajectory(eng, Xtr, ytr, Xte, yte, seed):
    shape = (eng.cfg.depth, eng.cfg.n_edges, 3)
    mask = np.ones(shape, dtype=int)
    params = eng.train(Xtr, ytr, mask, seed, WARMUP_ITERS)
    pts = []

    def record(mask, params):
        acc, ce = eng.eval(params, mask, Xte, yte)
        pts.append(dict(active_terms=int(mask.sum()), n2q=_real_n2q(mask),
                        test_acc=acc, test_ce=ce,
                        train_ce=eng.train_ce(params, Xtr, ytr, mask), mask=mask.copy()))

    record(mask, params)
    while mask.sum() > 0:
        active = list(zip(*np.where(mask == 1)))
        best = None
        for (d, k, t) in active:
            trial = mask.copy(); trial[d, k, t] = 0
            ce = eng.train_ce(params, Xtr, ytr, trial)
            if best is None or ce < best[0]:
                best = (ce, (d, k, t))
        d, k, t = best[1]; mask[d, k, t] = 0
        params = eng.train(Xtr, ytr, mask, seed, RETRAIN_ITERS, warm=params)
        record(mask, params)
    return pts


def _append(path: Path, rows):
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def run_sweep(difficulty, seeds=(0, 1, 2, 3, 4), lambdas=DEFAULT_LAMBDAS,
              n=200, data_seed=30, outroot="results_v2"):
    cfg = _cfg(); eng = Engine(cfg)
    X, y = get_checker_difficulty(difficulty, n=n, seed=data_seed)
    outdir = Path(outroot) / difficulty
    outdir.mkdir(parents=True, exist_ok=True)
    # fresh start: clear any prior partial CSVs for this difficulty
    for f in ("frontier.csv", "lambda_sweep.csv", "mask_ablation.csv"):
        (outdir / f).unlink(missing_ok=True)
    n_terms = cfg.depth * cfg.n_edges * 3

    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        traj = greedy_trajectory(eng, Xtr, ytr, Xte, yte, seed)

        _append(outdir / "frontier.csv", [
            dict(difficulty=difficulty, seed=int(seed), active_terms=p["active_terms"],
                 n2q=p["n2q"], test_acc=p["test_acc"], test_ce=p["test_ce"], train_ce=p["train_ce"])
            for p in traj])

        _append(outdir / "lambda_sweep.csv", [
            (lambda b: dict(difficulty=difficulty, seed=int(seed), lam=float(lam),
                            active_terms=b["active_terms"], n2q=b["n2q"],
                            sparsity=b["active_terms"] / n_terms,
                            test_acc=b["test_acc"], test_ce=b["test_ce"]))(
                min(traj, key=lambda pt: pt["train_ce"] + lam * pt["n2q"]))
            for lam in lambdas])

        rng = np.random.default_rng(1000 + seed)
        ab_rows = []
        for p in traj:
            b = p["active_terms"]
            if 0 < b < n_terms:
                ab_rows.append(dict(difficulty=difficulty, seed=int(seed), budget=b,
                                    n2q=p["n2q"], mode="learned", test_acc=p["test_acc"]))
                for r in range(N_RANDOM_PER_BUDGET):
                    flat = np.zeros(n_terms, dtype=int)
                    flat[rng.choice(n_terms, size=b, replace=False)] = 1
                    rmask = flat.reshape(cfg.depth, cfg.n_edges, 3)
                    rp = eng.train(Xtr, ytr, rmask, seed * 100 + r, RETRAIN_ITERS)
                    racc, _ = eng.eval(rp, rmask, Xte, yte)
                    ab_rows.append(dict(difficulty=difficulty, seed=int(seed), budget=b,
                                        n2q=_real_n2q(rmask), mode="random", test_acc=racc))
        _append(outdir / "mask_ablation.csv", ab_rows)
        print(f"[{difficulty}] seed {seed} done ({len(traj)} frontier pts)")

    fr = pd.read_csv(outdir / "frontier.csv")
    print(f"\n[{difficulty}] frontier (mean over {fr['seed'].nunique()} seeds):")
    print(fr.groupby("n2q").agg(acc=("test_acc", "mean"), ce=("test_ce", "mean"))
          .reset_index().to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="hard", choices=list(CHECKER_DIFFICULTY))
    ap.add_argument("--all", action="store_true", help="run easy, medium, hard sequentially")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    targets = list(CHECKER_DIFFICULTY) if args.all else [args.difficulty]
    for diff in targets:
        run_sweep(diff, seeds=tuple(args.seeds))
