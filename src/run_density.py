"""
run_density.py

Phase-D experiment runner for the LITERAL density-matrix quantum Bayesian learner
(src/qdensity.py): mixed-state belief, non-unital evidence channel with trace
renormalization. Same greedy structured-pruning compression and real FakeManila cost as
run_v2, but with the density-matrix model, so we can confirm the accuracy--complexity
frontier and the learned-vs-random effect persist for the principled cognitive model.

Writes results_v2_density/<difficulty>/{frontier,lambda_sweep,mask_ablation}.csv.

CLI:  python -m src.run_density --all --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import get_checker_difficulty, CHECKER_DIFFICULTY
from . import qdensity as qd
from .hardware_cost import transpiled_2q_cost

N_QUBITS, DEPTH, PAIRS = 2, 4, [(0, 1)]
N_REUPLOAD = 2
READOUT = ("Z0", "Z1", "X0", "X1")
ENTANGLER = "heisenberg"
DEFAULT_LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15)
N_RANDOM_PER_BUDGET = 2
WARMUP_ITERS = 500
RETRAIN_ITERS = 250
LR = 0.05


def _cfg():
    return qd.DensityConfig(n_qubits=N_QUBITS, depth=DEPTH, pairs=PAIRS, n_reupload=N_REUPLOAD,
                            readout_paulis=READOUT, prior="pure", damping=True)


def _n2q(mask):
    return transpiled_2q_cost(N_QUBITS, DEPTH, PAIRS, np.asarray(mask, dtype=int), ENTANGLER)


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss, self.vg, self.predict = qd.make_dynamic_fns(cfg)

    def train(self, Xtr, ytr, mask, seed, n_iters, warm=None):
        params = qd.init_params(self.cfg, seed=seed) if warm is None else warm
        opt = qd.Adam(params, lr=LR)
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
    mask = np.ones((eng.cfg.depth, eng.cfg.n_edges, 3), dtype=int)
    params = eng.train(Xtr, ytr, mask, seed, WARMUP_ITERS)
    pts = []

    def record(mask, params):
        acc, ce = eng.eval(params, mask, Xte, yte)
        pts.append(dict(active_terms=int(mask.sum()), n2q=_n2q(mask), test_acc=acc, test_ce=ce,
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


def _append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_sweep(difficulty, seeds=(0, 1, 2), lambdas=DEFAULT_LAMBDAS, n=200, data_seed=30,
              outroot="results_v2_density"):
    cfg = _cfg(); eng = Engine(cfg)
    X, y = get_checker_difficulty(difficulty, n=n, seed=data_seed)
    outdir = Path(outroot) / difficulty
    outdir.mkdir(parents=True, exist_ok=True)
    for f in ("frontier.csv", "lambda_sweep.csv", "mask_ablation.csv"):
        (outdir / f).unlink(missing_ok=True)
    n_terms = cfg.depth * cfg.n_edges * 3

    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        traj = greedy_trajectory(eng, Xtr, ytr, Xte, yte, seed)
        _append(outdir / "frontier.csv", [
            dict(difficulty=difficulty, seed=int(seed), active_terms=p["active_terms"], n2q=p["n2q"],
                 test_acc=p["test_acc"], test_ce=p["test_ce"], train_ce=p["train_ce"]) for p in traj])
        _append(outdir / "lambda_sweep.csv", [
            (lambda b: dict(difficulty=difficulty, seed=int(seed), lam=float(lam),
                            active_terms=b["active_terms"], n2q=b["n2q"], sparsity=b["active_terms"]/n_terms,
                            test_acc=b["test_acc"], test_ce=b["test_ce"]))(
                min(traj, key=lambda pt: pt["train_ce"] + lam*pt["n2q"])) for lam in lambdas])
        rng = np.random.default_rng(1000 + seed)
        ab = []
        for p in traj:
            b = p["active_terms"]
            if 0 < b < n_terms:
                ab.append(dict(difficulty=difficulty, seed=int(seed), budget=b, n2q=p["n2q"],
                               mode="learned", test_acc=p["test_acc"]))
                for r in range(N_RANDOM_PER_BUDGET):
                    flat = np.zeros(n_terms, dtype=int); flat[rng.choice(n_terms, size=b, replace=False)] = 1
                    rmask = flat.reshape(cfg.depth, cfg.n_edges, 3)
                    rp = eng.train(Xtr, ytr, rmask, seed*100+r, RETRAIN_ITERS)
                    racc, _ = eng.eval(rp, rmask, Xte, yte)
                    ab.append(dict(difficulty=difficulty, seed=int(seed), budget=b, n2q=_n2q(rmask),
                                   mode="random", test_acc=racc))
        _append(outdir / "mask_ablation.csv", ab)
        print(f"[{difficulty}] seed {seed} done ({len(traj)} pts)")

    fr = pd.read_csv(outdir / "frontier.csv")
    print(f"[{difficulty}] frontier (mean over {fr['seed'].nunique()} seeds):")
    print(fr.groupby("n2q").agg(acc=("test_acc", "mean")).reset_index().to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="hard", choices=list(CHECKER_DIFFICULTY))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    targets = list(CHECKER_DIFFICULTY) if args.all else [args.difficulty]
    for d in targets:
        run_sweep(d, seeds=tuple(args.seeds))
