"""Tests for src/run_v2.py: the Engine and greedy pruning trajectory."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from src import run_v2, qcore
from src.data import get_checker_difficulty


def _split(n=60, seed=0):
    X, y = get_checker_difficulty("easy", n=n, seed=30)
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def test_engine_trains_and_evals():
    """Engine trains a fixed full mask a few iters; eval gives acc in [0,1] and finite CE."""
    cfg = run_v2._cfg()
    eng = run_v2.Engine(cfg)
    Xtr, Xte, ytr, yte = _split(seed=0)

    mask = qcore.full_mask(cfg)
    params = eng.train(Xtr, ytr, mask, seed=0, n_iters=15)
    acc, ce = eng.eval(params, mask, Xte, yte)

    assert 0.0 <= acc <= 1.0
    assert np.isfinite(ce)
    assert ce >= 0.0  # mean BCE is non-negative

    # train_ce is also finite and non-negative
    tce = eng.train_ce(params, Xtr, ytr, mask)
    assert np.isfinite(tce) and tce >= 0.0


def test_greedy_trajectory_monotone_n2q(monkeypatch):
    """greedy_trajectory: n2q strictly decreases full->0, active_terms 12 down to 0."""
    # Keep it fast: tiny iteration counts.
    monkeypatch.setattr(run_v2, "WARMUP_ITERS", 8)
    monkeypatch.setattr(run_v2, "RETRAIN_ITERS", 4)

    cfg = run_v2._cfg()
    eng = run_v2.Engine(cfg)
    Xtr, Xte, ytr, yte = _split(n=48, seed=0)

    pts = run_v2.greedy_trajectory(eng, Xtr, ytr, Xte, yte, seed=0)

    n_terms = cfg.depth * cfg.n_edges * 3  # 12
    # one record at full mask + one per removed term down to 0 => n_terms + 1 points
    assert len(pts) == n_terms + 1

    active = [p["active_terms"] for p in pts]
    n2q = [p["n2q"] for p in pts]

    # active_terms goes from 12 down to 0, dropping by exactly one each step
    assert active[0] == n_terms
    assert active[-1] == 0
    assert active == list(range(n_terms, -1, -1))

    # n2q strictly decreases from full circuit to the empty (0) circuit
    assert n2q[0] > 0
    assert n2q[-1] == 0
    for a, b in zip(n2q, n2q[1:]):
        assert b < a, f"n2q not strictly decreasing: {n2q}"

    # sanity on recorded metrics
    for p in pts:
        assert 0.0 <= p["test_acc"] <= 1.0
        assert np.isfinite(p["test_ce"]) and np.isfinite(p["train_ce"])
