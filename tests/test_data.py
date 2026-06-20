"""Tests for src/data.py: Pothos-Chater style toy datasets."""
from __future__ import annotations

import numpy as np

from src import data


def _check_binary_in_unit_square(X, y, n_features=2):
    assert X.ndim == 2 and X.shape[1] == n_features
    assert X.shape[0] == y.shape[0]
    assert np.all(X >= 0.0) and np.all(X <= 1.0)
    assert set(np.unique(y).tolist()).issubset({0, 1})


def test_checker_shapes_and_labels():
    """Checkerboard returns X in [0,1], y in {0,1}, correct length; difficulty levels work."""
    n = 120
    X, y = data.get_pothos_chater_checker(freq=3, noise=0.0, n=n, seed=30)
    assert X.shape == (n, 2)
    assert y.shape == (n,)
    _check_binary_in_unit_square(X, y)

    # CHECKER_DIFFICULTY exposes easy/medium/hard
    assert set(data.CHECKER_DIFFICULTY.keys()) == {"easy", "medium", "hard"}

    # get_checker_difficulty works for each named level
    for level in ("easy", "medium", "hard"):
        Xl, yl = data.get_checker_difficulty(level, n=n, seed=30)
        assert Xl.shape == (n, 2)
        assert yl.shape == (n,)
        _check_binary_in_unit_square(Xl, yl)

    # noise actually flips some labels (caps accuracy below 1)
    X0, y0 = data.get_pothos_chater_checker(freq=3, noise=0.0, n=n, seed=30)
    Xn, yn = data.get_pothos_chater_checker(freq=3, noise=0.4, n=n, seed=30)
    assert np.any(y0 != yn)


def test_xor_and_parity():
    """XOR and parity datasets: expected shapes and balanced-ish binary labels."""
    # XOR: 4 prototypes, n_per_cluster each, 2 features
    n_per = 15
    Xx, yx = data.get_pothos_chater_xor(separation=0.6, std=0.1, n_per_cluster=n_per, seed=30)
    assert Xx.shape == (4 * n_per, 2)
    assert yx.shape == (4 * n_per,)
    _check_binary_in_unit_square(Xx, yx, n_features=2)
    # exactly balanced by construction (2 A-prototypes, 2 B-prototypes)
    assert int(np.sum(yx == 0)) == 2 * n_per
    assert int(np.sum(yx == 1)) == 2 * n_per

    # Parity: 2**n_features corners, n_per_corner each, n_features columns
    n_feat = 3
    n_corner = 8
    Xp, yp = data.get_pothos_chater_parity(
        n_features=n_feat, separation=0.5, std=0.1, n_per_corner=n_corner, seed=30
    )
    assert Xp.shape == (2 ** n_feat * n_corner, n_feat)
    assert yp.shape == (2 ** n_feat * n_corner,)
    _check_binary_in_unit_square(Xp, yp, n_features=n_feat)
    # parity over the hypercube corners is exactly balanced
    assert int(np.sum(yp == 0)) == int(np.sum(yp == 1))
