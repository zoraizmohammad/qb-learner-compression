"""Tests for src/hardware_cost.py: real FakeManila transpiled 2q cost."""
from __future__ import annotations

import numpy as np

from src.hardware_cost import transpiled_2q_cost


PAIRS = [(0, 1)]


def test_empty_mask_zero():
    """All-zero mask -> no entanglers -> transpiled 2q cost is 0."""
    depth = 4
    mask = np.zeros((depth, 1, 3), dtype=int)
    assert transpiled_2q_cost(2, depth, PAIRS, mask, "heisenberg") == 0


def test_monotonic_in_active_terms():
    """More active Heisenberg terms => non-decreasing N2q; full mask is positive."""
    depth = 4
    n_terms = depth * 1 * 3  # 12
    prev = -1
    counts = []
    for active in range(0, n_terms + 1):
        flat = np.zeros(n_terms, dtype=int)
        flat[:active] = 1
        mask = flat.reshape(depth, 1, 3)
        n2q = transpiled_2q_cost(2, depth, PAIRS, mask, "heisenberg")
        counts.append(n2q)
        assert n2q >= prev, f"N2q decreased at active={active}: {counts}"
        prev = n2q
    # full mask must produce a strictly positive 2q count
    assert counts[-1] > 0, f"full mask gave non-positive N2q: {counts[-1]}"


def test_deterministic():
    """Two identical calls return the same count."""
    depth = 4
    mask = np.ones((depth, 1, 3), dtype=int)
    a = transpiled_2q_cost(2, depth, PAIRS, mask, "heisenberg")
    b = transpiled_2q_cost(2, depth, PAIRS, mask, "heisenberg")
    assert a == b


def test_heisenberg_costs_more_than_hea():
    """At the full mask, Heisenberg (RXX+RYY+RZZ per edge) > HEA (one CX block)."""
    depth = 4
    mask = np.ones((depth, 1, 3), dtype=int)
    heis = transpiled_2q_cost(2, depth, PAIRS, mask, "heisenberg")
    hea = transpiled_2q_cost(2, depth, PAIRS, mask, "hea")
    assert heis > hea, f"expected heisenberg {heis} > hea {hea}"
