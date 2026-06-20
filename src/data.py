"""
data.py

Toy categorization dataset inspired by Pothos & Chater (2002),
'A simplicity principle in unsupervised human categorization.'
Cognitive Psychology, 45, 45–85.

The Pothos & Chater (P&C) framework uses continuous-valued
feature spaces where categories are formed from similarity
structures rather than strict binary dimensions. To create
a small but meaningful dataset consistent with their work,
we construct:

1) A SMALL dataset (8 stimuli):
   - Two category prototypes (A and B), each in R^2.
   - Four samples drawn around each prototype.

2) A MEDIUM dataset (20 stimuli):
   - Same prototypes as above.
   - Ten samples drawn from each category using Gaussian
     similarity consistent with P&C’s psychological distance
     assumptions.

All features lie in [0, 1], making them suitable for mapping
into quantum channel parameters.

References (documentation only):
- Pothos, E. M., & Chater, N. (2002). A simplicity principle
  in unsupervised human categorization. Cognitive Psychology.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Literal


DatasetName = Literal["pothos_chater_small", "pothos_chater_medium", "pothos_chater_large"]


# ---------------------------------------------------------
# Helper: generate samples around a category prototype
# ---------------------------------------------------------

def _sample_cluster(center: np.ndarray,
                    n: int,
                    std: float = 0.08,
                    seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=center, scale=std, size=(n, center.shape[0]))
    return np.clip(X, 0.0, 1.0)   # keep features in [0,1]


# ---------------------------------------------------------
# 1) SMALL P&C DATASET (8 total points)
# ---------------------------------------------------------

def get_pothos_chater_small() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a small categorization dataset inspired by
    Pothos & Chater’s similarity-based category structure.

    Two prototype category centers in R^2:
      Category A centered at (0.25, 0.25)
      Category B centered at (0.75, 0.75)

    Four points per category are sampled with low variance.

    Returns
    -------
    X : array, shape (8, 2)
    y : array, shape (8,)
    """

    center_A = np.array([0.25, 0.25])
    center_B = np.array([0.75, 0.75])

    XA = _sample_cluster(center_A, n=4, std=0.07, seed=1)
    XB = _sample_cluster(center_B, n=4, std=0.07, seed=2)

    X = np.vstack([XA, XB])
    y = np.array([0] * 4 + [1] * 4)

    return X, y


# ---------------------------------------------------------
# 2) MEDIUM P&C DATASET (20 total points)
# ---------------------------------------------------------

def get_pothos_chater_medium() -> Tuple[np.ndarray, np.ndarray]:
    """
    Larger dataset consistent with the P&C similarity
    framework. Two Gaussian-like clusters are generated
    around psychologically meaningful prototypes.

    Ten points per category.

    Returns
    -------
    X : array, shape (20, 2)
    y : array, shape (20,)
    """

    center_A = np.array([0.22, 0.30])
    center_B = np.array([0.78, 0.70])

    XA = _sample_cluster(center_A, n=10, std=0.10, seed=10)
    XB = _sample_cluster(center_B, n=10, std=0.10, seed=11)

    X = np.vstack([XA, XB])
    y = np.array([0] * 10 + [1] * 10)

    return X, y


# ---------------------------------------------------------
# 3) LARGE P&C DATASET (40 total points)
# ---------------------------------------------------------

def get_pothos_chater_large() -> Tuple[np.ndarray, np.ndarray]:
    """
    Large dataset consistent with the P&C similarity
    framework. Two Gaussian-like clusters are generated
    around psychologically meaningful prototypes.

    Twenty points per category.

    Category A prototype: [0.23, 0.27]
    Category B prototype: [0.77, 0.73]
    Standard deviation: 0.11

    Returns
    -------
    X : array, shape (40, 2)
    y : array, shape (40,)
    """

    center_A = np.array([0.23, 0.27])
    center_B = np.array([0.77, 0.73])

    XA = _sample_cluster(center_A, n=20, std=0.11, seed=20)
    XB = _sample_cluster(center_B, n=20, std=0.11, seed=21)

    X = np.vstack([XA, XB])
    y = np.array([0] * 20 + [1] * 20)

    return X, y


# ---------------------------------------------------------
# 4) STRUCTURED P&C DATASET WITH DIFFICULTY KNOB (XOR-style)
# ---------------------------------------------------------
#
# Motivation. Pothos & Chater categorization allows a category to be defined by
# MORE THAN ONE prototype: a category is a set of exemplars grouped by similarity,
# not necessarily a single Gaussian blob. We use a two-prototype-per-category
# arrangement on opposite diagonals of the feature square:
#
#     Category A prototypes: (lo, lo) and (hi, hi)   (the main diagonal)
#     Category B prototypes: (lo, hi) and (hi, lo)   (the anti-diagonal)
#
# This is a similarity-based ("simplicity principle") grouping in which the
# category boundary is NON-LINEAR (an XOR / parity structure). It is the natural
# choice for studying entangling capacity: a product (un-entangled) circuit that
# sees one feature per qubit cannot represent the feature interaction this boundary
# requires, whereas an entangled circuit can. Reducing the entangling structure
# therefore costs accuracy in a measurable way, which is exactly the
# accuracy--complexity trade-off this paper studies.
#
# A single ``separation`` knob (distance of prototypes from the center) together
# with the cluster ``std`` controls task difficulty / class overlap, so we can
# sweep difficulty levels and trace a family of accuracy--complexity frontiers.

def get_pothos_chater_xor(
    separation: float = 0.5,
    std: float = 0.12,
    n_per_cluster: int = 15,
    seed: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pothos--Chater style categorization with a non-linear (XOR/parity) boundary.

    Two prototypes per category placed on opposite diagonals, with Gaussian
    similarity clusters around each prototype. ``separation`` sets how far the
    prototypes sit from the center (larger = easier, more separated); ``std`` sets
    the within-cluster spread (larger = more overlap = harder).

    Parameters
    ----------
    separation : float
        Half-distance between the low/high prototype coordinates. Prototype
        coordinates are lo = 0.5 - separation/2, hi = 0.5 + separation/2,
        clipped to [0, 1].
    std : float
        Isotropic standard deviation of each Gaussian cluster.
    n_per_cluster : int
        Points sampled around each of the four prototypes.
    seed : int
        Base random seed (each cluster uses a distinct derived seed).

    Returns
    -------
    X : array, shape (4 * n_per_cluster, 2), values in [0, 1]
    y : array, shape (4 * n_per_cluster,), labels in {0, 1}
    """
    lo = float(np.clip(0.5 - separation / 2.0, 0.0, 1.0))
    hi = float(np.clip(0.5 + separation / 2.0, 0.0, 1.0))

    # Category A on the main diagonal, Category B on the anti-diagonal.
    protos_A = [np.array([lo, lo]), np.array([hi, hi])]
    protos_B = [np.array([lo, hi]), np.array([hi, lo])]

    blocks, labels = [], []
    for k, c in enumerate(protos_A):
        blocks.append(_sample_cluster(c, n=n_per_cluster, std=std, seed=seed + k))
        labels += [0] * n_per_cluster
    for k, c in enumerate(protos_B):
        blocks.append(_sample_cluster(c, n=n_per_cluster, std=std, seed=seed + 100 + k))
        labels += [1] * n_per_cluster

    X = np.vstack(blocks)
    y = np.array(labels)
    return X, y


def get_pothos_chater_parity(
    n_features: int = 3,
    separation: float = 0.5,
    std: float = 0.12,
    n_per_corner: int = 8,
    seed: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-feature Pothos--Chater categorization with a parity category rule.

    Prototypes sit at the 2**n_features corners of the feature hypercube (each
    coordinate either lo = 0.5 - separation/2 or hi = 0.5 + separation/2). A corner's
    category is the PARITY of how many coordinates are "high". Gaussian similarity
    clusters surround each prototype. For n_features=2 this reduces to the XOR task.

    Higher n_features requires entanglement distributed across more qubits, so that
    reducing the entangling structure degrades accuracy gradually -- producing a
    smooth accuracy--complexity frontier with a visible knee rather than a cliff.

    Returns
    -------
    X : array, shape (2**n_features * n_per_corner, n_features), values in [0, 1]
    y : array, shape (2**n_features * n_per_corner,), labels in {0, 1} (corner parity)
    """
    lo = float(np.clip(0.5 - separation / 2.0, 0.0, 1.0))
    hi = float(np.clip(0.5 + separation / 2.0, 0.0, 1.0))

    blocks, labels = [], []
    for corner in range(2 ** n_features):
        bits = [(corner >> b) & 1 for b in range(n_features)]
        center = np.array([hi if b else lo for b in bits])
        parity = sum(bits) % 2
        blocks.append(_sample_cluster(center, n=n_per_corner, std=std, seed=seed + corner))
        labels += [parity] * n_per_corner

    X = np.vstack(blocks)
    y = np.array(labels)
    return X, y


def get_pothos_chater_checker(
    freq: int = 3,
    noise: float = 0.0,
    n: int = 200,
    seed: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Structured categorization task with a controllable-complexity ("checkerboard")
    concept, in the spirit of Pothos & Chater's similarity-based grouping: nearby
    stimuli usually share a category (local similarity), but category membership
    depends on the JOINT configuration of both features, not either alone.

    label = (floor(freq*x0) + floor(freq*x1)) mod 2.

    The ``freq`` knob sets task complexity: higher frequency = finer cells = a more
    intricate boundary that requires more circuit capacity (entangling structure) to
    represent. This is the difficulty knob used for the difficulty sweep, and the
    reason the accuracy--complexity frontier degrades GRADUALLY (a smooth knee) rather
    than as an all-or-nothing cliff: capturing the boundary needs the feature
    interaction that only entangling gates provide, in graded amounts.

    Parameters
    ----------
    freq : int
        Checkerboard frequency per axis (>=2). 2 = coarse (XOR-like), larger = harder.
    noise : float
        Fraction of labels randomly flipped (caps achievable accuracy below 1).
    n : int
        Number of stimuli, sampled uniformly in the unit square.
    seed : int
        Random seed.

    Returns
    -------
    X : array, shape (n, 2), values in [0, 1]
    y : array, shape (n,), labels in {0, 1}
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    cell = np.floor(freq * X).sum(axis=1).astype(int)
    y = cell % 2
    if noise > 0:
        flip = rng.uniform(size=n) < noise
        y = np.where(flip, 1 - y, y)
    return X, y.astype(int)


# Named difficulty levels for the difficulty sweep. The difficulty knob is the
# checkerboard frequency: higher frequency => more intricate boundary => more
# entangling capacity required => a graded accuracy--complexity frontier.
# (Empirically validated in scripts/probe_checker.py: freq 2 = plateau+cliff,
#  freq 3 = mild graded decline, freq 4 = clear graded decline with a visible knee.)
CHECKER_DIFFICULTY: dict = {
    "easy":   dict(freq=2, noise=0.0),
    "medium": dict(freq=3, noise=0.0),
    "hard":   dict(freq=4, noise=0.0),
}


def get_checker_difficulty(level: str = "hard", n: int = 200, seed: int = 30):
    """Checkerboard dataset for a named difficulty level (the difficulty sweep)."""
    if level not in CHECKER_DIFFICULTY:
        raise ValueError(f"Unknown level {level!r}; choose from {list(CHECKER_DIFFICULTY)}")
    return get_pothos_chater_checker(n=n, seed=seed, **CHECKER_DIFFICULTY[level])


# Legacy XOR difficulty levels (separation/std). Larger separation and smaller
# std => easier (well-separated); smaller separation and larger std => harder
# (more class overlap, lower Bayes-optimal accuracy).
DIFFICULTY_LEVELS: dict = {
    "easy":   dict(separation=0.62, std=0.07),
    "medium": dict(separation=0.52, std=0.12),
    "hard":   dict(separation=0.44, std=0.16),
}


def get_difficulty_dataset(
    level: str = "medium",
    n_per_cluster: int = 15,
    seed: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning the XOR-style dataset for a named difficulty."""
    if level not in DIFFICULTY_LEVELS:
        raise ValueError(f"Unknown difficulty level {level!r}; choose from {list(DIFFICULTY_LEVELS)}")
    return get_pothos_chater_xor(n_per_cluster=n_per_cluster, seed=seed, **DIFFICULTY_LEVELS[level])


# ---------------------------------------------------------
# Generic entry point
# ---------------------------------------------------------

def get_toy_dataset(name: DatasetName = "pothos_chater_small") -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve a Pothos & Chater style dataset.

    Parameters
    ----------
    name : {"pothos_chater_small", "pothos_chater_medium", "pothos_chater_large"}

    Returns
    -------
    X : ndarray
    y : ndarray
    """
    if name == "pothos_chater_small":
        return get_pothos_chater_small()
    elif name == "pothos_chater_medium":
        return get_pothos_chater_medium()
    elif name == "pothos_chater_large":
        return get_pothos_chater_large()
    else:
        raise ValueError(f"Unknown dataset name: {name}")