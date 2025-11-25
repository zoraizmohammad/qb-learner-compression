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


DatasetName = Literal["pothos_chater_small", "pothos_chater_medium"]


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
# Generic entry point
# ---------------------------------------------------------

def get_toy_dataset(name: DatasetName = "pothos_chater_small") -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve a Pothos & Chater style dataset.

    Parameters
    ----------
    name : {"pothos_chater_small", "pothos_chater_medium"}

    Returns
    -------
    X : ndarray
    y : ndarray
    """
    if name == "pothos_chater_small":
        return get_pothos_chater_small()
    elif name == "pothos_chater_medium":
        return get_pothos_chater_medium()
    else:
        raise ValueError(f"Unknown dataset name: {name}")