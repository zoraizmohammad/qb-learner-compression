"""
data.py

Toy categorization dataset inspired by Pothos & Chater (2002),
"A simplicity principle in unsupervised human categorization."

Pothos & Chater investigate how simple geometric structures in
stimulus space influence categorization. Following this spirit,
we construct a small 2D continuous dataset consisting of four
stimuli with interpretable category boundaries.

This dataset is intentionally minimal so that the quantum Bayesian
learner focuses on belief updating dynamics rather than large-scale
classification.

Reference (for documentation only):
- Pothos, E. M., & Chater, N. (2002). A simplicity principle in
  unsupervised human categorization. Cognitive Psychology, 45(1), 45–85.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Literal


DatasetName = Literal["pothos_chater_toy"]


def get_pothos_chater_toy() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a small 2D continuous dataset inspired by Pothos & Chater (2002).

    Stimuli (X):
        Four 2D points with simple geometric structure. Values
        are scaled to [0, 1] for consistency with quantum channel
        parameterization.

        X = [
            [0.2, 0.8],
            [0.8, 0.2],
            [0.7, 0.7],
            [0.1, 0.3],
        ]

    Labels (y):
        Binary category labels chosen to reflect a simple but
        non-axis-aligned categorization structure.

        y = [0, 1, 1, 0]

    Returns
    -------
    X : np.ndarray, shape (4, 2)
        2D features representing toy stimuli.

    y : np.ndarray, shape (4,)
        Integer labels in {0, 1}.
    """
    X = np.array(
        [
            [0.2, 0.8],
            [0.8, 0.2],
            [0.7, 0.7],
            [0.1, 0.3],
        ],
        dtype=float,
    )

    y = np.array([0, 1, 1, 0], dtype=int)

    return X, y


def get_toy_dataset(name: DatasetName = "pothos_chater_toy") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic entry point for the toy categorization dataset used in the
    quantum Bayesian learner experiments.

    Parameters
    ----------
    name : {"pothos_chater_toy"}
        Name of the dataset variant.

    Returns
    -------
    X : np.ndarray
        Feature matrix.

    y : np.ndarray
        Label vector.
    """
    if name == "pothos_chater_toy":
        return get_pothos_chater_toy()
    else:
        raise ValueError(f"Unknown dataset name: {name!r}")