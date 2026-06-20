"""Shared pytest fixtures / setup for the qb-learner-compression test suite.

Ensures the repo root is on sys.path so ``import src...`` works regardless of
the directory pytest is invoked from.
"""
from __future__ import annotations

import os
import sys

# Repo root = parent of this tests/ directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
