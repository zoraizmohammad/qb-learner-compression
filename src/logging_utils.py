"""
logging_utils.py

Logging utilities for quantum Bayesian learner experiments.

This module provides reusable functions for saving experiment outputs in
various formats (JSON, CSV, NPZ) with automatic directory creation and
consistent file organization.

All functions ensure parent directories exist before writing, making them
safe to use without manual directory setup.

Usage Examples:
--------------
    from src.logging_utils import write_json, write_csv, save_npz, timestamp_str
    
    # Save configuration
    config = {"n_qubits": 2, "depth": 3, "lr": 0.01}
    write_json("results/runs/my_run/config.json", config)
    
    # Save training history
    import pandas as pd
    df = pd.DataFrame({"iteration": [0, 1, 2], "loss": [0.5, 0.4, 0.3]})
    write_csv("results/runs/my_run/training_history.csv", df)
    
    # Save numpy arrays
    theta = np.random.randn(3, 2, 5)
    mask = np.ones((3, 1), dtype=int)
    save_npz("results/runs/my_run/params_final.npz", theta=theta, mask=mask)
    
    # Generate timestamp for run names
    run_name = f"baseline_{timestamp_str()}"
    print(run_name)  # e.g., "baseline_2025-02-14_23-14-10"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Timestamp Utilities
# ============================================================================

def timestamp_str() -> str:
    """
    Generate a clean timestamp string for filenames.
    
    Returns a timestamp in the format: "YYYY-MM-DD_HH-MM-SS"
    This format is filesystem-safe and sortable.
    
    Returns
    -------
    str
        Timestamp string, e.g., "2025-02-14_23-14-10"
    
    Examples
    --------
    >>> ts = timestamp_str()
    >>> print(ts)  # "2025-02-14_23-14-10"
    >>> run_name = f"baseline_{ts}"
    >>> print(run_name)  # "baseline_2025-02-14_23-14-10"
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


# ============================================================================
# File Writing Utilities
# ============================================================================

def write_json(
    path: Union[str, Path],
    obj: Dict[str, Any],
    indent: int = 2,
    verbose: bool = True
) -> None:
    """
    Write a dictionary to a JSON file with pretty formatting.
    
    Automatically creates parent directories if they don't exist.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to output JSON file.
    obj : Dict[str, Any]
        Dictionary to save as JSON.
    indent : int, optional
        Indentation level for pretty printing (default: 2).
    verbose : bool, optional
        If True, print confirmation message (default: True).
    
    Examples
    --------
    >>> config = {"n_qubits": 2, "depth": 3, "lr": 0.01}
    >>> write_json("results/runs/my_run/config.json", config)
    >>> # Saved config.json to results/runs/my_run/config.json
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, 'w') as f:
        json.dump(obj, f, indent=indent, sort_keys=False)
    
    if verbose:
        print(f"Saved JSON to {path_obj}")


def write_csv(
    path: Union[str, Path],
    dataframe: pd.DataFrame,
    index: bool = False,
    verbose: bool = True
) -> None:
    """
    Write a pandas DataFrame to a CSV file.
    
    Automatically creates parent directories if they don't exist.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to output CSV file.
    dataframe : pd.DataFrame
        DataFrame to save as CSV.
    index : bool, optional
        Whether to write row indices (default: False).
    verbose : bool, optional
        If True, print confirmation message (default: True).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"iteration": [0, 1, 2], "loss": [0.5, 0.4, 0.3]})
    >>> write_csv("results/runs/my_run/training_history.csv", df)
    >>> # Saved CSV to results/runs/my_run/training_history.csv
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    dataframe.to_csv(path_obj, index=index)
    
    if verbose:
        print(f"Saved CSV to {path_obj}")


def save_npz(
    path: Union[str, Path],
    verbose: bool = True,
    **arrays: np.ndarray
) -> None:
    """
    Save multiple numpy arrays to a .npz file.
    
    Automatically creates parent directories if they don't exist.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to output .npz file.
    verbose : bool, optional
        If True, print confirmation message (default: True).
    **arrays : np.ndarray
        Keyword arguments where keys are array names and values are arrays.
        Example: save_npz("file.npz", theta=theta, mask=mask)
    
    Examples
    --------
    >>> theta = np.random.randn(3, 2, 5)
    >>> mask = np.ones((3, 1), dtype=int)
    >>> save_npz("results/runs/my_run/params_final.npz", theta=theta, mask=mask)
    >>> # Saved NPZ to results/runs/my_run/params_final.npz
    >>> 
    >>> # Load later:
    >>> data = np.load("results/runs/my_run/params_final.npz")
    >>> theta = data["theta"]
    >>> mask = data["mask"]
    """
    if not arrays:
        raise ValueError("At least one array must be provided")
    
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(path_obj, **arrays)
    
    if verbose:
        array_names = ", ".join(arrays.keys())
        print(f"Saved NPZ to {path_obj} (arrays: {array_names})")


# ============================================================================
# Convenience Functions for Common Patterns
# ============================================================================

def save_training_config(
    run_dir: Union[str, Path],
    config: Dict[str, Any],
    verbose: bool = True
) -> None:
    """
    Save training configuration to config.json.
    
    Convenience wrapper around write_json for saving experiment configs.
    
    Parameters
    ----------
    run_dir : Union[str, Path]
        Run directory path (e.g., "results/runs/baseline_2025-02-14_23-14-10").
    config : Dict[str, Any]
        Configuration dictionary.
    verbose : bool, optional
        If True, print confirmation (default: True).
    
    Examples
    --------
    >>> config = {"n_qubits": 2, "depth": 3, "lr": 0.01}
    >>> save_training_config("results/runs/my_run", config)
    """
    run_dir_obj = Path(run_dir)
    config_path = run_dir_obj / "config.json"
    write_json(config_path, config, verbose=verbose)


def save_final_metrics(
    run_dir: Union[str, Path],
    metrics: Dict[str, Any],
    verbose: bool = True
) -> None:
    """
    Save final experiment metrics to final_metrics.json.
    
    Convenience wrapper around write_json for saving final metrics.
    
    Parameters
    ----------
    run_dir : Union[str, Path]
        Run directory path.
    metrics : Dict[str, Any]
        Metrics dictionary (e.g., {"loss": 0.3, "accuracy": 0.85}).
    verbose : bool, optional
        If True, print confirmation (default: True).
    
    Examples
    --------
    >>> metrics = {"loss": 0.3, "accuracy": 0.85, "two_qubit_count": 12}
    >>> save_final_metrics("results/runs/my_run", metrics)
    """
    run_dir_obj = Path(run_dir)
    metrics_path = run_dir_obj / "final_metrics.json"
    write_json(metrics_path, metrics, verbose=verbose)


def save_training_history(
    run_dir: Union[str, Path],
    history: pd.DataFrame,
    verbose: bool = True
) -> None:
    """
    Save training history DataFrame to training_history.csv.
    
    Convenience wrapper around write_csv for saving training logs.
    
    Parameters
    ----------
    run_dir : Union[str, Path]
        Run directory path.
    history : pd.DataFrame
        Training history DataFrame.
    verbose : bool, optional
        If True, print confirmation (default: True).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"iteration": [0, 1, 2], "loss": [0.5, 0.4, 0.3]})
    >>> save_training_history("results/runs/my_run", df)
    """
    run_dir_obj = Path(run_dir)
    history_path = run_dir_obj / "training_history.csv"
    write_csv(history_path, history, verbose=verbose)


def save_parameters(
    run_dir: Union[str, Path],
    theta: np.ndarray,
    mask: Optional[np.ndarray] = None,
    verbose: bool = True
) -> None:
    """
    Save final parameters to params_final.npz.
    
    Convenience wrapper around save_npz for saving model parameters.
    
    Parameters
    ----------
    run_dir : Union[str, Path]
        Run directory path.
    theta : np.ndarray
        Parameter tensor.
    mask : np.ndarray, optional
        Mask array (for compressed models). If None, only theta is saved.
    verbose : bool, optional
        If True, print confirmation (default: True).
    
    Examples
    --------
    >>> theta = np.random.randn(3, 2, 5)
    >>> mask = np.ones((3, 1), dtype=int)
    >>> save_parameters("results/runs/my_run", theta, mask=mask)
    """
    run_dir_obj = Path(run_dir)
    params_path = run_dir_obj / "params_final.npz"
    
    if mask is not None:
        save_npz(params_path, theta=theta, mask=mask, verbose=verbose)
    else:
        save_npz(params_path, theta=theta, verbose=verbose)


def save_loss_history(
    run_dir: Union[str, Path],
    loss: np.ndarray,
    ce_loss: np.ndarray,
    gate_cost: np.ndarray,
    verbose: bool = True
) -> None:
    """
    Save loss history arrays to loss_history.npz.
    
    Convenience wrapper around save_npz for saving loss sequences.
    
    Parameters
    ----------
    run_dir : Union[str, Path]
        Run directory path.
    loss : np.ndarray
        Total loss history (1D array).
    ce_loss : np.ndarray
        Cross-entropy loss history (1D array).
    gate_cost : np.ndarray
        Gate cost history (1D array).
    verbose : bool, optional
        If True, print confirmation (default: True).
    
    Examples
    --------
    >>> loss = np.array([0.5, 0.4, 0.3])
    >>> ce_loss = np.array([0.45, 0.35, 0.25])
    >>> gate_cost = np.array([12, 12, 12])
    >>> save_loss_history("results/runs/my_run", loss, ce_loss, gate_cost)
    """
    run_dir_obj = Path(run_dir)
    loss_path = run_dir_obj / "loss_history.npz"
    save_npz(
        loss_path,
        loss=loss,
        ce_loss=ce_loss,
        gate_cost=gate_cost,
        verbose=verbose
    )


def save_mask_history(
    run_dir: Union[str, Path],
    mask_history: np.ndarray,
    verbose: bool = True
) -> None:
    """
    Save mask history to mask_history.npz.
    
    For compressed training, saves the mask at each iteration.
    
    Parameters
    ----------
    run_dir : Union[str, Path]
        Run directory path.
    mask_history : np.ndarray
        Mask history array with shape (n_iterations, depth, n_edges).
    verbose : bool, optional
        If True, print confirmation (default: True).
    
    Examples
    --------
    >>> # For compressed training with 3 iterations, depth=3, n_edges=1
    >>> mask_history = np.array([[[1], [1], [1]], [[1], [0], [1]], [[0], [0], [1]]])
    >>> save_mask_history("results/runs/my_run", mask_history)
    """
    run_dir_obj = Path(run_dir)
    mask_path = run_dir_obj / "mask_history.npz"
    save_npz(mask_path, mask_history=mask_history, verbose=verbose)

