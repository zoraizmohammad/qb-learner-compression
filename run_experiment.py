"""
run_experiment.py

Main experiment runner for quantum Bayesian learner experiments.

This script orchestrates training runs by:
1. Parsing YAML configuration files
2. Calling appropriate training script (baseline or compressed)
3. Organizing outputs into experiment-specific directories
4. Saving final metrics, parameters, and logs
5. Generating summary reports

Usage:
    python run_experiment.py --config experiments/config_baseline.yaml

The YAML config should specify:
    - experiment_name: unique name for this experiment
    - mode: "baseline" or "compressed"
    - training: parameters for training (n_qubits, depth, etc.)
    - dataset: dataset name

All outputs are saved to results/<experiment_name>/ with organized subdirectories.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ============================================================================
# Configuration Parsing
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    
    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    
    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If config file doesn't exist.
    yaml.YAMLError
        If YAML file is malformed.
    
    Examples
    --------
    >>> config = load_config("experiments/config_baseline.yaml")
    >>> print(config["experiment_name"])
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to run experiments. "
            "Install it with: pip install pyyaml"
        )
    
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path_obj, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e
    
    if config is None:
        raise ValueError(f"Configuration file {config_path} is empty")
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary has required fields.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate.
    
    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """
    # Check required top-level fields
    required_fields = ["experiment_name", "mode", "training", "dataset"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Validate mode
    mode = config["mode"].lower()
    if mode not in ["baseline", "compressed"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'baseline' or 'compressed'"
        )
    
    # Validate training section
    training = config.get("training", {})
    required_training = ["n_qubits", "depth", "n_iterations"]
    for field in required_training:
        if field not in training:
            raise ValueError(f"Missing required training field: {field}")
    
    # Validate dataset section
    dataset = config.get("dataset", {})
    if "name" not in dataset:
        raise ValueError("Missing required dataset field: name")
    
    # Warn about missing optional fields
    optional_training = ["lr", "lam", "seed", "channel_strength", "optimizer", "debug_predictions_every"]
    for field in optional_training:
        if field not in training:
            # Only warn for important fields, not all optional ones
            if field in ["lr", "lam", "seed"]:
                warnings.warn(
                    f"Optional training field '{field}' not specified. Using default.",
                    UserWarning
                )
    
    if mode == "compressed":
        if "prune_every" not in training:
            warnings.warn(
                "prune_every not specified for compressed mode. Using default.",
                UserWarning
            )
        if "tolerance" not in training:
            warnings.warn(
                "tolerance not specified for compressed mode. Using default.",
                UserWarning
            )


def get_training_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training keyword arguments from config.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of keyword arguments for training function.
    """
    training = config.get("training", {})
    dataset = config.get("dataset", {})
    
    kwargs = {
        "n_qubits": training["n_qubits"],
        "depth": training["depth"],
        "n_iterations": training["n_iterations"],
        "lr": training.get("lr", 0.01),
        "lam": training.get("lam", 0.1),
        "seed": training.get("seed", 42),
        "dataset_name": dataset["name"],
        "channel_strength": training.get("channel_strength", 0.4),
        "optimizer_type": training.get("optimizer", "finite_diff"),
        "debug_predictions_every": training.get("debug_predictions_every", None),
    }
    
    # Add compressed-specific parameters
    if config["mode"].lower() == "compressed":
        kwargs["prune_every"] = training.get("prune_every", 20)
        kwargs["tolerance"] = training.get("tolerance", 0.01)
    
    return kwargs


# ============================================================================
# Directory Management
# ============================================================================

def setup_experiment_directories(experiment_name: str, base_dir: str = "results") -> Path:
    """
    Create directory structure for experiment outputs.
    
    Creates:
        results/
        results/logs/
        results/figures/
        results/saved_models/
        results/<experiment_name>/
        results/<experiment_name>/logs/
        results/<experiment_name>/figures/
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    base_dir : str, optional
        Base directory for results (default: "results").
    
    Returns
    -------
    Path
        Path to experiment-specific directory.
    
    Examples
    --------
    >>> exp_dir = setup_experiment_directories("baseline_test_01")
    >>> print(exp_dir)  # results/baseline_test_01
    """
    base_path = Path(base_dir)
    
    # Create base directories
    base_path.mkdir(exist_ok=True)
    (base_path / "logs").mkdir(exist_ok=True)
    (base_path / "figures").mkdir(exist_ok=True)
    (base_path / "saved_models").mkdir(exist_ok=True)
    
    # Create experiment-specific directory
    exp_dir = base_path / experiment_name
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    
    return exp_dir


# ============================================================================
# Output Management
# ============================================================================

def save_experiment_outputs(
    exp_dir: Path,
    config: Dict[str, Any],
    results: Dict[str, Any],
    mode: str
) -> None:
    """
    Save all experiment outputs to organized directories.
    
    Saves:
        - config_used.yaml: copy of original config
        - final_metrics.json: final metrics
        - theta.npy: final parameters
        - mask.npy: final mask (if compressed)
        - Copies CSV logs from training scripts
    
    Parameters
    ----------
    exp_dir : Path
        Experiment directory path.
    config : Dict[str, Any]
        Configuration dictionary.
    results : Dict[str, Any]
        Results dictionary from training function.
    mode : str
        Training mode: "baseline" or "compressed".
    
    Examples
    --------
    >>> save_experiment_outputs(exp_dir, config, results, "baseline")
    """
    # Save config copy
    if HAS_YAML:
        config_path = exp_dir / "config_used.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Saved config copy to {config_path}")
    
    # Prepare final metrics
    metrics = {
        "experiment_name": config["experiment_name"],
        "mode": mode,
        "final_loss": float(results.get("loss", 0.0)),
        "final_two_qubit_count": int(results.get("two_qubit_count", 0)),
        "final_accuracy": float(results.get("accuracy", 0.0)),
    }
    
    # Add compressed-specific metrics
    if mode == "compressed" and "mask" in results:
        mask = results["mask"]
        sparsity = float(np.sum(mask == 1) / mask.size)
        metrics["final_mask_sparsity"] = sparsity
        metrics["final_active_entanglers"] = int(np.sum(mask == 1))
        metrics["total_entanglers"] = int(mask.size)
    
    # Save metrics JSON
    metrics_path = exp_dir / "final_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved final metrics to {metrics_path}")
    
    # Save theta
    if "theta" in results:
        theta_path = exp_dir / "theta.npy"
        np.save(theta_path, results["theta"])
        print(f"Saved theta to {theta_path}")
    
    # Save mask (if compressed)
    if mode == "compressed" and "mask" in results:
        mask_path = exp_dir / "mask.npy"
        np.save(mask_path, results["mask"])
        print(f"Saved mask to {mask_path}")
    
    # Copy CSV logs from training scripts
    base_results = Path("results")
    if mode == "baseline":
        log_source = base_results / "logs" / "baseline_log.csv"
        log_dest = exp_dir / "logs" / "baseline_log.csv"
    else:
        log_source = base_results / "logs" / "compressed_log.csv"
        log_dest = exp_dir / "logs" / "compressed_log.csv"
    
    if log_source.exists():
        shutil.copy2(log_source, log_dest)
        print(f"Copied training log to {log_dest}")
    else:
        warnings.warn(
            f"Training log not found at {log_source}. "
            f"It may not have been generated.",
            UserWarning
        )
    
    # Copy figures (if they exist)
    figures_source = base_results / "figures"
    if figures_source.exists():
        copied_count = 0
        for fig_file in figures_source.glob("*.png"):
            # Copy figures that match the experiment mode or are general
            if mode in fig_file.name.lower() or "baseline" in fig_file.name.lower() or "compressed" in fig_file.name.lower():
                fig_dest = exp_dir / "figures" / fig_file.name
                shutil.copy2(fig_file, fig_dest)
                copied_count += 1
        if copied_count > 0:
            print(f"Copied {copied_count} figure(s) to {exp_dir / 'figures'}")


def print_experiment_summary(
    exp_dir: Path,
    config: Dict[str, Any],
    results: Dict[str, Any],
    mode: str
) -> None:
    """
    Print clean summary of experiment results.
    
    Parameters
    ----------
    exp_dir : Path
        Experiment directory path.
    config : Dict[str, Any]
        Configuration dictionary.
    results : Dict[str, Any]
        Results dictionary from training function.
    mode : str
        Training mode: "baseline" or "compressed".
    
    Examples
    --------
    >>> print_experiment_summary(exp_dir, config, results, "baseline")
    """
    print("\n" + "=" * 60)
    print("Experiment Completed")
    print("=" * 60)
    print(f"Name: {config['experiment_name']}")
    print(f"Mode: {mode}")
    print(f"Final Loss: {results.get('loss', 0.0):.6f}")
    print(f"Final Accuracy: {results.get('accuracy', 0.0):.4f}")
    print(f"Final 2Q Gates: {results.get('two_qubit_count', 0)}")
    
    if mode == "compressed" and "mask" in results:
        mask = results["mask"]
        sparsity = float(np.sum(mask == 1) / mask.size)
        print(f"Final Mask Sparsity: {sparsity:.4f}")
        print(f"Active Entanglers: {np.sum(mask == 1)} / {mask.size}")
    
    print(f"\nLog saved to: {exp_dir / 'logs'}")
    print(f"Figures saved to: {exp_dir / 'figures'}")
    print(f"Metrics saved to: {exp_dir / 'final_metrics.json'}")
    print("=" * 60)


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(
    config_path: str,
    base_output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run a complete experiment from a YAML configuration file.
    
    This is the main entry point that:
    1. Loads and validates the configuration
    2. Sets up directory structure
    3. Runs the appropriate training script
    4. Saves all outputs
    5. Prints summary
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    base_output_dir : str, optional
        Base directory for results (default: "results").
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary from training function.
    
    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If config file doesn't exist.
    ValueError
        If configuration is invalid.
    
    Examples
    --------
    >>> results = run_experiment("experiments/config_baseline.yaml")
    >>> print(results["loss"])
    """
    # Load and validate configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    validate_config(config)
    
    experiment_name = config["experiment_name"]
    mode = config["mode"].lower()
    
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {mode}")
    
    # Set up directories
    exp_dir = setup_experiment_directories(experiment_name, base_output_dir)
    print(f"Output directory: {exp_dir}")
    
    # Get training arguments
    training_kwargs = get_training_kwargs(config)
    
    # Override output_dir to point to base results (training scripts will use it)
    # But we'll copy outputs to exp_dir afterward
    training_kwargs["output_dir"] = base_output_dir
    
    # Import and run appropriate training script
    if mode == "baseline":
        from src.train_baseline import main as run_baseline
        print("\nRunning baseline training...")
        results = run_baseline(**training_kwargs)
    elif mode == "compressed":
        from src.train_compressed import main as run_compressed
        print("\nRunning compressed training...")
        results = run_compressed(**training_kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Save experiment outputs
    print(f"\nSaving experiment outputs to {exp_dir}...")
    save_experiment_outputs(exp_dir, config, results, mode)
    
    # Print summary
    print_experiment_summary(exp_dir, config, results, mode)
    
    return results


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> None:
    """
    Main entry point for command-line interface.
    
    Parses command-line arguments and runs the experiment.
    """
    parser = argparse.ArgumentParser(
        description="Run quantum Bayesian learner experiments from YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python run_experiment.py --config experiments/config_baseline.yaml

The YAML config should have:
    experiment_name: "my_experiment"
    mode: "baseline" or "compressed"
    training:
        n_qubits: 2
        depth: 3
        n_iterations: 100
        ...
    dataset:
        name: "pothos_chater_small"
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base output directory (default: results)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_experiment(
            config_path=args.config,
            base_output_dir=args.output_dir
        )
        print("\n✓ Experiment completed successfully!")
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()

