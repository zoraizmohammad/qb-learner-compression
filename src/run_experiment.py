"""
run_experiment.py

One-click launcher for quantum Bayesian learner experiments.

This script runs a single experiment defined in a YAML configuration file.
It supports both baseline and compressed training modes.

Usage:
    Run from repo root: python -m src.run_experiment --config experiments/baseline_pothos_small.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml"
    )


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
    FileNotFoundError
        If config file doesn't exist.
    yaml.YAMLError
        If YAML file is malformed.
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path_obj, 'r') as f:
        config = yaml.safe_load(f)
    
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
    # Check required fields
    required_fields = ["experiment_name", "mode", "n_qubits", "depth", 
                      "n_iterations", "dataset_name"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Validate mode
    mode = config["mode"].lower()
    if mode not in ["baseline", "compressed"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'baseline' or 'compressed'"
        )
    
    # Validate dataset name
    valid_datasets = ["pothos_chater_small", "pothos_chater_medium", "pothos_chater_large"]
    if config["dataset_name"] not in valid_datasets:
        raise ValueError(
            f"Invalid dataset_name: {config['dataset_name']}. "
            f"Must be one of: {valid_datasets}"
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
    kwargs = {
        "n_qubits": config["n_qubits"],
        "depth": config["depth"],
        "n_iterations": config["n_iterations"],
        "lr": config.get("lr", 0.01),
        "lam": config.get("lam", 0.1),
        "seed": config.get("seed", 42),
        "dataset_name": config["dataset_name"],
        "channel_strength": config.get("channel_strength", 0.4),
        "optimizer_type": config.get("optimizer", "finite_diff"),
        "debug_predictions_every": config.get("debug_predictions_every", None),
        "output_dir": "results",
    }
    
    # Add compressed-specific parameters
    if config["mode"].lower() == "compressed":
        kwargs["prune_every"] = config.get("prune_every", 20)
        kwargs["tolerance"] = config.get("tolerance", 0.01)
    
    return kwargs


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
    python -m src.run_experiment --config experiments/baseline_pothos_small.yaml

The YAML config should have:
    experiment_name: "baseline_pothos_small"
    mode: "baseline"  # or "compressed"
    n_qubits: 2
    depth: 3
    n_iterations: 60
    lr: 0.01
    lam: 0.1
    prune_every: 20       # only used if mode == "compressed"
    tolerance: 0.01       # only used if mode == "compressed"
    dataset_name: "pothos_chater_small"
    channel_strength: 0.4
    seed: 42
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., experiments/baseline_pothos_small.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load and validate configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    validate_config(config)
    
    experiment_name = config["experiment_name"]
    mode = config["mode"].lower()
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")
    
    # Get training arguments
    training_kwargs = get_training_kwargs(config)
    
    # Import and run appropriate training script
    if mode == "baseline":
        from .train_baseline import main as run_baseline
        print("Running baseline training...\n")
        results = run_baseline(**training_kwargs)
    elif mode == "compressed":
        from .train_compressed import main as run_compressed
        print("Running compressed training...\n")
        results = run_compressed(**training_kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("Experiment Completed Successfully!")
    print(f"{'='*60}")
    print(f"Experiment name: {experiment_name}")
    print(f"Mode: {mode}")
    print(f"Final loss: {results.get('loss', 0.0):.6f}")
    print(f"Final accuracy: {results.get('accuracy', 0.0):.4f}")
    print(f"Final 2-qubit gate count: {results.get('two_qubit_count', 0)}")
    
    if mode == "compressed" and "mask" in results:
        mask = results["mask"]
        sparsity = float(sum(mask == 1) / mask.size)
        print(f"Final mask sparsity: {sparsity:.4f}")
        print(f"Active entanglers: {sum(mask == 1)} / {mask.size}")
    
    print(f"\nResults saved to:")
    print(f"  - CSV log: results/logs/{mode}_log.csv")
    if mode == "baseline":
        print(f"  - Final theta: results/baseline_final_theta.npy")
    else:
        print(f"  - Final parameters: results/compressed_final.npz")
    print(f"  - Figures: results/figures/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

