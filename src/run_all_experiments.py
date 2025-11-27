"""
run_all_experiments.py

Comprehensive experiment suite runner for quantum Bayesian learner.

This script discovers all YAML configuration files in experiments/,
runs training for each, organizes results, and generates comparison plots.

Usage:
    # Run all experiments
    python -m src.run_all_experiments
    
    # Dry run (list configs without running)
    python -m src.run_all_experiments --dry-run
    
    # Run only first 2 experiments
    python -m src.run_all_experiments --limit 2
    
    # Add tag to output directory
    python -m src.run_all_experiments --tag test_run
    
    # Skip failed experiments and continue
    python -m src.run_all_experiments --skip_failed
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML not found. Install with: pip install pyyaml")

from .train_baseline import main as run_baseline
from .train_compressed import main as run_compressed
from .plots import plot_pred_vs_true, plot_mask_heatmap, compare_methods_bar
from .logging_utils import write_json, write_csv, timestamp_str


# ============================================================================
# Configuration Loading and Validation
# ============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    
    Parameters
    ----------
    path : Path
        Path to YAML file.
    
    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    
    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If file doesn't exist.
    yaml.YAMLError
        If YAML is malformed.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required. Install with: pip install pyyaml"
        )
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML {path}: {e}") from e
    
    if config is None:
        raise ValueError(f"Configuration file {path} is empty")
    
    return config


def discover_configs(experiments_dir: Path) -> List[Path]:
    """
    Discover all YAML configuration files in experiments directory.
    
    Parameters
    ----------
    experiments_dir : Path
        Directory containing experiment configs.
    
    Returns
    -------
    List[Path]
        Sorted list of config file paths (without duplicates).
    """
    if not experiments_dir.exists():
        return []
    
    configs = set()
    for ext in ['.yaml', '.yml']:
        # Find files directly in experiments_dir
        configs.update(experiments_dir.glob(f'*{ext}'))
        # Also find files in subdirectories (recursive)
        configs.update(experiments_dir.glob(f'**/*{ext}'))
    
    # Remove duplicates and sort
    return sorted(list(configs))


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate configuration, applying defaults.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Raw configuration dictionary.
    
    Returns
    -------
    Dict[str, Any]
        Normalized configuration with defaults applied.
    """
    normalized = {}
    
    # Required fields - handle both "experiment_name" and "name"
    if 'experiment_name' not in config and 'name' not in config:
        raise ValueError("Config missing required field: experiment_name or name")
    normalized['experiment_name'] = config.get('experiment_name') or config.get('name', 'unnamed_experiment')
    
    # Handle mode - try to infer if missing
    if 'mode' not in config:
        # Try to infer from experiment name or default to baseline
        exp_name_lower = normalized['experiment_name'].lower()
        if 'compress' in exp_name_lower:
            mode = 'compressed'
        else:
            mode = 'baseline'
        warnings.warn(f"Config {normalized['experiment_name']} missing 'mode' field, defaulting to '{mode}'")
    else:
        mode = config['mode'].lower()
    
    if mode not in ['baseline', 'compressed']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'baseline' or 'compressed'")
    normalized['mode'] = mode
    
    # Extract training parameters (handle nested structure)
    training = config.get('training', {})
    if isinstance(training, dict):
        normalized['n_qubits'] = training.get('n_qubits', 2)
        normalized['depth'] = training.get('depth', 3)
        normalized['n_iterations'] = training.get('n_iterations', training.get('iterations', 100))
        normalized['lr'] = training.get('lr', training.get('learning_rate', 0.01))
        normalized['lam'] = training.get('lam', training.get('lambda', 0.1))
        normalized['seed'] = training.get('seed', 42)
        normalized['channel_strength'] = training.get('channel_strength', 0.4)
        normalized['optimizer_type'] = training.get('optimizer', training.get('optimizer_type', 'finite_diff'))
        normalized['debug_predictions_every'] = training.get('debug_predictions_every', None)
        
        # Compressed-specific
        if mode == 'compressed':
            normalized['prune_every'] = training.get('prune_every', 20)
            normalized['tolerance'] = training.get('tolerance', 0.01)
    else:
        # Fallback: assume flat structure
        normalized['n_qubits'] = config.get('n_qubits', 2)
        normalized['depth'] = config.get('depth', 3)
        normalized['n_iterations'] = config.get('n_iterations', config.get('iterations', 100))
        normalized['lr'] = config.get('lr', config.get('learning_rate', 0.01))
        normalized['lam'] = config.get('lam', config.get('lambda', 0.1))
        normalized['seed'] = config.get('seed', 42)
        normalized['channel_strength'] = config.get('channel_strength', 0.4)
        normalized['optimizer_type'] = config.get('optimizer', config.get('optimizer_type', 'finite_diff'))
        normalized['debug_predictions_every'] = config.get('debug_predictions_every', None)
        if mode == 'compressed':
            normalized['prune_every'] = config.get('prune_every', 20)
            normalized['tolerance'] = config.get('tolerance', 0.01)
    
    # Extract dataset
    dataset = config.get('dataset', {})
    if isinstance(dataset, dict):
        normalized['dataset_name'] = dataset.get('name', 'pothos_chater_small')
    else:
        normalized['dataset_name'] = config.get('dataset', 'pothos_chater_small')
    
    # Backend (optional)
    normalized['backend'] = config.get('backend', training.get('backend', None) if isinstance(training, dict) else None)
    
    # Store original config for reference
    normalized['_original_config'] = config
    
    return normalized


# ============================================================================
# Experiment Execution
# ============================================================================

def run_single_experiment(
    config: Dict[str, Any],
    base_output_dir: Path,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run a single experiment based on configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Normalized configuration dictionary.
    base_output_dir : Path
        Base output directory for results.
    verbose : bool, optional
        If True, print progress (default: True).
    
    Returns
    -------
    Tuple[Dict[str, Any], Optional[Dict[str, Any]]]
        (experiment_summary, training_results)
        training_results is None if training failed.
    """
    exp_name = config['experiment_name']
    mode = config['mode']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name} ({mode})")
        print(f"{'='*60}")
    
    # Create experiment-specific directory with timestamp to avoid overwriting
    timestamp = timestamp_str()
    exp_dir = base_output_dir / "experiments" / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    (exp_dir / "params").mkdir(exist_ok=True)
    
    # Prepare training arguments
    train_kwargs = {
        'n_qubits': config['n_qubits'],
        'depth': config['depth'],
        'n_iterations': config['n_iterations'],
        'lr': config['lr'],
        'lam': config['lam'],
        'seed': config['seed'],
        'dataset_name': config['dataset_name'],
        'channel_strength': config['channel_strength'],
        'optimizer_type': config['optimizer_type'],
        'debug_predictions_every': config['debug_predictions_every'],
        'output_dir': str(exp_dir),
    }
    
    # Add compressed-specific args
    if mode == 'compressed':
        train_kwargs['prune_every'] = config['prune_every']
        train_kwargs['tolerance'] = config['tolerance']
    
    # Run training
    try:
        if mode == 'baseline':
            results = run_baseline(**train_kwargs)
        elif mode == 'compressed':
            results = run_compressed(**train_kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Extract summary information
        summary = {
            'experiment_name': exp_name,
            'mode': mode,
            'n_qubits': config['n_qubits'],
            'depth': config['depth'],
            'n_iterations': config['n_iterations'],
            'final_loss': float(results.get('loss', np.nan)),
            'final_accuracy': float(results.get('accuracy', np.nan)),
            'two_qubit_count': int(results.get('two_qubit_count', 0)),
        }
        
        # Try to get CE loss from training history or log files
        try:
            history = results.get('history', None)
            if history is not None and hasattr(history, 'iloc'):
                # DataFrame from results
                if 'ce_loss' in history.columns and len(history) > 0:
                    summary['final_ce_loss'] = float(history.iloc[-1]['ce_loss'])
                else:
                    summary['final_ce_loss'] = summary['final_loss']  # Fallback
            else:
                # Try to read from main log file (training scripts write to results/logs/)
                log_path = Path("results") / "logs" / f"{mode}_log.csv"
                if log_path.exists():
                    df_log = pd.read_csv(log_path)
                    if 'ce_loss' in df_log.columns and len(df_log) > 0:
                        summary['final_ce_loss'] = float(df_log.iloc[-1]['ce_loss'])
                    else:
                        summary['final_ce_loss'] = summary['final_loss']
                else:
                    summary['final_ce_loss'] = summary['final_loss']  # Fallback
        except Exception:
            summary['final_ce_loss'] = summary['final_loss']  # Fallback
        
        # Add compressed-specific metrics
        if mode == 'compressed':
            mask = results.get('mask', None)
            if mask is not None:
                total_entanglers = mask.size
                active = int(np.sum(mask == 1))
                summary['active_entanglers'] = active
                summary['mask_sparsity'] = float(1.0 - active / total_entanglers)
            else:
                summary['active_entanglers'] = 0
                summary['mask_sparsity'] = np.nan
        else:
            # For baseline, estimate active entanglers (all are active)
            # Default: assume linear connectivity (n_qubits - 1 edges)
            n_edges = max(1, config['n_qubits'] - 1)
            summary['active_entanglers'] = config['depth'] * n_edges
        
        # Save summary JSON
        summary_path = exp_dir / "summary.json"
        write_json(summary_path, summary, verbose=False)
        
        # Copy config used
        config_path = exp_dir / "config_used.yaml"
        if HAS_YAML:
            with open(config_path, 'w') as f:
                yaml.dump(config['_original_config'], f, default_flow_style=False)
        
        if verbose:
            print(f"✓ Experiment {exp_name} completed successfully")
            print(f"  Final accuracy: {summary['final_accuracy']:.4f}")
            print(f"  Final loss: {summary['final_loss']:.6f}")
            print(f"  2Q gates: {summary['two_qubit_count']}")
        
        return summary, results
        
    except Exception as e:
        error_msg = f"Experiment {exp_name} failed: {str(e)}"
        if verbose:
            print(f"✗ {error_msg}")
            traceback.print_exc()
        
        # Save error information
        error_summary = {
            'experiment_name': exp_name,
            'mode': mode,
            'status': 'failed',
            'error': str(e),
        }
        error_path = exp_dir / "error.json"
        write_json(error_path, error_summary, verbose=False)
        
        return error_summary, None


# ============================================================================
# Results Aggregation and Comparison
# ============================================================================

def aggregate_results(
    all_summaries: List[Dict[str, Any]],
    summary_dir: Path
) -> None:
    """
    Aggregate all experiment results and generate comparison plots.
    
    Parameters
    ----------
    all_summaries : List[Dict[str, Any]]
        List of experiment summary dictionaries.
    summary_dir : Path
        Directory to save aggregated results.
    """
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out failed experiments
    successful = [s for s in all_summaries if s.get('status') != 'failed']
    
    if not successful:
        print("No successful experiments to aggregate.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(successful)
    
    # Save CSV table
    csv_path = summary_dir / "all_experiments_summary.csv"
    write_csv(csv_path, df, verbose=True)
    
    # Generate comparison plots
    import matplotlib.pyplot as plt
    
    # 1. Accuracy vs Gate Count scatter
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for mode in ['baseline', 'compressed']:
        mask_data = df[df['mode'] == mode]
        if len(mask_data) > 0:
            ax.scatter(
                mask_data['two_qubit_count'],
                mask_data['final_accuracy'],
                label=mode.capitalize(),
                alpha=0.7,
                s=100,
                edgecolors='black',
            )
    
    ax.set_xlabel('2-Qubit Gate Count')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Gate Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_path = summary_dir / "accuracy_vs_gatecount.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")
    
    # 2. Baseline vs Compressed bar chart
    if len(df[df['mode'] == 'baseline']) > 0 and len(df[df['mode'] == 'compressed']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Average metrics by mode
        baseline_avg = df[df['mode'] == 'baseline'].mean()
        compressed_avg = df[df['mode'] == 'compressed'].mean()
        
        metrics = ['final_accuracy', 'two_qubit_count']
        metric_labels = ['Accuracy', '2Q Gate Count']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            baseline_val = baseline_avg.get(metric, 0)
            compressed_val = compressed_avg.get(metric, 0)
            
            bars = ax.bar(
                ['Baseline', 'Compressed'],
                [baseline_val, compressed_val],
                alpha=0.7,
                edgecolor='black',
                color=['steelblue', 'lightcoral']
            )
            
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, [baseline_val, compressed_val]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{val:.4f}' if 'accuracy' in metric.lower() else f'{int(val)}',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )
        
        plt.tight_layout()
        comparison_path = summary_dir / "baseline_vs_compressed.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to {comparison_path}")
    
    # 3. Use compare_methods_bar to compare all experiments
    if len(successful) > 0:
        method_results = {}
        for summary in successful:
            exp_name = summary['experiment_name']
            # Use accuracy if available, otherwise approximate from CE loss
            if 'final_accuracy' in summary and not np.isnan(summary.get('final_accuracy', np.nan)):
                acc = summary['final_accuracy']
            elif 'final_ce_loss' in summary:
                # Approximate: lower CE loss ≈ higher accuracy
                ce_loss = summary['final_ce_loss']
                acc = max(0.0, min(1.0, 1.0 - ce_loss))
            else:
                acc = 0.0
            
            method_results[exp_name] = {
                'acc': acc,
                'cost': summary.get('two_qubit_count', 0),
            }
        
        if len(method_results) > 0:
            try:
                compare_methods_bar(
                    method_results,
                    fname='experiment_methods_comparison',
                    output_dir=str(summary_dir),
                    verbose=True,
                )
            except Exception as e:
                warnings.warn(f"Could not generate methods comparison bar chart: {e}")
    
    # 4. Experiment table (already saved as CSV above)
    print(f"Saved experiment table to {csv_path}")


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """Main entry point for experiment suite runner."""
    parser = argparse.ArgumentParser(
        description="Run all experiments from YAML configs in experiments/ directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments
    python -m src.run_all_experiments
    
    # Dry run (list configs without running)
    python -m src.run_all_experiments --dry-run
    
    # Run only first 2 experiments
    python -m src.run_all_experiments --limit 2
    
    # Add tag to output directory
    python -m src.run_all_experiments --tag test_run
        """
    )
    parser.add_argument(
        '--experiments_dir',
        type=str,
        default='experiments',
        help='Directory containing experiment configs (default: experiments)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Base output directory (default: results)'
    )
    parser.add_argument(
        '--skip_failed',
        action='store_true',
        help='Continue running even if some experiments fail'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List which configs would run without actually running them'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Run only first N experiments (default: all)'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Append tag to output directory name (default: timestamp only)'
    )
    
    args = parser.parse_args()
    
    # Check for YAML
    if not HAS_YAML:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)
    
    # Discover configs
    experiments_dir = Path(args.experiments_dir)
    config_paths = discover_configs(experiments_dir)
    
    if not config_paths:
        print(f"No YAML configs found in {experiments_dir}")
        return
    
    # Apply limit if specified
    if args.limit is not None:
        config_paths = config_paths[:args.limit]
    
    print("=" * 60)
    print("Experiment Suite Runner")
    print("=" * 60)
    print(f"Found {len(config_paths)} experiment config(s)")
    print(f"Experiments directory: {experiments_dir}")
    
    # Build output directory with tag if provided
    base_output_dir = Path(args.output_dir)
    if args.tag:
        timestamp = timestamp_str()
        base_output_dir = base_output_dir / f"experiments_{args.tag}_{timestamp}"
    print(f"Output directory: {base_output_dir}")
    print()
    
    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No experiments will be executed")
        print("-" * 60)
        for i, config_path in enumerate(config_paths, 1):
            print(f"[{i}] {config_path.name}")
            try:
                raw_config = load_yaml(config_path)
                config = normalize_config(raw_config)
                print(f"     Experiment: {config['experiment_name']}")
                print(f"     Mode: {config['mode']}")
                print(f"     n_qubits: {config['n_qubits']}, depth: {config['depth']}")
                print(f"     iterations: {config['n_iterations']}")
            except Exception as e:
                print(f"     ⚠ Warning: Could not parse config - {str(e)}")
        print("-" * 60)
        print(f"Would run {len(config_paths)} experiment(s)")
        return
    
    # Process each config
    all_summaries = []
    
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n[{i}/{len(config_paths)}] Processing: {config_path.name}")
        
        try:
            # Load and normalize config with improved error handling
            try:
                raw_config = load_yaml(config_path)
            except Exception as e:
                if HAS_YAML and isinstance(e, yaml.YAMLError):
                    error_msg = f"Malformed YAML in {config_path.name}: {str(e)}"
                else:
                    error_msg = f"Error loading {config_path.name}: {str(e)}"
                error_msg = f"Malformed YAML in {config_path.name}: {str(e)}"
                print(f"✗ {error_msg}")
                if args.skip_failed:
                    all_summaries.append({
                        'experiment_name': config_path.stem,
                        'status': 'failed',
                        'error': f"YAML parse error: {str(e)}",
                    })
                    continue
                else:
                    raise
            
            try:
                config = normalize_config(raw_config)
            except (ValueError, KeyError) as e:
                error_msg = f"Invalid config structure in {config_path.name}: {str(e)}"
                print(f"✗ {error_msg}")
                if args.skip_failed:
                    all_summaries.append({
                        'experiment_name': config_path.stem,
                        'status': 'failed',
                        'error': f"Config validation error: {str(e)}",
                    })
                    continue
                else:
                    raise
            
            # Run experiment
            summary, results = run_single_experiment(
                config,
                base_output_dir,
                verbose=args.verbose
            )
            
            all_summaries.append(summary)
            
        except Exception as e:
            error_msg = f"Failed to process {config_path.name}: {str(e)}"
            print(f"✗ {error_msg}")
            
            if args.verbose:
                traceback.print_exc()
            
            if not args.skip_failed:
                print("\nStopping due to error. Use --skip-failed to continue.")
                break
            
            # Add error summary
            all_summaries.append({
                'experiment_name': config_path.stem,
                'status': 'failed',
                'error': str(e),
            })
    
    # Aggregate results
    if all_summaries:
        print("\n" + "=" * 60)
        print("Aggregating results...")
        print("=" * 60)
        
        summary_dir = base_output_dir / "summary"
        aggregate_results(all_summaries, summary_dir)
        
        # Also save summary CSV to results/logs/experiment_summary.csv
        logs_dir = Path(args.output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        summary_csv_path = logs_dir / "experiment_summary.csv"
        
        df = pd.DataFrame(all_summaries)
        df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary CSV to: {summary_csv_path}")
        
        print("\n" + "=" * 60)
        print("All experiments completed.")
        print(f"Summary saved to: {summary_dir}")
        print(f"Summary CSV: {summary_csv_path}")
        print("=" * 60)
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    main()

