"""
Shared CLI utilities for metrics modernization.

This module provides common CLI argument patterns and parsing logic
to eliminate duplication across implementation files.
"""

import argparse
from typing import Optional

from .config import Config


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create base argument parser with common arguments."""
    
    parser = argparse.ArgumentParser(description=description)
    
    add_common_arguments(parser)
    
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=Config.get_supported_datasets(),
        required=True,
        help=f"Dataset to evaluate ({', '.join(Config.get_supported_datasets())})"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to preprocessed data JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )


def add_model_argument(parser: argparse.ArgumentParser, default_model: Optional[str] = None) -> None:
    """Add model selection argument for modern implementations."""
    
    parser.add_argument(
        "--model",
        type=str,
        default=default_model or Config.DEFAULT_MODEL,
        help=f"Model to use for evaluation (default: {default_model or Config.DEFAULT_MODEL})"
    )


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sampling and reproducibility arguments."""
    
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Limit number of samples for testing (optional)"
    )
    
    parser.add_argument(
        "--seed",
        type=int, 
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-specific arguments."""
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=Config.EVALUATION_TIMEOUT,
        help=f"Evaluation timeout in seconds (default: {Config.EVALUATION_TIMEOUT})"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=Config.MAX_CONCURRENT_EVALUATIONS,
        help=f"Maximum concurrent evaluations (default: {Config.MAX_CONCURRENT_EVALUATIONS})"
    )


def validate_common_args(args: argparse.Namespace) -> None:
    """Validate common arguments and raise appropriate errors."""
    
    # Validate dataset support
    Config.validate_dataset_support(args.dataset)
    
    # Validate data file exists
    from pathlib import Path
    if not Path(args.data_file).exists():
        from .exceptions import DataLoadError
        raise DataLoadError(args.data_file, "Data file does not exist")
    
    # Validate output directory can be created
    output_path = Path(args.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        from .exceptions import ConfigurationError
        raise ConfigurationError("output_dir", f"Cannot create output directory: {e}")


def get_output_filename(dataset: str, implementation: str, metric: str) -> str:
    """Generate standardized output filename."""
    return f"{dataset}_{implementation}.json"


def print_evaluation_summary(
    dataset_name: str, 
    metric: str,
    implementation: str, 
    average_score: float,
    num_successful: int,
    total_samples: int,
    output_path: str
) -> None:
    """Print standardized evaluation summary."""
    
    metric_display = metric.replace('_', ' ').title()
    implementation_display = implementation.replace('_', ' ').title()
    
    print(f"\n=== {dataset_name} {metric_display} Evaluation Results ({implementation_display}) ===")
    print(f"Average {metric_display} Score: {average_score:.4f}")
    print(f"Successful evaluations: {num_successful}/{total_samples}")
    print(f"Results saved to: {output_path}")


def create_metric_parser(metric: str, implementation_type: str) -> argparse.ArgumentParser:
    """Create a parser for a specific metric and implementation type."""
    
    metric_display = metric.replace('_', ' ').title()
    impl_display = implementation_type.replace('_', ' ').title()
    
    description = f"Evaluate {metric_display} using {impl_display}"
    parser = create_base_parser(description)
    
    if implementation_type in ["modern_simplified", "modern_exact_replica"]:
        add_model_argument(parser)
    
    return parser