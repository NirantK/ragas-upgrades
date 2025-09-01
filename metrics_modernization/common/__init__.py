"""
Common utilities for metrics modernization.

This package provides shared functionality across all metric implementations
to eliminate code duplication and improve maintainability.
"""

from .exceptions import (
    MetricsModernizationError,
    DatasetNotFoundError,
    DataLoadError,
    EvaluationError,
    ResultSaveError,
    ConfigurationError,
    UnsupportedDatasetError,
)

from .config import Config
from .data_loader import DataLoader
from .result_saver import ResultSaver
from .cli_utils import create_base_parser, add_common_arguments

__all__ = [
    "MetricsModernizationError",
    "DatasetNotFoundError", 
    "DataLoadError",
    "EvaluationError",
    "ResultSaveError",
    "ConfigurationError",
    "UnsupportedDatasetError",
    "Config",
    "DataLoader",
    "ResultSaver",
    "create_base_parser",
    "add_common_arguments",
]