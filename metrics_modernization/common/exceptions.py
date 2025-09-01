"""
Custom exception types for metrics modernization.

This module defines specific exception types to replace broad try-except blocks
and provide better error handling with clear failure modes.
"""


class MetricsModernizationError(Exception):
    """Base exception for all metrics modernization errors."""
    pass


class DatasetNotFoundError(MetricsModernizationError):
    """Raised when a requested dataset cannot be found or loaded."""
    
    def __init__(self, dataset_name: str, available_datasets: list = None):
        self.dataset_name = dataset_name
        self.available_datasets = available_datasets or []
        
        if self.available_datasets:
            message = f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(self.available_datasets)}"
        else:
            message = f"Dataset '{dataset_name}' not found"
            
        super().__init__(message)


class DataLoadError(MetricsModernizationError):
    """Raised when data loading fails due to format or content issues."""
    
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to load data from '{file_path}': {reason}")


class EvaluationError(MetricsModernizationError):
    """Raised when evaluation execution fails."""
    
    def __init__(self, metric: str, implementation: str, reason: str):
        self.metric = metric
        self.implementation = implementation
        self.reason = reason
        super().__init__(f"Evaluation failed for {metric}/{implementation}: {reason}")


class ResultSaveError(MetricsModernizationError):
    """Raised when saving evaluation results fails."""
    
    def __init__(self, output_path: str, reason: str):
        self.output_path = output_path
        self.reason = reason
        super().__init__(f"Failed to save results to '{output_path}': {reason}")


class ConfigurationError(MetricsModernizationError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, reason: str):
        self.config_key = config_key
        self.reason = reason
        super().__init__(f"Configuration error for '{config_key}': {reason}")


class UnsupportedDatasetError(MetricsModernizationError):
    """Raised when an implementation doesn't support a requested dataset."""
    
    def __init__(self, implementation: str, dataset: str, supported_datasets: list):
        self.implementation = implementation
        self.dataset = dataset
        self.supported_datasets = supported_datasets
        
        message = (f"Implementation '{implementation}' does not support dataset '{dataset}'. "
                  f"Supported datasets: {', '.join(supported_datasets)}")
        super().__init__(message)


class ImplementationNotFoundError(MetricsModernizationError):
    """Raised when a requested implementation cannot be found."""
    
    def __init__(self, metric: str, implementation: str, available_implementations: list = None):
        self.metric = metric
        self.implementation = implementation
        self.available_implementations = available_implementations or []
        
        if self.available_implementations:
            message = (f"Implementation '{implementation}' not found for metric '{metric}'. "
                      f"Available implementations: {', '.join(self.available_implementations)}")
        else:
            message = f"Implementation '{implementation}' not found for metric '{metric}'"
            
        super().__init__(message)


class DataValidationError(MetricsModernizationError):
    """Raised when data doesn't meet expected schema or format requirements."""
    
    def __init__(self, field: str, expected: str, actual: str = None):
        self.field = field
        self.expected = expected
        self.actual = actual
        
        if actual:
            message = f"Data validation failed for field '{field}': expected {expected}, got {actual}"
        else:
            message = f"Data validation failed for field '{field}': expected {expected}"
            
        super().__init__(message)