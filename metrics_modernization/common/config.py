"""
Configuration management for metrics modernization.

Centralizes configuration constants and provides environment-based overrides.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass 
class DatasetConfig:
    """Configuration for a dataset."""
    hf_name: str
    hf_config: Optional[str]
    split: str
    display_name: str
    question_field: str
    answer_field: str
    ground_truth_field: str
    contexts_field: str


class Config:
    """Central configuration manager."""
    
    # Dataset configurations
    DATASETS: Dict[str, DatasetConfig] = {
        "amnesty": DatasetConfig(
            hf_name="explodinggradients/amnesty_qa",
            hf_config="english_v3",
            split="eval",
            display_name="AmnestyQA",
            question_field="user_input",
            answer_field="response", 
            ground_truth_field="reference",
            contexts_field="retrieved_contexts"
        ),
        "fiqa": DatasetConfig(
            hf_name="explodinggradients/fiqa",
            hf_config="ragas_eval",
            split="baseline",
            display_name="FIQA",
            question_field="question",
            answer_field="answer",
            ground_truth_field="ground_truths",
            contexts_field="contexts"
        )
    }
    
    # Metric configuration
    METRIC_FIELD_NAMES: Dict[str, str] = {
        "faithfulness": "average_faithfulness",
        "answer_relevance": "average_answer_relevance", 
        "answer_correctness": "average_answer_correctness",
        "context_recall": "average_context_recall",
        "context_precision": "average_context_precision"
    }
    
    # Default model configurations
    DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EVALUATION_TIMEOUT = int(os.getenv("EVALUATION_TIMEOUT", "600"))  # 10 minutes
    MAX_CONCURRENT_EVALUATIONS = int(os.getenv("MAX_CONCURRENT", "10"))
    
    # File path patterns  
    RESULT_FILE_PATTERNS: List[str] = [
        "{dataset}_{implementation}.json",
        "{dataset}_*.json", 
        "*_{implementation}.json",
        "*.json"
    ]
    
    # Framework detection patterns
    FRAMEWORK_DETECTION: Dict[str, List[str]] = {
        "ragas_main": ["from ragas import", "import ragas", "ragas.metrics"],
        "modern_simplified": ["AsyncOpenAI", "openai", "beta.chat.completions.parse"],
        "modern_exact_replica": ["AsyncOpenAI", "exact_replica"]
    }
    
    # Required data fields for validation
    REQUIRED_DATA_FIELDS: List[str] = ["question", "answer"]
    OPTIONAL_DATA_FIELDS: List[str] = ["ground_truth", "contexts"]
    
    @classmethod
    def get_dataset_config(cls, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset."""
        if dataset_name not in cls.DATASETS:
            available = list(cls.DATASETS.keys())
            from .exceptions import DatasetNotFoundError
            raise DatasetNotFoundError(dataset_name, available)
        return cls.DATASETS[dataset_name]
    
    @classmethod
    def get_supported_datasets(cls) -> List[str]:
        """Get list of supported dataset names.""" 
        return list(cls.DATASETS.keys())
    
    @classmethod
    def get_metric_field_name(cls, metric: str) -> str:
        """Get the field name for a metric's average score."""
        return cls.METRIC_FIELD_NAMES.get(metric, f"average_{metric}")
    
    @classmethod
    def validate_dataset_support(cls, dataset: str) -> None:
        """Validate that a dataset is supported."""
        if dataset not in cls.DATASETS:
            available = list(cls.DATASETS.keys()) 
            from .exceptions import DatasetNotFoundError
            raise DatasetNotFoundError(dataset, available)
    
    @classmethod  
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            from .exceptions import ConfigurationError
            raise ConfigurationError(
                "OPENAI_API_KEY", 
                "API key not found in environment variables"
            )
        return api_key