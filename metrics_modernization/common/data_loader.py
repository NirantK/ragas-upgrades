"""
Centralized data loading utilities for metrics modernization.

This module consolidates all dataset loading logic that was duplicated across
implementation files, providing a single source of truth for data handling.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from loguru import logger
from ragas import EvaluationDataset

from .config import Config
from .exceptions import DataLoadError, DataValidationError, DatasetNotFoundError


class DataLoader:
    """Centralized data loading and processing."""
    
    @staticmethod
    def load_dataset_from_hf(
        dataset_name: str, 
        num_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        """Load dataset from HuggingFace and normalize format."""
        
        # Validate dataset is supported
        Config.validate_dataset_support(dataset_name)
        dataset_config = Config.get_dataset_config(dataset_name)
        
        logger.info(f"Loading {dataset_config.display_name} dataset from HuggingFace")
        
        try:
            # Load from HuggingFace
            if dataset_config.hf_config:
                hf_dataset = load_dataset(
                    dataset_config.hf_name, 
                    dataset_config.hf_config
                )[dataset_config.split]
            else:
                hf_dataset = load_dataset(dataset_config.hf_name)[dataset_config.split]
                
            # Sample if requested
            if num_samples is not None:
                max_samples = min(num_samples, len(hf_dataset))
                hf_dataset = hf_dataset.shuffle(seed=random_seed).select(range(max_samples))
                logger.info(f"Sampled {max_samples}/{len(hf_dataset)} samples")
            else:
                logger.info(f"Using complete dataset: {len(hf_dataset)} samples")
                
        except Exception as e:
            raise DataLoadError(f"HuggingFace:{dataset_config.hf_name}", str(e))
        
        # Normalize data format
        normalized_data = []
        for sample in hf_dataset:
            try:
                normalized_sample = DataLoader._normalize_sample(sample, dataset_config)
                normalized_data.append(normalized_sample)
            except Exception as e:
                logger.warning(f"Failed to normalize sample, skipping: {e}")
                continue
        
        if not normalized_data:
            raise DataLoadError(
                dataset_config.hf_name,
                "No valid samples found after normalization"
            )
        
        return normalized_data, dataset_config.display_name, len(normalized_data)
    
    @staticmethod
    def load_from_preprocessed_file(data_file_path: str) -> Tuple[List[Dict[str, Any]], str, int]:
        """Load data from preprocessed JSON file."""
        
        logger.info(f"Loading preprocessed data from {data_file_path}")
        
        if not Path(data_file_path).exists():
            raise DataLoadError(data_file_path, "File does not exist")
            
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoadError(data_file_path, f"Invalid JSON format: {e}")
        except Exception as e:
            raise DataLoadError(data_file_path, f"File read error: {e}")
        
        # Validate expected structure
        required_keys = ["data", "dataset_name", "total_samples"]
        for key in required_keys:
            if key not in dataset_info:
                raise DataValidationError(key, "required in preprocessed data file")
        
        data = dataset_info["data"]
        dataset_name = dataset_info["dataset_name"]
        total_samples = dataset_info["total_samples"]
        
        # Validate data samples
        DataLoader._validate_data_samples(data)
        
        logger.info(f"Loaded {total_samples} samples from {dataset_name}")
        return data, dataset_name, total_samples
    
    @staticmethod
    def convert_to_ragas_format(data: List[Dict[str, Any]]) -> EvaluationDataset:
        """Convert normalized data to Ragas EvaluationDataset format."""
        
        eval_data = []
        for sample in data:
            ragas_sample = {
                "user_input": sample["question"],
                "response": sample["answer"]
            }
            
            # Add contexts if available
            if "contexts" in sample and sample["contexts"]:
                # Ensure contexts is a list for Ragas
                contexts = sample["contexts"]
                if isinstance(contexts, str):
                    ragas_sample["retrieved_contexts"] = [contexts]
                elif isinstance(contexts, list):
                    ragas_sample["retrieved_contexts"] = contexts
                else:
                    ragas_sample["retrieved_contexts"] = [str(contexts)]
            
            # Add ground truth if available  
            if "ground_truth" in sample and sample["ground_truth"]:
                ragas_sample["reference"] = sample["ground_truth"]
                
            eval_data.append(ragas_sample)
        
        return EvaluationDataset.from_list(eval_data)
    
    @staticmethod
    def _normalize_sample(sample: Dict[str, Any], dataset_config) -> Dict[str, Any]:
        """Normalize a single sample to standard format."""
        
        normalized = {}
        
        # Extract question
        if dataset_config.question_field in sample:
            normalized["question"] = str(sample[dataset_config.question_field])
        else:
            raise DataValidationError(
                dataset_config.question_field,
                "question field",
                f"not found in sample keys: {list(sample.keys())}"
            )
        
        # Extract answer
        if dataset_config.answer_field in sample:
            normalized["answer"] = str(sample[dataset_config.answer_field])
        else:
            raise DataValidationError(
                dataset_config.answer_field,
                "answer field", 
                f"not found in sample keys: {list(sample.keys())}"
            )
        
        # Extract ground truth (optional)
        if dataset_config.ground_truth_field in sample:
            ground_truth = sample[dataset_config.ground_truth_field]
            if isinstance(ground_truth, list) and ground_truth:
                normalized["ground_truth"] = str(ground_truth[0])
            else:
                normalized["ground_truth"] = str(ground_truth) if ground_truth else ""
        
        # Extract contexts (optional)
        if dataset_config.contexts_field in sample:
            contexts = sample[dataset_config.contexts_field]
            if isinstance(contexts, list):
                normalized["contexts"] = "\n".join(str(c) for c in contexts if c)
            else:
                normalized["contexts"] = str(contexts) if contexts else ""
        
        return normalized
    
    @staticmethod
    def _validate_data_samples(data: List[Dict[str, Any]]) -> None:
        """Validate that data samples have required fields."""
        
        if not data:
            raise DataValidationError("data", "non-empty list")
        
        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise DataValidationError(f"data[{i}]", "dictionary")
            
            for field in Config.REQUIRED_DATA_FIELDS:
                if field not in sample:
                    raise DataValidationError(
                        f"data[{i}].{field}",
                        f"required field in sample {i}"
                    )
                    
                if not sample[field]:
                    raise DataValidationError(
                        f"data[{i}].{field}",
                        f"non-empty value in sample {i}"
                    )