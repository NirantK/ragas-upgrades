"""
Unified result saving utilities for metrics modernization.

This module consolidates all result saving logic that was duplicated across
implementation files, providing standardized result format and validation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .config import Config
from .exceptions import ResultSaveError, DataValidationError


class ResultSaver:
    """Centralized result saving and validation."""
    
    @staticmethod
    def save_ragas_results(
        result: Any,
        metric: str,
        dataset_name: str,
        output_path: str,
        implementation: str = "ragas_main"
    ) -> None:
        """Save results from Ragas evaluation."""
        
        logger.info(f"Saving Ragas {metric} results to {output_path}")
        
        try:
            # Convert Ragas result to pandas DataFrame if possible
            result_dict = result.to_pandas() if hasattr(result, "to_pandas") else result
            
            # Extract metric-specific data
            metric_field = ResultSaver._get_ragas_metric_field(metric, result_dict)
            
            if metric_field in result_dict:
                scores = result_dict[metric_field].tolist()
                average_score = result_dict[metric_field].mean()
                num_samples = len(result_dict)
            else:
                raise DataValidationError(
                    metric_field,
                    f"metric field in Ragas results for {metric}"
                )
            
            # Create standardized results structure
            results_data = ResultSaver._create_standard_results(
                metric=metric,
                dataset_name=dataset_name,
                implementation=implementation,
                num_samples=num_samples,
                average_score=average_score,
                scores=scores,
                detailed_results=result_dict.to_dict() if hasattr(result_dict, "to_dict") else str(result)
            )
            
            ResultSaver._write_results_file(results_data, output_path)
            
        except Exception as e:
            raise ResultSaveError(output_path, f"Failed to process Ragas results: {e}")
    
    @staticmethod
    def save_modern_results(
        results: List[Dict[str, Any]],
        metric: str,
        dataset_name: str,
        output_path: str,
        implementation: str
    ) -> None:
        """Save results from modern implementation."""
        
        logger.info(f"Saving modern {metric} results to {output_path}")
        
        try:
            # Extract valid scores
            valid_results = [
                r for r in results 
                if r.get("success", False) and r.get(f"{metric}_score") is not None
            ]
            
            if not valid_results:
                raise DataValidationError(
                    "results",
                    "at least one successful evaluation result"
                )
            
            scores = [r[f"{metric}_score"] for r in valid_results]
            average_score = sum(scores) / len(scores)
            
            # Create standardized results structure
            results_data = ResultSaver._create_standard_results(
                metric=metric,
                dataset_name=dataset_name,
                implementation=implementation,
                num_samples=len(results),
                num_successful=len(valid_results),
                average_score=average_score,
                scores=scores,
                detailed_results=results
            )
            
            ResultSaver._write_results_file(results_data, output_path)
            
        except Exception as e:
            raise ResultSaveError(output_path, f"Failed to process modern results: {e}")
    
    @staticmethod
    def save_generic_results(
        metric: str,
        dataset_name: str,
        implementation: str,
        output_path: str,
        scores: List[float],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save results in generic format."""
        
        logger.info(f"Saving generic {metric} results to {output_path}")
        
        try:
            if not scores:
                raise DataValidationError("scores", "non-empty list of scores")
            
            average_score = sum(scores) / len(scores)
            
            # Create standardized results structure
            results_data = ResultSaver._create_standard_results(
                metric=metric,
                dataset_name=dataset_name,
                implementation=implementation,
                num_samples=len(scores),
                average_score=average_score,
                scores=scores,
                detailed_results=additional_data or {}
            )
            
            ResultSaver._write_results_file(results_data, output_path)
            
        except Exception as e:
            raise ResultSaveError(output_path, f"Failed to save generic results: {e}")
    
    @staticmethod
    def _create_standard_results(
        metric: str,
        dataset_name: str,
        implementation: str,
        num_samples: int,
        average_score: float,
        scores: List[float],
        detailed_results: Any,
        num_successful: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create standardized results structure."""
        
        # Get the standardized average field name
        average_field_name = Config.get_metric_field_name(metric)
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "framework": implementation,
            "metric": metric,
            "num_samples": num_samples,
            average_field_name: average_score,
            "scores": scores,
            "detailed_results": detailed_results
        }
        
        # Add num_successful if different from num_samples
        if num_successful is not None and num_successful != num_samples:
            results_data["num_successful"] = num_successful
        
        return results_data
    
    @staticmethod
    def _get_ragas_metric_field(metric: str, result_dict: Any) -> str:
        """Get the field name for a metric in Ragas results."""
        
        # Common Ragas metric field mappings
        ragas_field_mapping = {
            "faithfulness": "faithfulness",
            "answer_relevance": "answer_relevancy",  # Note the spelling difference
            "answer_correctness": "answer_correctness",
            "context_recall": "context_recall",
            "context_precision": "context_precision"
        }
        
        return ragas_field_mapping.get(metric, metric)
    
    @staticmethod
    def _write_results_file(results_data: Dict[str, Any], output_path: str) -> None:
        """Write results data to file with proper error handling."""
        
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write results with pretty formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved successfully to {output_path}")
            
        except PermissionError:
            raise ResultSaveError(output_path, "Permission denied")
        except OSError as e:
            raise ResultSaveError(output_path, f"File system error: {e}")
        except Exception as e:
            raise ResultSaveError(output_path, f"Unexpected error: {e}")
    
    @staticmethod
    def load_results(file_path: str) -> Dict[str, Any]:
        """Load results from a saved file."""
        
        if not Path(file_path).exists():
            from .exceptions import DataLoadError
            raise DataLoadError(file_path, "Results file does not exist")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return results
        except json.JSONDecodeError as e:
            from .exceptions import DataLoadError
            raise DataLoadError(file_path, f"Invalid JSON format: {e}")
        except Exception as e:
            from .exceptions import DataLoadError
            raise DataLoadError(file_path, f"File read error: {e}")
    
    @staticmethod
    def validate_results_structure(results: Dict[str, Any], metric: str) -> None:
        """Validate that results have expected structure."""
        
        required_fields = [
            "timestamp", "dataset", "framework", "metric", 
            "num_samples", "scores"
        ]
        
        for field in required_fields:
            if field not in results:
                raise DataValidationError(field, f"required in {metric} results")
        
        # Check metric-specific average field
        average_field = Config.get_metric_field_name(metric)
        if average_field not in results:
            raise DataValidationError(average_field, f"required average score field for {metric}")
        
        # Validate scores is a list of numbers
        scores = results.get("scores", [])
        if not isinstance(scores, list) or not all(isinstance(s, (int, float)) for s in scores):
            raise DataValidationError("scores", "list of numbers")