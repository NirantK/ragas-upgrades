"""
Metrics Registry System for Auto-Discovery of Implementations

This module provides a registry system that automatically discovers and manages
metric implementations across the metrics_modernization directory structure.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from common.config import Config
from common.exceptions import (
    ImplementationNotFoundError,
    UnsupportedDatasetError, 
    DataValidationError
)


class ImplementationInfo(BaseModel):
    """Information about a metric implementation"""

    name: str
    metric: str
    file_path: str
    framework: str
    description: str
    supports_datasets: List[str]
    main_function: str = "main"


class MetricsRegistry:
    """Registry for discovering and managing metric implementations"""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the registry with optional base path"""
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.implementations: Dict[str, List[ImplementationInfo]] = {}
        self.discover_implementations()

    def discover_implementations(self):
        """Discover all metric implementations in the directory structure"""
        logger.info(f"Discovering implementations in {self.base_path}")

        # Look for metric directories
        for metric_dir in self.base_path.iterdir():
            if not metric_dir.is_dir() or metric_dir.name.startswith("."):
                continue

            implementations_dir = metric_dir / "implementations"
            if not implementations_dir.exists():
                continue

            metric_name = metric_dir.name
            self.implementations[metric_name] = []

            # Discover Python files in implementations directory
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                impl_info = self._analyze_implementation(py_file, metric_name)
                if impl_info:
                    self.implementations[metric_name].append(impl_info)

        logger.info(
            f"Discovered implementations for metrics: {list(self.implementations.keys())}"
        )

    def _analyze_implementation(
        self, file_path: Path, metric: str
    ) -> Optional[ImplementationInfo]:
        """Analyze a Python file to extract implementation information"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Extract basic information
            name = file_path.stem
            description = self._extract_description(content)
            framework = self._extract_framework(content, name)
            supports_datasets = self._extract_supported_datasets(content)

            return ImplementationInfo(
                name=name,
                metric=metric,
                file_path=str(file_path),
                framework=framework,
                description=description,
                supports_datasets=supports_datasets,
            )

        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return None

    def _extract_description(self, content: str) -> str:
        """Extract description from file docstring or comments"""
        lines = content.split("\n")

        # Look for triple-quoted docstring at top
        in_docstring = False
        description_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') and not in_docstring:
                in_docstring = True
                desc = stripped[3:]
                if desc.endswith('"""'):
                    return desc[:-3].strip()
                if desc:
                    description_lines.append(desc)
            elif in_docstring:
                if stripped.endswith('"""'):
                    desc = stripped[:-3].strip()
                    if desc:
                        description_lines.append(desc)
                    break
                description_lines.append(stripped)

        return (
            " ".join(description_lines)
            if description_lines
            else f"Implementation in {content[:50]}..."
        )

    def _extract_framework(self, content: str, name: str) -> str:
        """Extract framework information from content"""
        if "ragas" in content.lower() and "import" in content:
            if "from ragas import" in content or "import ragas" in content:
                return "ragas_main"

        if "openai" in content.lower() and "asyncopenai" in content:
            if "exact_replica" in name:
                return "modern_exact_replica"
            return "modern_simplified"

        return "unknown"

    def _extract_supported_datasets(self, content: str) -> List[str]:
        """Extract supported datasets from content"""
        datasets = []
        if "amnesty" in content.lower():
            datasets.append("amnesty")
        if "fiqa" in content.lower():
            datasets.append("fiqa")
        
        # Require explicit dataset declarations - no silent fallbacks
        if not datasets:
            raise DataValidationError(
                "supported_datasets",
                "explicit dataset support declarations in implementation file"
            )
        return datasets

    def get_implementations(self, metric: str) -> List[ImplementationInfo]:
        """Get all implementations for a specific metric"""
        return self.implementations.get(metric, [])

    def get_all_metrics(self) -> List[str]:
        """Get list of all available metrics"""
        return list(self.implementations.keys())

    def get_implementation_by_name(
        self, metric: str, name: str
    ) -> ImplementationInfo:
        """Get specific implementation by metric and name"""
        implementations = self.get_implementations(metric)
        for impl in implementations:
            if impl.name == name:
                return impl
        
        # Fail fast instead of returning None
        available = [impl.name for impl in implementations]
        raise ImplementationNotFoundError(metric, name, available)

    def list_implementations(
        self, metric: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """List all implementations with detailed information"""
        result = {}

        metrics_to_process = [metric] if metric else self.get_all_metrics()

        for m in metrics_to_process:
            implementations = self.get_implementations(m)
            result[m] = [
                {
                    "name": impl.name,
                    "framework": impl.framework,
                    "description": impl.description,
                    "supports_datasets": impl.supports_datasets,
                    "file_path": impl.file_path,
                }
                for impl in implementations
            ]

        return result

    def save_registry_info(self, output_path: str):
        """Save registry information to JSON file"""
        registry_data = {
            "timestamp": datetime.now().isoformat(),
            "base_path": str(self.base_path),
            "metrics": self.list_implementations(),
        }

        with open(output_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Registry information saved to {output_path}")
    
    def validate_implementation_dataset_support(
        self, metric: str, implementation_name: str, dataset: str
    ) -> None:
        """Validate that an implementation supports a dataset."""
        
        # Get the implementation
        impl = self.get_implementation_by_name(metric, implementation_name)
        
        # Check if dataset is supported
        if dataset not in impl.supports_datasets:
            raise UnsupportedDatasetError(
                implementation_name, 
                dataset, 
                impl.supports_datasets
            )
    
    def get_compatible_implementations(
        self, metric: str, dataset: str
    ) -> List[ImplementationInfo]:
        """Get all implementations that support a specific dataset."""
        
        implementations = self.get_implementations(metric)
        compatible = [
            impl for impl in implementations 
            if dataset in impl.supports_datasets
        ]
        
        if not compatible:
            available_datasets = set()
            for impl in implementations:
                available_datasets.update(impl.supports_datasets)
            
            raise UnsupportedDatasetError(
                f"any implementation for {metric}",
                dataset,
                list(available_datasets)
            )
        
        return compatible


def create_registry(base_path: Optional[str] = None) -> MetricsRegistry:
    """Factory function to create a metrics registry"""
    return MetricsRegistry(base_path)


if __name__ == "__main__":
    # Demo usage
    registry = create_registry()

    print("=== Metrics Registry ===")
    print(f"Available metrics: {registry.get_all_metrics()}")

    implementations = registry.list_implementations()
    for metric, impls in implementations.items():
        print(f"\n{metric.upper()} implementations:")
        for impl in impls:
            print(f"  - {impl['name']} ({impl['framework']})")
            print(f"    Description: {impl['description'][:100]}...")
            print(f"    Datasets: {', '.join(impl['supports_datasets'])}")
