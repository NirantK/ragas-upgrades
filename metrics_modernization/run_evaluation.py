"""
Unified Evaluation Driver for Metrics Modernization

This script provides a unified interface to run all metric implementations
across different datasets with support for concurrent execution and
standardized result output.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from datasets import load_dataset
from loguru import logger
from metrics_registry import ImplementationInfo, MetricsRegistry
from pydantic import BaseModel

from common.config import Config
from common.exceptions import (
    EvaluationError, 
    ResultSaveError, 
    DataLoadError,
    UnsupportedDatasetError
)


def load_dataset_by_name(
    dataset_name: str, 
    num_samples: Optional[int] = None, 
    random_seed: int = 42
) -> Dict[str, Any]:
    """Centralized dataset loading with optional sampling
    
    Args:
        dataset_name: Name of dataset to load ('amnesty' or 'fiqa')
        num_samples: Number of samples to load. If None, loads full dataset
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Dictionary containing processed data, dataset display name, and sample count
    """
    logger.info(f"Loading {dataset_name} dataset")
    
    if dataset_name.lower() == "amnesty":
        hf_dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")["eval"]
        dataset_display_name = "AmnestyQA"
    elif dataset_name.lower() == "fiqa":
        hf_dataset = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]
        dataset_display_name = "FIQA"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: amnesty, fiqa")
    
    # Use full dataset by default, sample only if specified
    if num_samples is None:
        selected_samples = hf_dataset
        logger.info(f"Loading complete {dataset_display_name} dataset: {len(selected_samples)} samples")
    else:
        max_samples = min(num_samples, len(hf_dataset))
        selected_samples = hf_dataset.shuffle(seed=random_seed).select(range(max_samples))
        logger.info(f"Loading {dataset_display_name} dataset: {max_samples}/{len(hf_dataset)} samples")
    
    # Normalize data format for all implementations
    data = []
    for sample in selected_samples:
        if dataset_name.lower() == "fiqa":
            data.append({
                "question": sample["question"],
                "answer": sample["answer"],
                "ground_truth": sample["ground_truths"][0] if sample["ground_truths"] else "",
                "contexts": "\n".join(sample["contexts"])
            })
        else:  # amnesty
            data.append({
                "question": sample["user_input"],
                "answer": sample["response"],
                "ground_truth": sample["reference"],
                "contexts": "\n".join(sample["retrieved_contexts"])
            })
    
    return {
        "data": data,
        "dataset_name": dataset_display_name,
        "total_samples": len(selected_samples)
    }


class EvaluationJob(BaseModel):
    """Represents a single evaluation job"""

    metric: str
    implementation: str
    dataset: str
    samples: Optional[int]
    seed: int
    output_dir: str
    implementation_info: ImplementationInfo


class EvaluationResult(BaseModel):
    """Results from an evaluation job"""

    job: EvaluationJob
    success: bool
    output_file: Optional[str] = None
    average_score: Optional[float] = None
    num_samples: Optional[int] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


class UnifiedEvaluationDriver:
    """Main driver for unified metric evaluations"""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the driver"""
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.registry = MetricsRegistry(str(self.base_path))
        self.results_dir = self.base_path / "results"

    def create_evaluation_jobs(
        self,
        metrics: List[str],
        implementations: Optional[List[str]] = None,
        datasets: List[str] = ["amnesty", "fiqa"],
        samples: Optional[int] = None,
        seed: int = 42,
    ) -> List[EvaluationJob]:
        """Create evaluation jobs for specified metrics and implementations"""
        jobs = []

        for metric in metrics:
            metric_implementations = self.registry.get_implementations(metric)

            if not metric_implementations:
                logger.warning(f"No implementations found for metric: {metric}")
                continue

            for impl_info in metric_implementations:
                # Filter by implementation names if specified
                if implementations and impl_info.name not in implementations:
                    continue

                for dataset in datasets:
                    # Check if implementation supports this dataset - fail fast
                    if dataset not in impl_info.supports_datasets:
                        raise UnsupportedDatasetError(
                            impl_info.name, 
                            dataset, 
                            impl_info.supports_datasets
                        )

                    # Create output directory structure
                    output_dir = self.results_dir / metric / dataset

                    job = EvaluationJob(
                        metric=metric,
                        implementation=impl_info.name,
                        dataset=dataset,
                        samples=samples,
                        seed=seed,
                        output_dir=str(output_dir),
                        implementation_info=impl_info,
                    )
                    jobs.append(job)

        return jobs

    def execute_job(self, job: EvaluationJob) -> EvaluationResult:
        """Execute a single evaluation job"""
        start_time = datetime.now()

        try:
            samples_text = " (full dataset)" if job.samples is None else f" ({job.samples} samples)"
            logger.info(
                f"Executing {job.metric}/{job.implementation} on {job.dataset}"
                f"{samples_text}, seed={job.seed})"
            )

            # Ensure output directory exists
            Path(job.output_dir).mkdir(parents=True, exist_ok=True)

            # Load data centrally using the centralized loader
            dataset_info = load_dataset_by_name(
                job.dataset,
                num_samples=job.samples,
                random_seed=job.seed
            )
            
            # Write temporary data file
            temp_data_file = Path(job.output_dir) / f"{job.dataset}_data.json"
            with open(temp_data_file, 'w') as f:
                json.dump(dataset_info, f)

            # Build command to run the implementation with data file
            cmd = [
                sys.executable,  # Use current Python interpreter
                job.implementation_info.file_path,
                "--dataset",
                job.dataset,
                "--data-file",
                str(temp_data_file),
                "--output-dir",
                job.output_dir,
            ]

            # Execute the command
            env = os.environ.copy()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,  # 10 minute timeout
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            if result.returncode != 0:
                error_msg = f"Command failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"

                return EvaluationResult(
                    job=job,
                    success=False,
                    execution_time=execution_time,
                    error_message=error_msg,
                )

            # Try to find and parse the output file
            output_file = self._find_output_file(job)
            average_score, num_samples = self._parse_results(output_file)

            return EvaluationResult(
                job=job,
                success=True,
                output_file=output_file,
                average_score=average_score,
                num_samples=num_samples,
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired:
            return EvaluationResult(
                job=job,
                success=False,
                execution_time=600,
                error_message="Execution timed out after 10 minutes",
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return EvaluationResult(
                job=job,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    def _find_output_file(self, job: EvaluationJob) -> Optional[str]:
        """Find the output file for a job"""
        output_dir = Path(job.output_dir)

        # Look for JSON files matching expected patterns
        patterns = [
            f"{job.dataset}_*.json",
            f"*_{job.implementation}.json",
            f"{job.dataset}*.json",
            "*.json",
        ]

        for pattern in patterns:
            files = list(output_dir.glob(pattern))
            if files:
                # Return the most recently modified file
                return str(max(files, key=lambda f: f.stat().st_mtime))

        return None

    def _parse_results(
        self, output_file: Optional[str]
    ) -> Tuple[Optional[float], Optional[int]]:
        """Parse results from output file"""
        if not output_file or not Path(output_file).exists():
            return None, None

        try:
            with open(output_file, "r") as f:
                data = json.load(f)

            # Get metric-specific field name from config
            metric_name = job.metric if hasattr(job, 'metric') else "unknown"
            expected_field = Config.get_metric_field_name(metric_name)
            
            # Try expected field first, then common alternatives
            score_fields = [
                expected_field,
                "average_score",
                "mean_score",
                "avg_score",
            ]
            
            average_score = None
            for field in score_fields:
                if field in data:
                    average_score = data[field]
                    break
            
            if average_score is None:
                raise DataLoadError(
                    output_file, 
                    f"No valid score field found. Expected: {score_fields}"
                )

            # Get sample count with validation
            num_samples = data.get("num_samples") or data.get("sample_count")
            if num_samples is None:
                raise DataLoadError(
                    output_file,
                    "No valid sample count field found (num_samples or sample_count)"
                )

            return average_score, num_samples

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            raise DataLoadError(output_file, f"Could not parse results: {e}")
        except Exception as e:
            raise DataLoadError(output_file, f"Unexpected error parsing results: {e}")

    def execute_jobs_parallel(
        self, jobs: List[EvaluationJob], max_workers: int = 4
    ) -> List[EvaluationResult]:
        """Execute multiple jobs in parallel"""
        results = []

        logger.info(f"Executing {len(jobs)} jobs with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.execute_job, job): job for job in jobs
            }

            # Collect results as they complete
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)

                if result.success:
                    logger.info(
                        f"✅ {result.job.metric}/{result.job.implementation}/{result.job.dataset} "
                        f"completed in {result.execution_time:.1f}s"
                    )
                else:
                    logger.error(
                        f"❌ {result.job.metric}/{result.job.implementation}/{result.job.dataset} "
                        f"failed: {result.error_message}"
                    )

        return results

    def save_summary_report(self, results: List[EvaluationResult], output_path: str):
        """Save a summary report of all evaluation results"""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_jobs": len(results),
            "successful_jobs": sum(1 for r in results if r.success),
            "failed_jobs": sum(1 for r in results if not r.success),
            "results": [],
        }

        for result in results:
            job_data = {
                "metric": result.job.metric,
                "implementation": result.job.implementation,
                "dataset": result.job.dataset,
                "samples": result.job.samples,
                "seed": result.job.seed,
                "success": result.success,
                "execution_time": result.execution_time,
                "output_file": result.output_file,
                "average_score": result.average_score,
                "num_samples": result.num_samples,
                "error_message": result.error_message,
            }
            report_data["results"].append(job_data)

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Summary report saved to {output_path}")

    def print_results_summary(self, results: List[EvaluationResult]):
        """Print a summary of results to console"""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"Total Jobs: {len(results)}")
        print(f"Successful: {len(successful)} ✅")
        print(f"Failed: {len(failed)} ❌")

        if successful:
            print("\nSUCCESSFUL EVALUATIONS:")
            print("-" * 80)
            print(
                f"{'Metric':<15} {'Implementation':<20} {'Dataset':<10} {'Score':<8} {'Samples':<8} {'Time(s)':<8}"
            )
            print("-" * 80)

            for result in successful:
                score_str = (
                    f"{result.average_score:.4f}"
                    if result.average_score is not None
                    else "N/A"
                )
                samples_str = (
                    str(result.num_samples) if result.num_samples is not None else "N/A"
                )
                time_str = (
                    f"{result.execution_time:.1f}"
                    if result.execution_time is not None
                    else "N/A"
                )

                print(
                    f"{result.job.metric:<15} {result.job.implementation:<20} "
                    f"{result.job.dataset:<10} {score_str:<8} {samples_str:<8} {time_str:<8}"
                )

        if failed:
            print("\nFAILED EVALUATIONS:")
            print("-" * 80)
            for result in failed:
                print(
                    f"❌ {result.job.metric}/{result.job.implementation}/{result.job.dataset}"
                )
                if result.error_message:
                    error_line = result.error_message.split("\n")[0]  # First line only
                    print(f"   Error: {error_line}")


@click.command()
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    help="Metrics to evaluate (e.g., faithfulness, answer_relevancy). Use 'all' for all available metrics.",
)
@click.option(
    "--implementations",
    "-i",
    multiple=True,
    help="Specific implementations to run (optional). If not specified, runs all implementations.",
)
@click.option(
    "--datasets",
    "-d",
    multiple=True,
    type=click.Choice(["amnesty", "fiqa"]),
    default=["amnesty", "fiqa"],
    help="Datasets to evaluate on (default: amnesty fiqa)",
)
@click.option(
    "--limit-samples",
    type=int,
    help="Limit number of samples for testing (optional). Uses full dataset by default.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible sampling (default: 42)",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="Number of parallel workers (default: 4)",
)
@click.option(
    "--list-implementations",
    "-l",
    is_flag=True,
    help="List all available implementations and exit",
)
@click.option(
    "--output-summary",
    "-o",
    type=click.Path(),
    help="Path to save summary report (optional)",
)
def main(
    metrics: tuple[str, ...],
    implementations: tuple[str, ...],
    datasets: tuple[str, ...],
    limit_samples: Optional[int],
    seed: int,
    workers: int,
    list_implementations: bool,
    output_summary: Optional[str],
) -> None:
    """Unified Metrics Evaluation Driver

    Run metric implementations across datasets with concurrent execution.
    """
    # Initialize driver
    driver = UnifiedEvaluationDriver()

    # Handle list implementations
    if list_implementations:
        implementations_dict = driver.registry.list_implementations()
        click.echo("\n=== AVAILABLE IMPLEMENTATIONS ===")
        for metric, impls in implementations_dict.items():
            click.echo(f"\n{metric.upper()}:")
            for impl in impls:
                click.echo(f"  • {impl['name']} ({impl['framework']})")
                click.echo(f"    {impl['description'][:80]}...")
                click.echo(f"    Datasets: {', '.join(impl['supports_datasets'])}")
        return

    # Validate metrics argument
    if not metrics:
        click.echo("Error: --metrics argument is required", err=True)
        raise click.ClickException("No metrics specified")

    # Handle 'all' metrics
    metrics_list = list(metrics)
    if metrics_list == ["all"]:
        metrics_list = driver.registry.get_all_metrics()

    # Validate metrics exist
    available_metrics = driver.registry.get_all_metrics()
    invalid_metrics = [m for m in metrics_list if m not in available_metrics]
    if invalid_metrics:
        click.echo(f"Error: Unknown metrics: {invalid_metrics}", err=True)
        click.echo(f"Available metrics: {available_metrics}")
        raise click.ClickException("Invalid metrics specified")

    # Create evaluation jobs
    jobs = driver.create_evaluation_jobs(
        metrics=metrics_list,
        implementations=list(implementations) if implementations else None,
        datasets=list(datasets),
        samples=limit_samples,
        seed=seed,
    )

    if not jobs:
        click.echo(
            "No evaluation jobs created. Check your metric and implementation arguments.",
            err=True,
        )
        return

    click.echo(f"\nCreated {len(jobs)} evaluation jobs")

    # Execute jobs
    results = driver.execute_jobs_parallel(jobs, max_workers=workers)

    # Print results summary
    driver.print_results_summary(results)

    # Save summary report if requested
    if output_summary:
        driver.save_summary_report(results, output_summary)


if __name__ == "__main__":
    main()
