"""
Unified faithfulness evaluation using Ragas Main
Supports both AmnestyQA and FIQA datasets via CLI

Refactored to use shared utilities for better maintainability.
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from ragas import evaluate
from ragas.metrics import faithfulness

from common.cli_utils import create_metric_parser, print_evaluation_summary, get_output_filename, validate_common_args
from common.data_loader import DataLoader
from common.result_saver import ResultSaver
from common.exceptions import EvaluationError

# Load environment variables
load_dotenv()


def load_preprocessed_data_to_ragas(data_file_path: str) -> tuple[Any, str, int]:
    """Load preprocessed data and convert to Ragas EvaluationDataset format"""
    # Use centralized data loader
    data, dataset_name, total_samples = DataLoader.load_from_preprocessed_file(data_file_path)
    
    # Convert to Ragas format using shared utility
    eval_dataset = DataLoader.convert_to_ragas_format(data)
    
    return eval_dataset, dataset_name, total_samples


def evaluate_faithfulness(eval_dataset: Any) -> Any:
    """Evaluate faithfulness on the dataset"""
    logger.info("Starting faithfulness evaluation...")

    try:
        # Run evaluation
        result = evaluate(
            eval_dataset,
            metrics=[faithfulness],
            show_progress=True,
        )
        logger.info("Faithfulness evaluation completed")
        return result
    except Exception as e:
        raise EvaluationError("faithfulness", "ragas_main", str(e))


def save_results(result: Any, dataset_display_name: str, output_path: str):
    """Save evaluation results to file"""
    # Use centralized result saver
    ResultSaver.save_ragas_results(
        result=result,
        metric="faithfulness", 
        dataset_name=dataset_display_name,
        output_path=output_path,
        implementation="ragas_main"
    )


def main():
    """Main execution function"""
    # Use shared CLI utilities
    parser = create_metric_parser("faithfulness", "ragas_main")
    args = parser.parse_args()
    
    # Validate common arguments
    validate_common_args(args)

    logger.info(
        f"Starting {args.dataset.upper()} faithfulness evaluation with Ragas Main"
    )

    try:
        # Load data using shared utility
        eval_dataset, dataset_display_name, total_samples = load_preprocessed_data_to_ragas(
            args.data_file
        )

        # Evaluate faithfulness
        result = evaluate_faithfulness(eval_dataset)

        # Save results using shared utility
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "results" / args.dataset
        results_dir.mkdir(parents=True, exist_ok=True)

        output_filename = get_output_filename(args.dataset, "ragas_main", "faithfulness")
        output_path = results_dir / output_filename
        save_results(result, dataset_display_name, str(output_path))

        # Print summary using shared utility
        result_dict = result.to_pandas() if hasattr(result, "to_pandas") else result
        if "faithfulness" in result_dict:
            average_score = result_dict["faithfulness"].mean()
            print_evaluation_summary(
                dataset_name=dataset_display_name,
                metric="faithfulness",
                implementation="ragas_main",
                average_score=average_score,
                num_successful=total_samples,
                total_samples=total_samples,
                output_path=str(output_path)
            )

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()