"""
Unified faithfulness evaluation using Ragas Main
Supports both AmnestyQA and FIQA datasets via CLI
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.metrics import faithfulness

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_preprocessed_data_to_ragas(data_file_path: str) -> tuple[EvaluationDataset, str, int]:
    """Load preprocessed data and convert to Ragas EvaluationDataset format"""
    logger.info(f"Loading preprocessed data from {data_file_path}")
    
    with open(data_file_path, 'r') as f:
        dataset_info = json.load(f)
    
    data = dataset_info["data"]
    dataset_name = dataset_info["dataset_name"]
    total_samples = dataset_info["total_samples"]
    
    # Convert normalized data format to Ragas format
    eval_data = []
    for sample in data:
        eval_data.append({
            "user_input": sample["question"],
            "response": sample["answer"],
            "retrieved_contexts": [sample["contexts"]]  # Ragas expects a list
        })
    
    # Create EvaluationDataset from list of dicts
    eval_dataset = EvaluationDataset.from_list(eval_data)
    
    logger.info(f"Loaded {total_samples} samples from {dataset_name}")
    return eval_dataset, dataset_name, total_samples


def evaluate_faithfulness(eval_dataset: EvaluationDataset) -> Dict[str, Any]:
    """Evaluate faithfulness on the dataset"""
    logger.info("Starting faithfulness evaluation...")

    # Run evaluation
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness],
        show_progress=True,
    )

    logger.info("Faithfulness evaluation completed")
    return result


def save_results(result, dataset_display_name: str, output_path: str):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")

    # Convert result to serializable format
    result_dict = result.to_pandas() if hasattr(result, "to_pandas") else result

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "ragas_main",
        "metric": "faithfulness",
        "num_samples": len(result_dict)
        if hasattr(result_dict, "__len__")
        else "unknown",
        "average_faithfulness": result_dict["faithfulness"].mean()
        if "faithfulness" in result_dict
        else None,
        "scores": result_dict["faithfulness"].tolist()
        if "faithfulness" in result_dict
        else [],
        "detailed_results": result_dict.to_dict()
        if hasattr(result_dict, "to_dict")
        else str(result),
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Results saved successfully")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness using Ragas Main"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amnesty", "fiqa"],
        required=True,
        help="Dataset to evaluate (amnesty or fiqa)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to preprocessed data JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting {args.dataset.upper()} faithfulness evaluation with Ragas Main"
    )

    try:
        # Load data
        eval_dataset, dataset_display_name, total_samples = load_preprocessed_data_to_ragas(
            args.data_file
        )

        # Evaluate faithfulness
        result = evaluate_faithfulness(eval_dataset)

        # Save results
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{args.dataset}_ragas_main.json"
        output_path = results_dir / output_filename
        save_results(result, dataset_display_name, str(output_path))

        # Print summary
        result_dict = result.to_pandas() if hasattr(result, "to_pandas") else result
        if "faithfulness" in result_dict:
            print(
                f"\n=== {dataset_display_name} Faithfulness Evaluation Results (Ragas Main) ==="
            )
            print(
                f"Average Faithfulness Score: {result_dict['faithfulness'].mean():.4f}"
            )
            print(f"Number of samples: {total_samples}")
            print(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
