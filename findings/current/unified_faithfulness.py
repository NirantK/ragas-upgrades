"""
Unified faithfulness evaluation using Ragas Main
Supports both AmnestyQA and FIQA datasets via CLI
"""
import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.metrics import faithfulness

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_by_name(dataset_name: str, num_samples: int = 10, random_seed: int = 42) -> EvaluationDataset:
    """Load dataset by name and convert to Ragas format"""
    logger.info(f"Loading {dataset_name} dataset with {num_samples} samples")
    
    if dataset_name.lower() == "amnesty":
        # Load AmnestyQA dataset
        hf_dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3", trust_remote_code=True)["eval"]
        dataset_display_name = "AmnestyQA"
    elif dataset_name.lower() == "fiqa":
        # Load FIQA dataset
        hf_dataset = load_dataset("explodinggradients/fiqa", "ragas_eval", trust_remote_code=True)["baseline"]
        dataset_display_name = "FIQA"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: amnesty, fiqa")
    
    # Select samples with consistent seed
    selected_samples = hf_dataset.shuffle(seed=random_seed).select(range(num_samples))
    
    # Map column names for FIQA dataset
    if dataset_name.lower() == "fiqa":
        # FIQA uses 'question', 'answer', 'contexts' instead of 'user_input', 'response', 'retrieved_contexts'
        selected_samples = selected_samples.rename_columns({
            "question": "user_input",
            "answer": "response", 
            "contexts": "retrieved_contexts"
        })
    
    # Convert to Ragas EvaluationDataset
    eval_dataset = EvaluationDataset.from_hf_dataset(selected_samples)
    
    logger.info(f"Loaded {len(eval_dataset)} samples from {dataset_display_name}")
    return eval_dataset, dataset_display_name


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
    result_dict = result.to_pandas() if hasattr(result, 'to_pandas') else result
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "ragas_main",
        "metric": "faithfulness",
        "num_samples": len(result_dict) if hasattr(result_dict, '__len__') else "unknown",
        "average_faithfulness": result_dict["faithfulness"].mean() if "faithfulness" in result_dict else None,
        "scores": result_dict["faithfulness"].tolist() if "faithfulness" in result_dict else [],
        "detailed_results": result_dict.to_dict() if hasattr(result_dict, 'to_dict') else str(result)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved successfully")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Evaluate faithfulness using Ragas Main")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["amnesty", "fiqa"],
        required=True,
        help="Dataset to evaluate (amnesty or fiqa)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Output directory for results (default: results)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.dataset.upper()} faithfulness evaluation with Ragas Main")
    
    try:
        # Load data
        eval_dataset, dataset_display_name = load_dataset_by_name(
            args.dataset, 
            num_samples=args.samples,
            random_seed=args.seed
        )
        
        # Evaluate faithfulness
        result = evaluate_faithfulness(eval_dataset)
        
        # Save results
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / args.output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{args.dataset}_ragas_main.json"
        output_path = results_dir / output_filename
        save_results(result, dataset_display_name, str(output_path))
        
        # Print summary
        result_dict = result.to_pandas() if hasattr(result, 'to_pandas') else result
        if "faithfulness" in result_dict:
            print(f"\n=== {dataset_display_name} Faithfulness Evaluation Results (Ragas Main) ===")
            print(f"Average Faithfulness Score: {result_dict['faithfulness'].mean():.4f}")
            print(f"Number of samples: {len(result_dict)}")
            print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()