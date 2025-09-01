"""
Simplified context precision evaluation using direct OpenAI calls (modern simplified approach)
Supports both AmnestyQA and FIQA datasets via CLI
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextPrecisionResponse(BaseModel):
    """Response model for context precision evaluation"""

    value: float = Field(
        ..., description="Context precision score between 0 and 1", ge=0.0, le=1.0
    )
    reason: str = Field(..., description="Reasoning for the context precision score")


def load_preprocessed_data(data_file_path: str) -> tuple[List[Dict[str, Any]], str, int]:
    """Load preprocessed data from centralized data file"""
    logger.info(f"Loading preprocessed data from {data_file_path}")
    
    with open(data_file_path, 'r') as f:
        dataset_info = json.load(f)
    
    data = dataset_info["data"]
    dataset_name = dataset_info["dataset_name"]
    total_samples = dataset_info["total_samples"]
    
    logger.info(f"Loaded {total_samples} samples from {dataset_name}")
    return data, dataset_name, total_samples


async def evaluate_sample_context_precision(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = "gpt-5-mini-2025-08-07"
) -> Dict[str, Any]:
    """Evaluate context precision for a single sample using OpenAI"""

    prompt = f"""
Given a question, answer, and retrieved contexts, evaluate the precision of the contexts in relation to answering the question.

Context precision measures how relevant and useful the retrieved contexts are for answering the given question. A high context precision score means the contexts contain mostly relevant information with little irrelevant content.

Question: {sample["question"]}
Answer: {sample["answer"]}
Retrieved Contexts: {sample["contexts"]}

Please evaluate the context precision on a scale from 0 to 1, where:
- 0: The contexts are completely irrelevant and not useful for answering the question
- 1: The contexts are highly relevant and contain only useful information for answering the question

Provide your evaluation as a JSON object with 'value' (float between 0 and 1) and 'reason' (string explanation).
"""

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=ContextPrecisionResponse,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return {
        "sample": sample,
        "context_precision_score": result.value,
        "reasoning": result.reason,
        "success": True,
    }


async def evaluate_context_precision_modern_simplified(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = "gpt-5-mini-2025-08-07"
) -> List[Dict[str, Any]]:
    """Evaluate context precision using modern simplified approach"""
    logger.info("Starting context precision evaluation with modern simplified approach...")

    # Process samples concurrently
    tasks = [evaluate_sample_context_precision(client, sample, model) for sample in data]
    results = await asyncio.gather(*tasks)

    logger.info("Context precision evaluation completed")
    return results


def save_results(
    results: List[Dict[str, Any]], dataset_display_name: str, output_path: str
):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")

    valid_scores = [
        r["context_precision_score"]
        for r in results
        if r["success"] and r["context_precision_score"] is not None
    ]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "modern_simplified",
        "metric": "context_precision",
        "num_samples": len(results),
        "num_successful": len(valid_scores),
        "average_context_precision": avg_score,
        "scores": valid_scores,
        "detailed_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Results saved successfully")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Evaluate context precision using Modern Simplified Approach"
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
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model to use for evaluation (default: gpt-5-mini-2025-08-07)",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting {args.dataset.upper()} context precision evaluation with Modern Simplified Approach"
    )

    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data, dataset_display_name, total_samples = load_preprocessed_data(args.data_file)

    results = await evaluate_context_precision_modern_simplified(data, client, args.model)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_modern_simplified.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))

    valid_scores = [
        r["context_precision_score"]
        for r in results
        if r["success"] and r["context_precision_score"] is not None
    ]
    if valid_scores:
        print(
            f"\n=== {dataset_display_name} Context Precision Evaluation Results (Modern Simplified) ==="
        )
        print(
            f"Average Context Precision Score: {sum(valid_scores) / len(valid_scores):.4f}"
        )
        print(f"Successful evaluations: {len(valid_scores)}/{total_samples}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())