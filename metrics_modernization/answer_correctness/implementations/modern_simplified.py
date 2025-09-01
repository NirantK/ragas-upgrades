"""
Simplified answer_correctness evaluation using direct OpenAI calls (modern simplified approach)
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


class AnswerCorrectnessResponse(BaseModel):
    """Response model for answer correctness evaluation"""

    value: float = Field(
        ..., description="Answer correctness score between 0 and 1", ge=0.0, le=1.0
    )
    factuality_score: float = Field(
        ..., description="Factuality component score between 0 and 1", ge=0.0, le=1.0
    )
    similarity_score: float = Field(
        ..., description="Semantic similarity component score between 0 and 1", ge=0.0, le=1.0
    )
    reason: str = Field(..., description="Reasoning for the answer correctness score")


def load_preprocessed_data(data_file_path: str) -> tuple[List[Dict[str, Any]], str, int]:
    """Load preprocessed data from centralized data file"""
    logger.info(f"Loading preprocessed data from {data_file_path}")
    
    with open(data_file_path, 'r') as f:
        dataset_info = json.load(f)
    
    data = dataset_info["data"]
    dataset_name = dataset_info["dataset_name"]
    total_samples = dataset_info["total_samples"]
    
    # Add ground truth field if missing
    for sample in data:
        if "ground_truth" not in sample and "reference" in sample:
            sample["ground_truth"] = sample["reference"]
        elif "ground_truth" not in sample and "reference" not in sample:
            # For datasets without explicit ground truth, use a placeholder
            sample["ground_truth"] = ""
    
    logger.info(f"Loaded {total_samples} samples from {dataset_name}")
    return data, dataset_name, total_samples


async def evaluate_sample_answer_correctness(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Evaluate answer correctness for a single sample using OpenAI"""

    question = sample["question"]
    answer = sample["answer"]
    ground_truth = sample.get("ground_truth", "")
    
    if not ground_truth:
        return {
            "sample": sample,
            "answer_correctness_score": 0.0,
            "factuality_score": 0.0,
            "similarity_score": 0.0,
            "reasoning": "No ground truth provided for comparison",
            "success": True,
        }

    prompt = f"""
Given a question, an answer, and the ground truth (correct answer), evaluate the answer correctness which combines both factual accuracy and semantic similarity.

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Please evaluate the answer correctness by considering two components:

1. **Factuality** (0-1): How factually accurate is the answer compared to the ground truth? Consider:
   - Are the facts in the answer supported by the ground truth?
   - Are there any contradictions or false information?
   - Are important facts from ground truth missing?

2. **Semantic Similarity** (0-1): How semantically similar is the answer to the ground truth? Consider:
   - Do they convey the same meaning?
   - Are they talking about the same concepts?
   - How similar are they in content and intent?

**Overall Answer Correctness** should be a weighted combination (factuality: 75%, similarity: 25%).

Provide your evaluation as a JSON object with:
- 'value': overall answer correctness score (0-1)
- 'factuality_score': factual accuracy score (0-1) 
- 'similarity_score': semantic similarity score (0-1)
- 'reason': detailed explanation of the scoring

Scale:
- 0: Completely incorrect/no similarity
- 0.5: Partially correct/moderate similarity
- 1: Completely correct/very similar
"""

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=AnswerCorrectnessResponse,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return {
        "sample": sample,
        "answer_correctness_score": result.value,
        "factuality_score": result.factuality_score,
        "similarity_score": result.similarity_score,
        "reasoning": result.reason,
        "success": True,
    }


async def evaluate_answer_correctness_modern_simplified(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """Evaluate answer correctness using modern simplified approach"""
    logger.info("Starting answer correctness evaluation with modern simplified approach...")

    # Process samples concurrently
    tasks = [evaluate_sample_answer_correctness(client, sample, model) for sample in data]
    results = await asyncio.gather(*tasks)

    logger.info("Answer correctness evaluation completed")
    return results


def save_results(
    results: List[Dict[str, Any]], dataset_display_name: str, output_path: str
):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")

    valid_scores = [
        r["answer_correctness_score"]
        for r in results
        if r["success"] and r["answer_correctness_score"] is not None
    ]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "modern_simplified",
        "metric": "answer_correctness",
        "num_samples": len(results),
        "num_successful": len(valid_scores),
        "average_answer_correctness": avg_score,
        "scores": valid_scores,
        "detailed_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Results saved successfully")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Evaluate answer_correctness using Modern Simplified Approach"
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
        default="gpt-4o-mini",
        help="Model to use for evaluation (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting {args.dataset.upper()} answer_correctness evaluation with Modern Simplified Approach"
    )

    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data, dataset_display_name, total_samples = load_preprocessed_data(args.data_file)

    results = await evaluate_answer_correctness_modern_simplified(data, client, args.model)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_modern_simplified.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))

    valid_scores = [
        r["answer_correctness_score"]
        for r in results
        if r["success"] and r["answer_correctness_score"] is not None
    ]
    if valid_scores:
        print(
            f"\n=== {dataset_display_name} Answer Correctness Evaluation Results (Modern Simplified) ==="
        )
        print(
            f"Average Answer Correctness Score: {sum(valid_scores) / len(valid_scores):.4f}"
        )
        print(f"Successful evaluations: {len(valid_scores)}/{total_samples}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())