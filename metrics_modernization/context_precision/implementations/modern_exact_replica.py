"""
Exact replica of Ragas Main context precision evaluation using official prompts
Supports both AmnestyQA and FIQA datasets via CLI
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Verification(BaseModel):
    """Individual context verification for context precision"""

    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


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


def split_contexts(contexts: str) -> List[str]:
    """Split combined contexts into individual context chunks"""
    # Simple splitting - could be improved based on actual context format
    # For now, treat entire contexts as single chunk since that's what the data seems to contain
    return [contexts]


async def evaluate_context_usefulness(
    client: AsyncOpenAI, question: str, context: str, answer: str, model: str = "gpt-5-mini-2025-08-07"
) -> Verification:
    """Evaluate context usefulness using exact Ragas Main context precision prompt"""

    # Use exact Ragas Main prompt for context precision
    context_precision_prompt = f"""Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.

--------EXAMPLES-----------
Example 1
Input: {{"question": "What can you tell me about Albert Einstein?", "context": "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.", "answer": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics."}}
Output: {{"reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.", "verdict": 1}}

Example 2
Input: {{"question": "who won 2020 icc world cup?", "context": "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.", "answer": "England"}}
Output: {{"reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.", "verdict": 1}}

Example 3
Input: {{"question": "What is the tallest mountain in the world?", "context": "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.", "answer": "Mount Everest."}}
Output: {{"reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.", "verdict": 0}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "context": "{context}", "answer": "{answer}"}}
Output: """

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": context_precision_prompt,
            }
        ],
        response_format=Verification,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return result


def calculate_average_precision(verdicts: List[Verification]) -> float:
    """Calculate average precision from verdicts"""
    verdict_list = [1 if ver.verdict else 0 for ver in verdicts]
    denominator = sum(verdict_list) + 1e-10
    numerator = sum(
        [
            (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
            for i in range(len(verdict_list))
        ]
    )
    score = numerator / denominator
    return score


async def evaluate_sample_context_precision_exact(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = "gpt-5-mini-2025-08-07"
) -> Dict[str, Any]:
    """Evaluate context precision using exact Ragas Main process"""

    # Split contexts into individual chunks (in our case, it's usually one chunk)
    context_chunks = split_contexts(sample["contexts"])
    
    if not context_chunks:
        return {
            "sample": sample,
            "context_precision_score": np.nan,
            "reasoning": "No context chunks found",
            "verifications": [],
            "success": True,
        }

    # Evaluate each context chunk for usefulness
    verifications = []
    for context in context_chunks:
        verification = await evaluate_context_usefulness(
            client, sample["question"], context, sample["answer"], model
        )
        verifications.append(verification)

    # Calculate average precision
    context_precision_score = calculate_average_precision(verifications)

    return {
        "sample": sample,
        "context_precision_score": context_precision_score,
        "reasoning": f"Context precision calculated from {len(verifications)} context chunks",
        "verifications": [
            {"reason": v.reason, "verdict": v.verdict}
            for v in verifications
        ],
        "success": True,
    }


async def evaluate_context_precision_exact(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = "gpt-5-mini-2025-08-07"
) -> List[Dict[str, Any]]:
    """Evaluate context precision using exact Ragas Main methodology"""
    logger.info("Starting context precision evaluation with exact Ragas Main approach...")

    tasks = [evaluate_sample_context_precision_exact(client, sample, model) for sample in data]
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
        if r["success"]
        and r["context_precision_score"] is not None
        and not np.isnan(r["context_precision_score"])
    ]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "modern_exact_replica",
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
        description="Evaluate context precision using Modern Exact Replica"
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
        f"Starting {args.dataset.upper()} context precision evaluation with Modern Exact Replica"
    )

    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data, dataset_display_name, total_samples = load_preprocessed_data(args.data_file)

    results = await evaluate_context_precision_exact(data, client, args.model)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_modern_exact_replica.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))

    valid_scores = [
        r["context_precision_score"]
        for r in results
        if r["success"]
        and r["context_precision_score"] is not None
        and not np.isnan(r["context_precision_score"])
    ]
    if valid_scores:
        print(
            f"\n=== {dataset_display_name} Context Precision Evaluation Results (Modern Exact Replica) ==="
        )
        print(
            f"Average Context Precision Score: {sum(valid_scores) / len(valid_scores):.4f}"
        )
        print(f"Successful evaluations: {len(valid_scores)}/{total_samples}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())