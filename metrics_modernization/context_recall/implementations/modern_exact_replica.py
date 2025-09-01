"""
Exact replica of Ragas Main context recall evaluation using official prompts
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


class ContextRecallClassification(BaseModel):
    """Individual statement classification for context recall"""

    statement: str = Field(..., description="The original statement, word-by-word")
    reason: str = Field(..., description="The reason for the classification")
    attributed: int = Field(..., description="Binary classification (0/1) of attribution")


class ContextRecallClassifications(BaseModel):
    """Output model for context recall classification"""

    classifications: List[ContextRecallClassification]


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


def split_into_statements(text: str) -> List[str]:
    """Split text into individual statements/sentences"""
    # Simple sentence splitting - could be improved with more sophisticated NLP
    import re
    sentences = re.split(r'[.!?]+', text)
    statements = [s.strip() for s in sentences if s.strip()]
    return statements


async def evaluate_statements_context_recall(
    client: AsyncOpenAI, question: str, context: str, answer: str, model: str = "gpt-5-mini-2025-08-07"
) -> List[ContextRecallClassification]:
    """Evaluate statements using exact Ragas Main context recall prompt"""

    # Split answer into statements
    statements = split_into_statements(answer)
    
    if not statements:
        return []

    # Use exact Ragas Main prompt for context recall classification
    context_recall_prompt = f"""Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason.

--------EXAMPLES-----------
Example 1
Input: {{"question": "What can you tell me about albert Albert Einstein?", "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.", "answer": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895."}}
Output: {{"classifications": [{{"statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.", "reason": "The date of birth of Einstein is mentioned clearly in the context.", "attributed": 1}}, {{"statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.", "reason": "The exact sentence is present in the given context.", "attributed": 1}}, {{"statement": "He published 4 papers in 1905.", "reason": "There is no mention about papers he wrote in the given context.", "attributed": 0}}, {{"statement": "Einstein moved to Switzerland in 1895.", "reason": "There is no supporting evidence for this in the given context.", "attributed": 0}}]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "context": "{context}", "answer": "{answer}"}}
Output: """

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": context_recall_prompt,
            }
        ],
        response_format=ContextRecallClassifications,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return result.classifications


async def evaluate_sample_context_recall_exact(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = "gpt-5-mini-2025-08-07"
) -> Dict[str, Any]:
    """Evaluate context recall using exact Ragas Main process"""

    classifications = await evaluate_statements_context_recall(
        client, sample["question"], sample["contexts"], sample["answer"], model
    )

    if not classifications:
        return {
            "sample": sample,
            "context_recall_score": np.nan,
            "reasoning": "No statements found in answer",
            "classifications": [],
            "success": True,
        }

    attributed_statements = sum(1 for c in classifications if c.attributed == 1)
    total_statements = len(classifications)
    context_recall_score = (
        attributed_statements / total_statements if total_statements > 0 else 0.0
    )

    return {
        "sample": sample,
        "context_recall_score": context_recall_score,
        "reasoning": f"Attributed statements: {attributed_statements}/{total_statements}",
        "classifications": [
            {"statement": c.statement, "reason": c.reason, "attributed": c.attributed}
            for c in classifications
        ],
        "success": True,
    }


async def evaluate_context_recall_exact(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = "gpt-5-mini-2025-08-07"
) -> List[Dict[str, Any]]:
    """Evaluate context recall using exact Ragas Main methodology"""
    logger.info("Starting context recall evaluation with exact Ragas Main approach...")

    tasks = [evaluate_sample_context_recall_exact(client, sample, model) for sample in data]
    results = await asyncio.gather(*tasks)

    logger.info("Context recall evaluation completed")
    return results


def save_results(
    results: List[Dict[str, Any]], dataset_display_name: str, output_path: str
):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")

    valid_scores = [
        r["context_recall_score"]
        for r in results
        if r["success"]
        and r["context_recall_score"] is not None
        and not np.isnan(r["context_recall_score"])
    ]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "modern_exact_replica",
        "metric": "context_recall",
        "num_samples": len(results),
        "num_successful": len(valid_scores),
        "average_context_recall": avg_score,
        "scores": valid_scores,
        "detailed_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Results saved successfully")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Evaluate context recall using Modern Exact Replica"
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
        f"Starting {args.dataset.upper()} context recall evaluation with Modern Exact Replica"
    )

    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data, dataset_display_name, total_samples = load_preprocessed_data(args.data_file)

    results = await evaluate_context_recall_exact(data, client, args.model)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_modern_exact_replica.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))

    valid_scores = [
        r["context_recall_score"]
        for r in results
        if r["success"]
        and r["context_recall_score"] is not None
        and not np.isnan(r["context_recall_score"])
    ]
    if valid_scores:
        print(
            f"\n=== {dataset_display_name} Context Recall Evaluation Results (Modern Exact Replica) ==="
        )
        print(
            f"Average Context Recall Score: {sum(valid_scores) / len(valid_scores):.4f}"
        )
        print(f"Successful evaluations: {len(valid_scores)}/{total_samples}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())