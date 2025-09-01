"""
Simplified faithfulness evaluation using direct OpenAI calls (mimicking ragas experimental approach)
Supports both AmnestyQA and FIQA datasets via CLI
"""
import json
import logging
import argparse
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaithfulnessResponse(BaseModel):
    """Response model for faithfulness evaluation"""
    value: float = Field(..., description="Faithfulness score between 0 and 1", ge=0.0, le=1.0)
    reason: str = Field(..., description="Reasoning for the faithfulness score")


def load_dataset_by_name(dataset_name: str, num_samples: int = 10, random_seed: int = 42) -> tuple[List[Dict[str, Any]], str]:
    """Load dataset by name"""
    logger.info(f"Loading {dataset_name} dataset with {num_samples} samples")
    
    if dataset_name.lower() == "amnesty":
        hf_dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3", trust_remote_code=True)["eval"]
        dataset_display_name = "AmnestyQA"
    elif dataset_name.lower() == "fiqa":
        hf_dataset = load_dataset("explodinggradients/fiqa", "ragas_eval", trust_remote_code=True)["baseline"]
        dataset_display_name = "FIQA"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: amnesty, fiqa")
    
    selected_samples = hf_dataset.shuffle(seed=random_seed).select(range(num_samples))
    
    data = []
    for sample in selected_samples:
        if dataset_name.lower() == "fiqa":
            question = sample["question"] 
            answer = sample["answer"]
            contexts = "\n".join(sample["contexts"])
        else:
            question = sample["user_input"]
            answer = sample["response"] 
            contexts = "\n".join(sample["retrieved_contexts"])
            
        data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts
        })
    
    logger.info(f"Loaded {len(data)} samples from {dataset_display_name}")
    return data, dataset_display_name


async def evaluate_sample_faithfulness(client: AsyncOpenAI, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate faithfulness for a single sample using OpenAI"""
    
    prompt = f"""
Given a question, answer, and retrieved contexts, evaluate how faithful the answer is to the provided contexts.

Faithfulness measures whether the answer can be inferred from the given contexts. A highly faithful answer contains only information that can be verified from the contexts.

Question: {sample['question']}
Answer: {sample['answer']}
Retrieved Contexts: {sample['contexts']}

Please evaluate the faithfulness of the answer on a scale from 0 to 1, where:
- 0: Answer contains information not supported by contexts or contradicts them
- 1: Answer is completely faithful and all information can be verified from contexts

Provide your evaluation as a JSON object with 'value' (float between 0 and 1) and 'reason' (string explanation).
"""

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=FaithfulnessResponse,
        temperature=0
    )
    
    result = response.choices[0].message.parsed
    return {
        "sample": sample,
        "faithfulness_score": result.value,
        "reasoning": result.reason,
        "success": True
    }


async def evaluate_faithfulness_experimental(
    data: List[Dict[str, Any]], 
    client: AsyncOpenAI
) -> List[Dict[str, Any]]:
    """Evaluate faithfulness using experimental approach"""
    logger.info("Starting faithfulness evaluation with experimental approach...")
    
    # Process samples concurrently
    tasks = [evaluate_sample_faithfulness(client, sample) for sample in data]
    results = await asyncio.gather(*tasks)
    
    logger.info("Faithfulness evaluation completed")
    return results


def save_results(results: List[Dict[str, Any]], dataset_display_name: str, output_path: str):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")
    
    valid_scores = [r["faithfulness_score"] for r in results if r["success"] and r["faithfulness_score"] is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "ragas_experimental_simple",
        "metric": "faithfulness",
        "num_samples": len(results),
        "num_successful": len(valid_scores),
        "average_faithfulness": avg_score,
        "scores": valid_scores,
        "detailed_results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info("Results saved successfully")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Evaluate faithfulness using Simplified Experimental Approach")
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
    
    logger.info(f"Starting {args.dataset.upper()} faithfulness evaluation with Experimental Approach")
    
    import os
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    data, dataset_display_name = load_dataset_by_name(
        args.dataset,
        num_samples=args.samples,
        random_seed=args.seed
    )
    
    results = await evaluate_faithfulness_experimental(data, client)
    
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{args.dataset}_ragas_experimental.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))
    
    valid_scores = [r["faithfulness_score"] for r in results if r["success"] and r["faithfulness_score"] is not None]
    if valid_scores:
        print(f"\n=== {dataset_display_name} Faithfulness Evaluation Results (Experimental) ===")
        print(f"Average Faithfulness Score: {sum(valid_scores) / len(valid_scores):.4f}")
        print(f"Successful evaluations: {len(valid_scores)}/{len(results)}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())