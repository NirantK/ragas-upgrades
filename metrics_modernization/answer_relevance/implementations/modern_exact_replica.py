"""
Exact replica of Ragas Main answer_relevance evaluation using official prompts
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
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseRelevanceInput(BaseModel):
    """Input model for response relevance evaluation"""
    response: str = Field(description="The response to evaluate")


class ResponseRelevanceOutput(BaseModel):
    """Output model for response relevance evaluation"""
    question: str = Field(description="Generated question for the response")
    noncommittal: int = Field(description="1 if noncommittal, 0 if committal")


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


async def generate_questions(
    client: AsyncOpenAI, response: str, model: str = "gpt-4o-mini", strictness: int = 3
) -> List[ResponseRelevanceOutput]:
    """Step 1: Generate questions from response using exact Ragas Main prompt"""

    question_prompt = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers

--------EXAMPLES-----------
Example 1
Input: {{"response": "Albert Einstein was born in Germany."}}
Output: {{"question": "Where was Albert Einstein born?", "noncommittal": 0}}

Example 2
Input: {{"response": "I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. "}}
Output: {{"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}}
-----------------------------

Now perform the same with the following input
input: {{"response": "{response}"}}
Output: """

    # Generate multiple questions based on strictness parameter
    tasks = []
    for _ in range(strictness):
        task = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": question_prompt.format(response=response),
                }
            ],
            response_format=ResponseRelevanceOutput,
            temperature=1e-8,  # Match Ragas Main temperature
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    results = [response.choices[0].message.parsed for response in responses]
    return results


def calculate_similarity(question: str, generated_questions: List[str], embeddings_model) -> np.ndarray:
    """Calculate cosine similarity between original question and generated questions"""
    if not generated_questions:
        return np.array([])
    
    # Get embeddings for original question
    question_vec = embeddings_model.encode([question])
    question_vec = question_vec.reshape(1, -1)
    
    # Get embeddings for generated questions
    gen_question_vec = embeddings_model.encode(generated_questions)
    gen_question_vec = gen_question_vec.reshape(len(generated_questions), -1)
    
    # Calculate cosine similarity
    norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
    similarity = np.dot(gen_question_vec, question_vec.T).reshape(-1) / norm
    
    return similarity


async def evaluate_sample_answer_relevance_exact(
    client: AsyncOpenAI, sample: Dict[str, Any], embeddings_model, model: str = "gpt-4o-mini", strictness: int = 3
) -> Dict[str, Any]:
    """Evaluate answer_relevance using exact Ragas Main methodology"""

    # Step 1: Generate questions from answer
    generated_responses = await generate_questions(client, sample["answer"], model, strictness)
    
    generated_questions = [resp.question for resp in generated_responses]
    noncommittal_flags = [resp.noncommittal for resp in generated_responses]
    
    # Check if all generated questions are empty (similar to Ragas Main handling)
    if all(q == "" for q in generated_questions):
        return {
            "sample": sample,
            "answer_relevance_score": np.nan,
            "reasoning": "Invalid JSON response. Expected dictionary with key 'question'",
            "generated_questions": generated_questions,
            "noncommittal_flags": noncommittal_flags,
            "success": True,
        }
    
    # Step 2: Calculate similarity between original question and generated questions
    cosine_similarities = calculate_similarity(sample["question"], generated_questions, embeddings_model)
    
    # Step 3: Calculate final score using Ragas Main methodology
    # Score = average_similarity * (1 - any_noncommittal)
    avg_similarity = cosine_similarities.mean() if len(cosine_similarities) > 0 else 0.0
    committal_factor = int(not np.any(noncommittal_flags))  # 1 if all committal, 0 if any noncommittal
    
    answer_relevance_score = float(avg_similarity * committal_factor)
    
    return {
        "sample": sample,
        "answer_relevance_score": answer_relevance_score,
        "reasoning": f"Average similarity: {avg_similarity:.4f}, Committal factor: {committal_factor}",
        "generated_questions": generated_questions,
        "noncommittal_flags": noncommittal_flags,
        "cosine_similarities": cosine_similarities.tolist() if len(cosine_similarities) > 0 else [],
        "success": True,
    }


async def evaluate_answer_relevance_exact(
    data: List[Dict[str, Any]], client: AsyncOpenAI, embeddings_model, model: str = "gpt-4o-mini", strictness: int = 3
) -> List[Dict[str, Any]]:
    """Evaluate answer_relevance using exact Ragas Main methodology"""
    logger.info("Starting answer_relevance evaluation with exact Ragas Main approach...")

    tasks = [evaluate_sample_answer_relevance_exact(client, sample, embeddings_model, model, strictness) for sample in data]
    results = await asyncio.gather(*tasks)

    logger.info("Answer_relevance evaluation completed")
    return results


def save_results(
    results: List[Dict[str, Any]], dataset_display_name: str, output_path: str
):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")

    valid_scores = [
        r["answer_relevance_score"]
        for r in results
        if r["success"]
        and r["answer_relevance_score"] is not None
        and not np.isnan(r["answer_relevance_score"])
    ]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "modern_exact_replica",
        "metric": "answer_relevance",
        "num_samples": len(results),
        "num_successful": len(valid_scores),
        "average_answer_relevance": avg_score,
        "scores": valid_scores,
        "detailed_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Results saved successfully")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Evaluate answer_relevance using Modern Exact Replica"
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
    parser.add_argument(
        "--strictness",
        type=int,
        default=3,
        help="Number of questions to generate per answer (default: 3)",
    )
    parser.add_argument(
        "--embeddings-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Sentence transformers model for embeddings (default: BAAI/bge-small-en-v1.5)",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting {args.dataset.upper()} answer_relevance evaluation with Modern Exact Replica"
    )

    import os

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize embeddings model
    logger.info(f"Loading embeddings model: {args.embeddings_model}")
    embeddings_model = SentenceTransformer(args.embeddings_model)

    data, dataset_display_name, total_samples = load_preprocessed_data(args.data_file)

    results = await evaluate_answer_relevance_exact(data, client, embeddings_model, args.model, args.strictness)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_modern_exact_replica.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))

    valid_scores = [
        r["answer_relevance_score"]
        for r in results
        if r["success"]
        and r["answer_relevance_score"] is not None
        and not np.isnan(r["answer_relevance_score"])
    ]
    if valid_scores:
        print(
            f"\n=== {dataset_display_name} Answer Relevance Evaluation Results (Modern Exact Replica) ==="
        )
        print(
            f"Average Answer Relevance Score: {sum(valid_scores) / len(valid_scores):.4f}"
        )
        print(f"Successful evaluations: {len(valid_scores)}/{total_samples}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())