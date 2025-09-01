"""
Exact replica of Ragas Main answer_correctness evaluation using official prompts
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
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatementGeneratorInput(BaseModel):
    """Input model for statement generation"""

    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementGeneratorOutput(BaseModel):
    """Output model for statement generation"""

    statements: List[str] = Field(description="The generated statements")


class StatementsWithReason(BaseModel):
    """Individual statement with reason"""

    statement: str = Field(description="The statement text")
    reason: str = Field(description="The reason for classification")


class ClassificationWithReason(BaseModel):
    """Classification of statements into TP/FP/FN with reasoning"""

    TP: List[StatementsWithReason] = Field(description="True positive statements")
    FP: List[StatementsWithReason] = Field(description="False positive statements") 
    FN: List[StatementsWithReason] = Field(description="False negative statements")


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


async def generate_statements(
    client: AsyncOpenAI, question: str, answer: str, model: str = "gpt-4o-mini"
) -> List[str]:
    """Generate statements from answer using exact Ragas Main prompt"""

    statement_prompt = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

--------EXAMPLES-----------
Example 1
Input: {{"question": "Who was Albert Einstein and what is he best known for?", "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."}}
Output: {{"statements": ["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", "Albert Einstein was best known for developing the theory of relativity.", "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": "{answer}"}}
Output: """

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": statement_prompt.format(question=question, answer=answer),
            }
        ],
        response_format=StatementGeneratorOutput,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return result.statements


async def classify_statements(
    client: AsyncOpenAI, question: str, answer_statements: List[str], ground_truth_statements: List[str], model: str = "gpt-4o-mini"
) -> ClassificationWithReason:
    """Classify statements using exact Ragas Main CorrectnessClassifier prompt"""

    classification_prompt = """Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.

--------EXAMPLES-----------
Example 1
Input: {{"question": "What powers the sun and what is its primary function?", "answer": ["The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "The primary function of the sun is to provide light to the solar system."], "ground_truth": ["The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "This fusion process in the sun's core releases a tremendous amount of energy.", "The energy from the sun provides heat and light, which are essential for life on Earth.", "The sun's light plays a critical role in Earth's climate system.", "Sunlight helps to drive the weather and ocean currents."]}}
Output: {{"TP": [{{"statement": "The primary function of the sun is to provide light to the solar system.", "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."}}], "FP": [{{"statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."}}], "FN": [{{"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "reason": "This accurate description of the sun's power source is not included in the answer."}}, {{"statement": "This fusion process in the sun's core releases a tremendous amount of energy.", "reason": "This process and its significance are not mentioned in the answer."}}, {{"statement": "The energy from the sun provides heat and light, which are essential for life on Earth.", "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."}}, {{"statement": "The sun's light plays a critical role in Earth's climate system.", "reason": "This broader impact of the sun's light on Earth's climate system is not addressed in the answer."}}, {{"statement": "Sunlight helps to drive the weather and ocean currents.", "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."}}]}}

Example 2
Input: {{"question": "What is the boiling point of water?", "answer": ["The boiling point of water is 100 degrees Celsius at sea level"], "ground_truth": ["The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "The boiling point of water can change with altitude."]}}
Output: {{"TP": [{{"statement": "The boiling point of water is 100 degrees Celsius at sea level", "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."}}], "FP": [], "FN": [{{"statement": "The boiling point of water can change with altitude.", "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."}}]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": {answer_statements_json}, "ground_truth": {ground_truth_statements_json}}}
Output: """

    import json

    answer_statements_json = json.dumps(answer_statements)
    ground_truth_statements_json = json.dumps(ground_truth_statements)

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": classification_prompt.format(
                    question=question, 
                    answer_statements_json=answer_statements_json,
                    ground_truth_statements_json=ground_truth_statements_json
                ),
            }
        ],
        response_format=ClassificationWithReason,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return result


async def get_embedding(client: AsyncOpenAI, text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for text using OpenAI"""
    response = await client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


async def calculate_similarity(client: AsyncOpenAI, answer: str, ground_truth: str) -> float:
    """Calculate semantic similarity between answer and ground truth"""
    if not answer or not ground_truth:
        return 0.0
        
    answer_emb, gt_emb = await asyncio.gather(
        get_embedding(client, answer),
        get_embedding(client, ground_truth)
    )
    
    # Calculate cosine similarity
    similarity = cosine_similarity([answer_emb], [gt_emb])[0][0]
    return float(similarity)


def calculate_f1_score(tp_count: int, fp_count: int, fn_count: int, beta: float = 1.0) -> float:
    """Calculate F-beta score from TP/FP/FN counts"""
    if tp_count + fp_count == 0:
        precision = 0.0
    else:
        precision = tp_count / (tp_count + fp_count)
    
    if tp_count + fn_count == 0:
        recall = 0.0
    else:
        recall = tp_count / (tp_count + fn_count)
    
    if precision + recall == 0:
        return 0.0
    
    beta_squared = beta ** 2
    f_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    return f_score


async def evaluate_sample_answer_correctness(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = "gpt-4o-mini", weights: List[float] = [0.75, 0.25]
) -> Dict[str, Any]:
    """Evaluate answer correctness for a single sample using exact Ragas Main methodology"""

    question = sample["question"]
    answer = sample["answer"]
    ground_truth = sample.get("ground_truth", "")
    
    if not ground_truth:
        return {
            "sample": sample,
            "answer_correctness_score": np.nan,
            "factuality_score": np.nan,
            "similarity_score": np.nan,
            "reasoning": "No ground truth provided",
            "success": False,
        }

    # Step 1: Generate statements from both answer and ground truth
    answer_statements, ground_truth_statements = await asyncio.gather(
        generate_statements(client, question, answer, model),
        generate_statements(client, question, ground_truth, model)
    )

    if not answer_statements or not ground_truth_statements:
        return {
            "sample": sample,
            "answer_correctness_score": np.nan,
            "factuality_score": np.nan,
            "similarity_score": np.nan,
            "reasoning": "Failed to generate statements",
            "success": False,
        }

    # Step 2: Classify statements
    classification = await classify_statements(
        client, question, answer_statements, ground_truth_statements, model
    )

    # Step 3: Calculate F1 score
    tp_count = len(classification.TP)
    fp_count = len(classification.FP)
    fn_count = len(classification.FN)
    
    factuality_score = calculate_f1_score(tp_count, fp_count, fn_count)

    # Step 4: Calculate semantic similarity if weight > 0
    if weights[1] > 0:
        similarity_score = await calculate_similarity(client, answer, ground_truth)
    else:
        similarity_score = 0.0

    # Step 5: Calculate weighted average
    answer_correctness_score = np.average(
        [factuality_score, similarity_score],
        weights=weights,
    )

    return {
        "sample": sample,
        "answer_correctness_score": float(answer_correctness_score),
        "factuality_score": factuality_score,
        "similarity_score": similarity_score,
        "reasoning": f"TP: {tp_count}, FP: {fp_count}, FN: {fn_count}, F1: {factuality_score:.4f}, Similarity: {similarity_score:.4f}",
        "answer_statements": answer_statements,
        "ground_truth_statements": ground_truth_statements,
        "classification": {
            "TP": [{"statement": s.statement, "reason": s.reason} for s in classification.TP],
            "FP": [{"statement": s.statement, "reason": s.reason} for s in classification.FP],
            "FN": [{"statement": s.statement, "reason": s.reason} for s in classification.FN],
        },
        "success": True,
    }


async def evaluate_answer_correctness_exact(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = "gpt-4o-mini", weights: List[float] = [0.75, 0.25]
) -> List[Dict[str, Any]]:
    """Evaluate answer correctness using exact Ragas Main methodology"""
    logger.info("Starting answer correctness evaluation with exact Ragas Main approach...")

    tasks = [evaluate_sample_answer_correctness(client, sample, model, weights) for sample in data]
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
        if r["success"]
        and r["answer_correctness_score"] is not None
        and not np.isnan(r["answer_correctness_score"])
    ]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "modern_exact_replica",
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
        description="Evaluate answer_correctness using Modern Exact Replica"
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
        "--weights",
        type=float,
        nargs=2,
        default=[0.75, 0.25],
        help="Weights for factuality and similarity (default: 0.75 0.25)",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting {args.dataset.upper()} answer_correctness evaluation with Modern Exact Replica"
    )

    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data, dataset_display_name, total_samples = load_preprocessed_data(args.data_file)

    results = await evaluate_answer_correctness_exact(data, client, args.model, args.weights)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_modern_exact_replica.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))

    valid_scores = [
        r["answer_correctness_score"]
        for r in results
        if r["success"]
        and r["answer_correctness_score"] is not None
        and not np.isnan(r["answer_correctness_score"])
    ]
    if valid_scores:
        print(
            f"\n=== {dataset_display_name} Answer Correctness Evaluation Results (Modern Exact Replica) ==="
        )
        print(
            f"Average Answer Correctness Score: {sum(valid_scores) / len(valid_scores):.4f}"
        )
        print(f"Successful evaluations: {len(valid_scores)}/{total_samples}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())