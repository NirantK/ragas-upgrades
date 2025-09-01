"""
Comprehensive Answer Correctness Evaluation - Run All Three Methods
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_correctness
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_with_ground_truth(data_file_path: str) -> tuple[List[Dict[str, Any]], str, int]:
    """Load data with proper ground truth"""
    logger.info(f"Loading data from {data_file_path}")
    
    with open(data_file_path, 'r') as f:
        dataset_info = json.load(f)
    
    data = dataset_info["data"]
    dataset_name = dataset_info["dataset_name"]
    total_samples = dataset_info["total_samples"]
    
    logger.info(f"Loaded {total_samples} samples from {dataset_name}")
    return data, dataset_name, total_samples


# === METHOD 1: CURRENT RAGAS MAIN ===
def evaluate_ragas_main(data: List[Dict], dataset_name: str) -> Dict[str, Any]:
    """Method 1: Current Ragas Main Implementation"""
    logger.info("=== RUNNING RAGAS MAIN ===")
    
    eval_data = []
    for sample in data:
        eval_data.append({
            "user_input": sample["question"],
            "response": sample["answer"],
            "reference": sample["ground_truth"],
            "retrieved_contexts": [sample.get("contexts", "")]
        })
    
    eval_dataset = EvaluationDataset.from_list(eval_data)
    result = evaluate(eval_dataset, metrics=[answer_correctness], show_progress=True)
    result_dict = result.to_pandas()
    
    return {
        "method": "ragas_main",
        "dataset": dataset_name,
        "average_score": result_dict["answer_correctness"].mean(),
        "scores": result_dict["answer_correctness"].tolist(),
        "num_samples": len(result_dict),
        "timestamp": datetime.now().isoformat()
    }


# === METHOD 2: MODERN EXACT REPLICA ===
class StatementGeneratorOutput(BaseModel):
    statements: List[str] = Field(description="The generated statements")

class StatementsWithReason(BaseModel):
    statement: str = Field(description="The statement text")
    reason: str = Field(description="The reason for classification")

class ClassificationWithReason(BaseModel):
    TP: List[StatementsWithReason] = Field(description="True positive statements")
    FP: List[StatementsWithReason] = Field(description="False positive statements") 
    FN: List[StatementsWithReason] = Field(description="False negative statements")


async def generate_statements(client: AsyncOpenAI, question: str, answer: str) -> List[str]:
    """Generate statements from answer"""
    statement_prompt = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

Now perform the same with the following input
input: {{"question": "{question}", "answer": "{answer}"}}
Output: """

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{
            "role": "user", 
            "content": statement_prompt.format(question=question, answer=answer)
        }],
        response_format=StatementGeneratorOutput,
        temperature=1,
    )
    return response.choices[0].message.parsed.statements


async def classify_statements(client: AsyncOpenAI, question: str, answer_statements: List[str], ground_truth_statements: List[str]) -> ClassificationWithReason:
    """Classify statements using CorrectnessClassifier logic"""
    classification_prompt = """Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.

Now perform the same with the following input
input: {{"question": "{question}", "answer": {answer_statements_json}, "ground_truth": {ground_truth_statements_json}}}
Output: """

    import json
    answer_statements_json = json.dumps(answer_statements)
    ground_truth_statements_json = json.dumps(ground_truth_statements)

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": classification_prompt.format(
                question=question, 
                answer_statements_json=answer_statements_json,
                ground_truth_statements_json=ground_truth_statements_json
            ),
        }],
        response_format=ClassificationWithReason,
        temperature=1,
    )
    return response.choices[0].message.parsed


async def get_embedding(client: AsyncOpenAI, text: str) -> List[float]:
    """Get embedding for text"""
    response = await client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


async def calculate_similarity(client: AsyncOpenAI, answer: str, ground_truth: str) -> float:
    """Calculate semantic similarity"""
    if not answer or not ground_truth:
        return 0.0
        
    answer_emb, gt_emb = await asyncio.gather(
        get_embedding(client, answer),
        get_embedding(client, ground_truth)
    )
    
    similarity = cosine_similarity([answer_emb], [gt_emb])[0][0]
    return float(similarity)


def calculate_f1_score(tp_count: int, fp_count: int, fn_count: int) -> float:
    """Calculate F1 score"""
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
    
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


async def evaluate_sample_exact_replica(client: AsyncOpenAI, sample: Dict) -> Dict[str, Any]:
    """Evaluate single sample using exact replica methodology"""
    question = sample["question"]
    answer = sample["answer"]
    ground_truth = sample["ground_truth"]
    
    if not ground_truth:
        return {"answer_correctness_score": 0.0, "success": False}

    # Step 1: Generate statements
    answer_statements, ground_truth_statements = await asyncio.gather(
        generate_statements(client, question, answer),
        generate_statements(client, question, ground_truth)
    )

    if not answer_statements or not ground_truth_statements:
        return {"answer_correctness_score": 0.0, "success": False}

    # Step 2: Classify statements
    classification = await classify_statements(client, question, answer_statements, ground_truth_statements)

    # Step 3: Calculate F1 score
    tp_count = len(classification.TP)
    fp_count = len(classification.FP)
    fn_count = len(classification.FN)
    factuality_score = calculate_f1_score(tp_count, fp_count, fn_count)

    # Step 4: Calculate similarity
    similarity_score = await calculate_similarity(client, answer, ground_truth)

    # Step 5: Weighted combination (75% factuality, 25% similarity)
    answer_correctness_score = np.average([factuality_score, similarity_score], weights=[0.75, 0.25])

    return {
        "answer_correctness_score": float(answer_correctness_score),
        "factuality_score": factuality_score,
        "similarity_score": similarity_score,
        "success": True
    }


async def evaluate_exact_replica(data: List[Dict], dataset_name: str) -> Dict[str, Any]:
    """Method 2: Modern Exact Replica Implementation"""
    logger.info("=== RUNNING MODERN EXACT REPLICA ===")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    tasks = [evaluate_sample_exact_replica(client, sample) for sample in data]
    results = await asyncio.gather(*tasks)
    
    valid_scores = [r["answer_correctness_score"] for r in results if r["success"]]
    
    return {
        "method": "modern_exact_replica",
        "dataset": dataset_name,
        "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "scores": valid_scores,
        "num_samples": len(valid_scores),
        "timestamp": datetime.now().isoformat()
    }


# === METHOD 3: MODERN SIMPLIFIED ===
class AnswerCorrectnessResponse(BaseModel):
    value: float = Field(..., description="Answer correctness score", ge=0.0, le=1.0)
    factuality_score: float = Field(..., description="Factuality score", ge=0.0, le=1.0)
    similarity_score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    reason: str = Field(..., description="Reasoning")


async def evaluate_sample_simplified(client: AsyncOpenAI, sample: Dict) -> Dict[str, Any]:
    """Evaluate single sample using simplified methodology"""
    question = sample["question"]
    answer = sample["answer"]
    ground_truth = sample["ground_truth"]
    
    if not ground_truth:
        return {"answer_correctness_score": 0.0, "success": False}

    prompt = f"""Given a question, an answer, and the ground truth (correct answer), evaluate the answer correctness which combines both factual accuracy and semantic similarity.

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Please evaluate the answer correctness by considering:
1. **Factuality** (0-1): How factually accurate is the answer compared to the ground truth?
2. **Semantic Similarity** (0-1): How semantically similar is the answer to the ground truth?

**Overall Answer Correctness** should be a weighted combination (factuality: 75%, similarity: 25%).

Provide your evaluation as a JSON object with 'value', 'factuality_score', 'similarity_score', and 'reason'.
"""

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=AnswerCorrectnessResponse,
        temperature=1,
    )

    result = response.choices[0].message.parsed
    return {
        "answer_correctness_score": result.value,
        "factuality_score": result.factuality_score,
        "similarity_score": result.similarity_score,
        "success": True
    }


async def evaluate_simplified(data: List[Dict], dataset_name: str) -> Dict[str, Any]:
    """Method 3: Modern Simplified Implementation"""
    logger.info("=== RUNNING MODERN SIMPLIFIED ===")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    tasks = [evaluate_sample_simplified(client, sample) for sample in data]
    results = await asyncio.gather(*tasks)
    
    valid_scores = [r["answer_correctness_score"] for r in results if r["success"]]
    
    return {
        "method": "modern_simplified", 
        "dataset": dataset_name,
        "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "scores": valid_scores,
        "num_samples": len(valid_scores),
        "timestamp": datetime.now().isoformat()
    }


async def run_comprehensive_evaluation():
    """Run all three evaluation methods on both datasets"""
    
    datasets = [
        ("metrics_modernization/answer_correctness/data/amnesty_data.json", "amnesty"),
        ("metrics_modernization/answer_correctness/data/fiqa_data.json", "fiqa")
    ]
    
    all_results = {}
    
    for data_file, dataset_key in datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING {dataset_key.upper()} DATASET")
        logger.info(f"{'='*80}")
        
        data, dataset_name, total_samples = load_data_with_ground_truth(data_file)
        
        # Method 1: Ragas Main
        try:
            ragas_result = evaluate_ragas_main(data, dataset_name)
            all_results[f"{dataset_key}_ragas_main"] = ragas_result
            logger.info(f"✅ Ragas Main - Average Score: {ragas_result['average_score']:.4f}")
        except Exception as e:
            logger.error(f"❌ Ragas Main failed: {e}")
        
        # Method 2: Exact Replica  
        try:
            exact_result = await evaluate_exact_replica(data, dataset_name)
            all_results[f"{dataset_key}_exact_replica"] = exact_result
            logger.info(f"✅ Exact Replica - Average Score: {exact_result['average_score']:.4f}")
        except Exception as e:
            logger.error(f"❌ Exact Replica failed: {e}")
        
        # Method 3: Simplified
        try:
            simple_result = await evaluate_simplified(data, dataset_name)
            all_results[f"{dataset_key}_simplified"] = simple_result
            logger.info(f"✅ Simplified - Average Score: {simple_result['average_score']:.4f}")
        except Exception as e:
            logger.error(f"❌ Simplified failed: {e}")
    
    # Save comprehensive results
    output_file = "metrics_modernization/answer_correctness/comprehensive_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n🎉 All evaluations complete! Results saved to {output_file}")
    
    # Print final comparison
    logger.info(f"\n{'='*80}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'='*80}")
    
    for dataset in ["amnesty", "fiqa"]:
        logger.info(f"\n{dataset.upper()} Dataset:")
        for method in ["ragas_main", "exact_replica", "simplified"]:
            key = f"{dataset}_{method}"
            if key in all_results:
                score = all_results[key]['average_score']
                samples = all_results[key]['num_samples']
                logger.info(f"  {method.replace('_', ' ').title()}: {score:.4f} ({samples} samples)")
            else:
                logger.info(f"  {method.replace('_', ' ').title()}: FAILED")
    
    return all_results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_evaluation())