"""
Exact replica of Ragas Main faithfulness evaluation using two-step process
Supports both AmnestyQA and FIQA datasets via CLI
"""
import json
import logging
import argparse
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import numpy as np

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

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


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement faithfulness assessment"""
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementInput(BaseModel):
    """Input model for NLI statement evaluation"""
    context: str = Field(..., description="The context of the question")
    statements: List[str] = Field(..., description="The statements to judge")


class NLIStatementOutput(BaseModel):
    """Output model for NLI statement evaluation"""
    statements: List[StatementFaithfulnessAnswer]


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


async def generate_statements(client: AsyncOpenAI, question: str, answer: str) -> List[str]:
    """Step 1: Generate statements from answer using exact Ragas Main prompt"""
    
    statement_prompt = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"$defs": {{"StatementGeneratorOutput": {{"properties": {{"statements": {{"description": "The generated statements", "items": {{"type": "string"}}, "title": "Statements", "type": "array"}}}}, "required": ["statements"], "title": "StatementGeneratorOutput", "type": "object"}}}}, "$ref": "#/$defs/StatementGeneratorOutput"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{"question": "Who was Albert Einstein and what is he best known for?", "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."}}
Output: {{"statements": ["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", "Albert Einstein was best known for developing the theory of relativity.", "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": "{answer}"}}
Output: """

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": statement_prompt.format(question=question, answer=answer)}],
        response_format=StatementGeneratorOutput,
        temperature=1e-8
    )
    
    result = response.choices[0].message.parsed
    return result.statements


async def evaluate_statements_nli(client: AsyncOpenAI, context: str, statements: List[str]) -> List[StatementFaithfulnessAnswer]:
    """Step 2: Evaluate statements using exact Ragas Main NLI prompt"""
    
    nli_prompt = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"$defs": {{"NLIStatementOutput": {{"properties": {{"statements": {{"items": {{"$ref": "#/$defs/StatementFaithfulnessAnswer"}}, "title": "Statements", "type": "array"}}}}, "required": ["statements"], "title": "NLIStatementOutput", "type": "object"}}, "StatementFaithfulnessAnswer": {{"properties": {{"reason": {{"description": "the reason of the verdict", "title": "Reason", "type": "string"}}, "statement": {{"description": "the original statement, word-by-word", "title": "Statement", "type": "string"}}, "verdict": {{"description": "the verdict(0/1) of the faithfulness.", "title": "Verdict", "type": "integer"}}}}, "required": ["statement", "reason", "verdict"], "title": "StatementFaithfulnessAnswer", "type": "object"}}}}, "$ref": "#/$defs/NLIStatementOutput"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{"context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.", "statements": ["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]}}
Output: {{"statements": [{{"statement": "John is majoring in Biology.", "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", "verdict": 0}}, {{"statement": "John is taking a course on Artificial Intelligence.", "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.", "verdict": 0}}, {{"statement": "John is a dedicated student.", "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.", "verdict": 1}}, {{"statement": "John has a part-time job.", "reason": "There is no information given in the context about John having a part-time job.", "verdict": 0}}]}}

Example 2
Input: {{"context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.", "statements": ["Albert Einstein was a genius."]}}
Output: {{"statements": [{{"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated", "verdict": 0}}]}}
-----------------------------

Now perform the same with the following input
input: {{"context": "{context}", "statements": {statements_json}}}
Output: """

    import json
    statements_json = json.dumps(statements)
    
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": nli_prompt.format(context=context, statements_json=statements_json)}],
        response_format=NLIStatementOutput,
        temperature=1e-8
    )
    
    result = response.choices[0].message.parsed
    return result.statements


async def evaluate_sample_faithfulness_exact(client: AsyncOpenAI, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate faithfulness using exact Ragas Main two-step process"""
    
    statements = await generate_statements(client, sample['question'], sample['answer'])
    
    if not statements:
        return {
            "sample": sample,
            "faithfulness_score": np.nan,
            "reasoning": "No statements generated from answer",
            "statements": [],
            "verdicts": [],
            "success": True
        }
    
    verdicts = await evaluate_statements_nli(client, sample['contexts'], statements)
    
    faithful_statements = sum(1 for verdict in verdicts if verdict.verdict == 1)
    total_statements = len(statements)
    faithfulness_score = faithful_statements / total_statements if total_statements > 0 else 0.0
    
    return {
        "sample": sample,
        "faithfulness_score": faithfulness_score,
        "reasoning": f"Faithful statements: {faithful_statements}/{total_statements}",
        "statements": statements,
        "verdicts": [{"statement": v.statement, "reason": v.reason, "verdict": v.verdict} for v in verdicts],
        "success": True
    }


async def evaluate_faithfulness_exact(data: List[Dict[str, Any]], client: AsyncOpenAI) -> List[Dict[str, Any]]:
    """Evaluate faithfulness using exact Ragas Main methodology"""
    logger.info("Starting faithfulness evaluation with exact Ragas Main approach...")
    
    tasks = [evaluate_sample_faithfulness_exact(client, sample) for sample in data]
    results = await asyncio.gather(*tasks)
    
    logger.info("Faithfulness evaluation completed")
    return results


def save_results(results: List[Dict[str, Any]], dataset_display_name: str, output_path: str):
    """Save evaluation results to file"""
    logger.info(f"Saving results to {output_path}")
    
    valid_scores = [r["faithfulness_score"] for r in results if r["success"] and r["faithfulness_score"] is not None and not np.isnan(r["faithfulness_score"])]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_display_name,
        "framework": "ragas_experimental_exact",
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
    parser = argparse.ArgumentParser(description="Evaluate faithfulness using Exact Ragas Main Replica")
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
    
    logger.info(f"Starting {args.dataset.upper()} faithfulness evaluation with Exact Ragas Main Replica")
    
    import os
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    data, dataset_display_name = load_dataset_by_name(
        args.dataset,
        num_samples=args.samples,
        random_seed=args.seed
    )
    
    results = await evaluate_faithfulness_exact(data, client)
    
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{args.dataset}_ragas_experimental_exact.json"
    output_path = results_dir / output_filename
    save_results(results, dataset_display_name, str(output_path))
    
    valid_scores = [r["faithfulness_score"] for r in results if r["success"] and r["faithfulness_score"] is not None and not np.isnan(r["faithfulness_score"])]
    if valid_scores:
        print(f"\n=== {dataset_display_name} Faithfulness Evaluation Results (Exact Replica) ===")
        print(f"Average Faithfulness Score: {sum(valid_scores) / len(valid_scores):.4f}")
        print(f"Successful evaluations: {len(valid_scores)}/{len(results)}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())