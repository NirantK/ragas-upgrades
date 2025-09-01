"""Faithfulness evaluation using OpenAI direct calls"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Change to the metrics_modernization directory to enable relative imports
script_dir = Path(__file__).parent
metrics_dir = script_dir.parent.parent
os.chdir(metrics_dir)

# Add the metrics_modernization directory to Python path
if str(metrics_dir) not in sys.path:
    sys.path.insert(0, str(metrics_dir))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from common.cli_utils import (
    create_metric_parser,
    get_output_filename, 
    print_evaluation_summary,
    validate_common_args,
)
from common.config import Config
from common.data_loader import DataLoader
from common.result_saver import ResultSaver

load_dotenv()


class FaithfulnessResponse(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str


async def evaluate_sample_faithfulness(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = Config.DEFAULT_MODEL
) -> Dict[str, Any]:
    prompt = f"""Given a question, answer, and retrieved contexts, evaluate how faithful the answer is to the provided contexts.

Question: {sample["question"]}
Answer: {sample["answer"]}  
Retrieved Contexts: {sample.get("contexts", "")}

Evaluate faithfulness on a scale from 0 to 1 where:
- 0: Answer contains information not supported by contexts
- 1: Answer is completely faithful to contexts

Provide JSON with 'value' (float 0-1) and 'reason' (string)."""

    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=FaithfulnessResponse,
            temperature=0.1,
        )
        result = response.choices[0].message.parsed
        return {
            "sample": sample,
            "faithfulness_score": result.value,
            "reasoning": result.reason,
            "success": True,
        }
    except Exception as e:
        return {
            "sample": sample,
            "faithfulness_score": None,
            "reasoning": str(e),
            "success": False,
        }


async def evaluate_faithfulness_modern_simplified(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = Config.DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_EVALUATIONS)
    
    async def evaluate_with_limit(sample):
        async with semaphore:
            return await evaluate_sample_faithfulness(client, sample, model)
    
    tasks = [evaluate_with_limit(sample) for sample in data]
    return await asyncio.gather(*tasks)


async def main():
    parser = create_metric_parser("faithfulness", "modern_simplified")
    args = parser.parse_args()
    validate_common_args(args)

    api_key = Config.get_openai_api_key()
    client = AsyncOpenAI(api_key=api_key)
    data, dataset_display_name, total_samples = DataLoader.load_from_preprocessed_file(args.data_file)
    results = await evaluate_faithfulness_modern_simplified(data, client, args.model)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = get_output_filename(args.dataset, "modern_simplified", "faithfulness")
    output_path = results_dir / output_filename
    
    ResultSaver.save_modern_results(
        results=results,
        metric="faithfulness",
        dataset_name=dataset_display_name,
        output_path=str(output_path),
        implementation="modern_simplified"
    )

    valid_results = [r for r in results if r["success"] and r["faithfulness_score"] is not None]
    if valid_results:
        scores = [r["faithfulness_score"] for r in valid_results]
        average_score = sum(scores) / len(scores)
        
        print_evaluation_summary(
            dataset_name=dataset_display_name,
            metric="faithfulness",
            implementation="modern_simplified",
            average_score=average_score,
            num_successful=len(valid_results),
            total_samples=total_samples,
            output_path=str(output_path)
        )


if __name__ == "__main__":
    asyncio.run(main())