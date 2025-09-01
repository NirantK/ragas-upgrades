"""
Simplified faithfulness evaluation using direct OpenAI calls (modern simplified approach)
Supports both AmnestyQA and FIQA datasets via CLI

Refactored to use shared utilities for better maintainability.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger
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

# Load environment variables
load_dotenv()


class FaithfulnessResponse(BaseModel):
    """Response model for faithfulness evaluation"""

    value: float = Field(
        ..., description="Faithfulness score between 0 and 1", ge=0.0, le=1.0
    )
    reason: str = Field(..., description="Reasoning for the faithfulness score")


async def evaluate_sample_faithfulness(
    client: AsyncOpenAI, sample: Dict[str, Any], model: str = Config.DEFAULT_MODEL
) -> Dict[str, Any]:
    """Evaluate faithfulness for a single sample using OpenAI"""

    prompt = f"""
Given a question, answer, and retrieved contexts, evaluate how faithful the answer is to the provided contexts.

Faithfulness measures whether the answer can be inferred from the given contexts. A highly faithful answer contains only information that can be verified from the contexts.

Question: {sample["question"]}
Answer: {sample["answer"]}
Retrieved Contexts: {sample.get("contexts", "")}

Please evaluate the faithfulness of the answer on a scale from 0 to 1, where:
- 0: Answer contains information not supported by contexts or contradicts them
- 1: Answer is completely faithful and all information can be verified from contexts

Provide your evaluation as a JSON object with 'value' (float between 0 and 1) and 'reason' (string explanation).
"""

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
        logger.warning(f"Failed to evaluate sample: {e}")
        return {
            "sample": sample,
            "faithfulness_score": None,
            "reasoning": str(e),
            "success": False,
        }


async def evaluate_faithfulness_modern_simplified(
    data: List[Dict[str, Any]], client: AsyncOpenAI, model: str = Config.DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    """Evaluate faithfulness using modern simplified approach"""
    logger.info("Starting faithfulness evaluation with modern simplified approach...")

    # Process samples with controlled concurrency
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_EVALUATIONS)
    
    async def evaluate_with_limit(sample):
        async with semaphore:
            return await evaluate_sample_faithfulness(client, sample, model)
    
    tasks = [evaluate_with_limit(sample) for sample in data]
    results = await asyncio.gather(*tasks)

    logger.info("Faithfulness evaluation completed")
    return results


async def main():
    """Main execution function"""
    # Use shared CLI utilities (model argument already included for modern implementations)
    parser = create_metric_parser("faithfulness", "modern_simplified")
    args = parser.parse_args()
    
    # Validate common arguments
    validate_common_args(args)

    logger.info(
        f"Starting {args.dataset.upper()} faithfulness evaluation with Modern Simplified Approach"
    )

    try:
        # Get OpenAI API key from config
        api_key = Config.get_openai_api_key()
        client = AsyncOpenAI(api_key=api_key)

        # Load data using shared utility
        data, dataset_display_name, total_samples = DataLoader.load_from_preprocessed_file(args.data_file)

        # Evaluate faithfulness
        results = await evaluate_faithfulness_modern_simplified(data, client, args.model)

        # Save results using shared utility
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

        # Print summary using shared utility
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

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())