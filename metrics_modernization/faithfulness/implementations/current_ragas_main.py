"""Faithfulness evaluation using Ragas Main framework"""

import os
import sys
from pathlib import Path
from typing import Any

# Change to the metrics_modernization directory to enable relative imports
script_dir = Path(__file__).parent
metrics_dir = script_dir.parent.parent
os.chdir(metrics_dir)

# Add the metrics_modernization directory to Python path
if str(metrics_dir) not in sys.path:
    sys.path.insert(0, str(metrics_dir))

from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness

from common.cli_utils import create_metric_parser, print_evaluation_summary, get_output_filename, validate_common_args
from common.data_loader import DataLoader
from common.exceptions import EvaluationError
from common.result_saver import ResultSaver

load_dotenv()


def load_preprocessed_data_to_ragas(data_file_path: str) -> tuple[Any, str, int]:
    data, dataset_name, total_samples = DataLoader.load_from_preprocessed_file(data_file_path)
    eval_dataset = DataLoader.convert_to_ragas_format(data)
    return eval_dataset, dataset_name, total_samples


def evaluate_faithfulness(eval_dataset: Any) -> Any:
    try:
        result = evaluate(eval_dataset, metrics=[faithfulness], show_progress=True)
        return result
    except Exception as e:
        raise EvaluationError("faithfulness", "ragas_main", str(e))


def save_results(result: Any, dataset_display_name: str, output_path: str):
    ResultSaver.save_ragas_results(
        result=result,
        metric="faithfulness", 
        dataset_name=dataset_display_name,
        output_path=output_path,
        implementation="ragas_main"
    )


def main():
    parser = create_metric_parser("faithfulness", "ragas_main")
    args = parser.parse_args()
    validate_common_args(args)

    eval_dataset, dataset_display_name, total_samples = load_preprocessed_data_to_ragas(args.data_file)
    result = evaluate_faithfulness(eval_dataset)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = get_output_filename(args.dataset, "ragas_main", "faithfulness")
    output_path = results_dir / output_filename
    save_results(result, dataset_display_name, str(output_path))

    result_dict = result.to_pandas() if hasattr(result, "to_pandas") else result
    if "faithfulness" in result_dict:
        average_score = result_dict["faithfulness"].mean()
        print_evaluation_summary(
            dataset_name=dataset_display_name,
            metric="faithfulness",
            implementation="ragas_main",
            average_score=average_score,
            num_successful=total_samples,
            total_samples=total_samples,
            output_path=str(output_path)
        )


if __name__ == "__main__":
    main()