"""
Universal comparison tool for evaluation results (faithfulness, answer_relevance, etc.)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def load_results(metric: str) -> Dict[str, Any]:
    """Load all evaluation results for a specific metric"""
    results_dir = Path(__file__).parent / metric / "results"

    results = {}

    files = [
        (f"amnesty_ragas_main.json", "AmnestyQA", "Ragas Main"),
        (f"amnesty_modern_simplified.json", "AmnestyQA", "Modern Simplified"),
        (f"amnesty_modern_exact_replica.json", "AmnestyQA", "Modern Exact Replica"),
        (f"fiqa_ragas_main.json", "FIQA", "Ragas Main"),
        (f"fiqa_modern_simplified.json", "FIQA", "Modern Simplified"),
        (f"fiqa_modern_exact_replica.json", "FIQA", "Modern Exact Replica"),
    ]

    for filename, dataset, framework in files:
        # Determine subdirectory based on dataset
        dataset_subdir = "amnesty" if "amnesty" in filename else "fiqa"
        filepath = results_dir / dataset_subdir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
                results[f"{dataset}_{framework}"] = data
        else:
            print(f"Warning: {filepath} not found")

    return results


def get_metric_column_name(metric: str) -> str:
    """Get the appropriate column name for the metric"""
    metric_columns = {
        "faithfulness": "average_faithfulness",
        "answer_relevance": "average_answer_relevance",
        "answer_correctness": "average_answer_correctness",
    }
    return metric_columns.get(metric, f"average_{metric}")


def print_summary(results: Dict[str, Any], metric: str):
    """Print comparison summary"""
    metric_column = get_metric_column_name(metric)
    
    print("=" * 80)
    print(f"{metric.upper().replace('_', ' ')} EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    datasets = ["AmnestyQA", "FIQA"]
    frameworks = ["Ragas Main", "Modern Simplified", "Modern Exact Replica"]

    for dataset in datasets:
        print(f"\n{dataset} Dataset:")
        print("-" * 40)

        for framework in frameworks:
            key = f"{dataset}_{framework}"
            if key in results:
                data = results[key]
                avg_score = data.get(metric_column, "N/A")
                num_samples = data.get("num_samples", "N/A")
                num_successful = data.get("num_successful", num_samples)

                print(f"  {framework}:")
                if isinstance(avg_score, float):
                    print(f"    Average Score: {avg_score:.4f}")
                else:
                    print(f"    Average Score: {avg_score}")
                print(f"    Samples: {num_successful}/{num_samples}")
                print(f"    Timestamp: {data.get('timestamp', 'N/A')}")
            else:
                print(f"  {framework}: No data available")

    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)

    for dataset in datasets:
        main_key = f"{dataset}_Ragas Main"
        exp_key = f"{dataset}_Modern Simplified"
        exact_key = f"{dataset}_Modern Exact Replica"

        if main_key in results and exp_key in results:
            main_data = results[main_key]
            exp_data = results[exp_key]

            main_scores = main_data.get("scores", [])
            exp_scores = exp_data.get("scores", [])

            print(
                f"\n{dataset} - Score Comparison (Ragas Main vs Modern Simplified):"
            )
            print("  Sample  | Ragas Main | Modern Simplified | Difference")
            print("  --------|------------|-------------------|----------")

            for i, (main_score, exp_score) in enumerate(zip(main_scores, exp_scores)):
                diff = (
                    exp_score - main_score
                    if isinstance(main_score, (int, float))
                    and isinstance(exp_score, (int, float))
                    else "N/A"
                )
                diff_str = f"{diff:+.4f}" if isinstance(diff, (int, float)) else diff
                print(
                    f"  {i + 1:6d}  | {main_score:10.4f} | {exp_score:12.4f} | {diff_str:>10}"
                )

            if main_scores and exp_scores:
                main_avg = sum(main_scores) / len(main_scores)
                exp_avg = sum(exp_scores) / len(exp_scores)
                avg_diff = exp_avg - main_avg
                print("  --------|------------|-------------------|----------")
                print(
                    f"  Average | {main_avg:10.4f} | {exp_avg:12.4f} | {avg_diff:+10.4f}"
                )

        if main_key in results and exact_key in results:
            main_data = results[main_key]
            exact_data = results[exact_key]

            main_scores = main_data.get("scores", [])
            exact_scores = exact_data.get("scores", [])

            print(f"\n{dataset} - Score Comparison (Ragas Main vs Modern Exact Replica):")
            print("  Sample  | Ragas Main | Modern Exact Replica | Difference")
            print("  --------|------------|----------------------|----------")

            for i, (main_score, exact_score) in enumerate(
                zip(main_scores, exact_scores)
            ):
                diff = (
                    exact_score - main_score
                    if isinstance(main_score, (int, float))
                    and isinstance(exact_score, (int, float))
                    else "N/A"
                )
                diff_str = f"{diff:+.4f}" if isinstance(diff, (int, float)) else diff
                print(
                    f"  {i + 1:6d}  | {main_score:10.4f} | {exact_score:13.4f} | {diff_str:>10}"
                )

            if main_scores and exact_scores:
                main_avg = sum(main_scores) / len(main_scores)
                exact_avg = sum(exact_scores) / len(exact_scores)
                avg_diff = exact_avg - main_avg
                abs_diff = abs(avg_diff)
                target_met = "✓ ACHIEVED" if abs_diff < 0.01 else "✗ TARGET NOT MET"
                print("  --------|------------|----------------------|----------")
                print(
                    f"  Average | {main_avg:10.4f} | {exact_avg:13.4f} | {avg_diff:+10.4f}"
                )
                print(f"  Abs Diff: {abs_diff:.4f} (<0.01 target) - {target_met}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results for different metrics"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["faithfulness", "answer_relevance", "answer_correctness", "context_recall", "context_precision"],
        required=True,
        help="Metric to compare (faithfulness, answer_relevance, answer_correctness, context_recall, or context_precision)",
    )

    args = parser.parse_args()

    results = load_results(args.metric)

    if not results:
        print(f"No results found for {args.metric}. Make sure evaluations have been run.")
        return

    print_summary(results, args.metric)

    print(f"\n\n{args.metric.replace('_', ' ').title()} results loaded from:")
    for key in results.keys():
        print(f"  - {key}")


if __name__ == "__main__":
    main()