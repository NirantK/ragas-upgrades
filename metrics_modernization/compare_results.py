"""Universal comparison tool for evaluation results"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Change to the metrics_modernization directory to enable relative imports
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Add the current directory to Python path
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from common.config import Config
from common.exceptions import DataLoadError
from common.result_saver import ResultSaver


class ResultsComparator:
    def __init__(self, metric: str):
        self.metric = metric
        self.results_dir = Path(__file__).parent / metric / "results"
        self.metric_field = Config.get_metric_field_name(metric)
    
    def load_all_results(self) -> Dict[str, Any]:
        results = {}
        files = [
            ("amnesty_ragas_main.json", "AmnestyQA", "Ragas Main"),
            ("amnesty_modern_simplified.json", "AmnestyQA", "Modern Simplified"),
            ("amnesty_modern_exact_replica.json", "AmnestyQA", "Modern Exact Replica"),
            ("fiqa_ragas_main.json", "FIQA", "Ragas Main"),
            ("fiqa_modern_simplified.json", "FIQA", "Modern Simplified"),
            ("fiqa_modern_exact_replica.json", "FIQA", "Modern Exact Replica"),
        ]
        
        for filename, dataset, framework in files:
            dataset_subdir = "amnesty" if "amnesty" in filename else "fiqa"
            filepath = self.results_dir / dataset_subdir / filename
            
            key = f"{dataset}_{framework}"
            if filepath.exists():
                try:
                    results[key] = ResultSaver.load_results(str(filepath))
                except DataLoadError:
                    continue
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comparison summary"""
        print("=" * 80)
        print(f"{self.metric.upper().replace('_', ' ')} EVALUATION RESULTS COMPARISON")
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
                    avg_score = data.get(self.metric_field, "N/A")
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
    
    def print_detailed_comparison(self, results: Dict[str, Any]):
        """Print detailed score comparison"""
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON")
        print("=" * 80)

        datasets = ["AmnestyQA", "FIQA"]

        for dataset in datasets:
            main_key = f"{dataset}_Ragas Main"
            simplified_key = f"{dataset}_Modern Simplified"
            exact_key = f"{dataset}_Modern Exact Replica"

            # Compare Main vs Simplified
            if main_key in results and simplified_key in results:
                self._compare_implementations(
                    results[main_key], 
                    results[simplified_key],
                    dataset,
                    "Ragas Main",
                    "Modern Simplified"
                )

            # Compare Main vs Exact Replica
            if main_key in results and exact_key in results:
                self._compare_implementations(
                    results[main_key], 
                    results[exact_key],
                    dataset,
                    "Ragas Main", 
                    "Modern Exact Replica",
                    exact_match_check=True
                )
    
    def _compare_implementations(
        self, 
        main_data: Dict[str, Any],
        other_data: Dict[str, Any], 
        dataset: str,
        main_label: str,
        other_label: str,
        exact_match_check: bool = False
    ):
        """Compare two implementations"""
        main_scores = main_data.get("scores", [])
        other_scores = other_data.get("scores", [])

        if not main_scores or not other_scores:
            print(f"\n{dataset} - No scores available for comparison")
            return

        print(f"\n{dataset} - Score Comparison ({main_label} vs {other_label}):")
        
        # Adjust header width based on label length
        other_width = max(len(other_label), 12)
        print(f"  {'Sample':<7} | {'Ragas Main':<11} | {other_label:<{other_width}} | Difference")
        print(f"  {'-' * 7}|{'-' * 13}|{'-' * (other_width + 2)}|{'-' * 10}")

        # Compare sample by sample
        for i, (main_score, other_score) in enumerate(zip(main_scores, other_scores)):
            diff = (
                other_score - main_score
                if isinstance(main_score, (int, float))
                and isinstance(other_score, (int, float))
                else "N/A"
            )
            diff_str = f"{diff:+.4f}" if isinstance(diff, (int, float)) else diff
            print(
                f"  {i + 1:6d}  | {main_score:10.4f} | {other_score:{other_width}.4f} | {diff_str:>10}"
            )

        # Calculate averages
        if main_scores and other_scores:
            main_avg = sum(main_scores) / len(main_scores)
            other_avg = sum(other_scores) / len(other_scores)
            avg_diff = other_avg - main_avg
            
            print(f"  {'-' * 7}|{'-' * 13}|{'-' * (other_width + 2)}|{'-' * 10}")
            print(
                f"  Average | {main_avg:10.4f} | {other_avg:{other_width}.4f} | {avg_diff:+10.4f}"
            )
            
            # Exact match validation for replica
            if exact_match_check:
                abs_diff = abs(avg_diff)
                target_met = "✓ ACHIEVED" if abs_diff < 0.01 else "✗ TARGET NOT MET"
                print(f"  Abs Diff: {abs_diff:.4f} (<0.01 target) - {target_met}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results for different metrics"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=list(Config.METRIC_FIELD_NAMES.keys()),
        required=True,
        help=f"Metric to compare ({', '.join(Config.METRIC_FIELD_NAMES.keys())})"
    )

    args = parser.parse_args()

    # Create comparator for the metric
    comparator = ResultsComparator(args.metric)
    
    # Load results
    results = comparator.load_all_results()

    if not results:
        print(f"No results found for {args.metric}. Make sure evaluations have been run.")
        return

    # Print comparison
    comparator.print_summary(results)
    comparator.print_detailed_comparison(results)

    print(f"\n\n{args.metric.replace('_', ' ').title()} results loaded from:")
    for key in results.keys():
        print(f"  - {key}")


if __name__ == "__main__":
    main()