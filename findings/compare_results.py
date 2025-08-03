"""
Compare faithfulness evaluation results between Ragas Main and Experimental approaches
"""
import json
from pathlib import Path

def load_results():
    """Load all evaluation results"""
    results_dir = Path(__file__).parent / "results"
    
    results = {}
    
    files = [
        ("amnesty_ragas_main.json", "AmnestyQA", "Ragas Main"),
        ("amnesty_ragas_experimental.json", "AmnestyQA", "Experimental"),
        ("fiqa_ragas_main.json", "FIQA", "Ragas Main"),
        ("fiqa_ragas_experimental.json", "FIQA", "Experimental")
    ]
    
    for filename, dataset, framework in files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[f"{dataset}_{framework}"] = data
        else:
            print(f"Warning: {filename} not found")
    
    return results

def print_summary(results):
    """Print comparison summary"""
    print("=" * 80)
    print("FAITHFULNESS EVALUATION RESULTS COMPARISON")
    print("=" * 80)
    
    datasets = ["AmnestyQA", "FIQA"]
    frameworks = ["Ragas Main", "Experimental"]
    
    for dataset in datasets:
        print(f"\n{dataset} Dataset:")
        print("-" * 40)
        
        for framework in frameworks:
            key = f"{dataset}_{framework}"
            if key in results:
                data = results[key]
                avg_score = data.get("average_faithfulness", "N/A")
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
        exp_key = f"{dataset}_Experimental"
        
        if main_key in results and exp_key in results:
            main_data = results[main_key]
            exp_data = results[exp_key]
            
            main_scores = main_data.get("scores", [])
            exp_scores = exp_data.get("scores", [])
            
            print(f"\n{dataset} - Score Comparison:")
            print("  Sample  | Ragas Main | Experimental | Difference")
            print("  --------|------------|--------------|----------")
            
            for i, (main_score, exp_score) in enumerate(zip(main_scores, exp_scores)):
                diff = exp_score - main_score if isinstance(main_score, (int, float)) and isinstance(exp_score, (int, float)) else "N/A"
                diff_str = f"{diff:+.4f}" if isinstance(diff, (int, float)) else diff
                print(f"  {i+1:6d}  | {main_score:10.4f} | {exp_score:12.4f} | {diff_str:>10}")
            
            if main_scores and exp_scores:
                main_avg = sum(main_scores) / len(main_scores)
                exp_avg = sum(exp_scores) / len(exp_scores)
                avg_diff = exp_avg - main_avg
                print("  --------|------------|--------------|----------")
                print(f"  Average | {main_avg:10.4f} | {exp_avg:12.4f} | {avg_diff:+10.4f}")

def main():
    """Main function"""
    results = load_results()
    
    if not results:
        print("No results found. Make sure evaluations have been run.")
        return
    
    print_summary(results)
    
    print("\n\nResults loaded from:")
    for key in results.keys():
        print(f"  - {key}")

if __name__ == "__main__":
    main()