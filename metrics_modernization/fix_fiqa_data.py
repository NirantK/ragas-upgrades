"""
Fix FIQA dataset extraction to include ground truth
"""

import json
import os
from datasets import load_dataset

def fix_fiqa_dataset():
    """Fix FIQA dataset with proper ground truth"""
    print("Loading FIQA dataset...")
    
    dataset = load_dataset("explodinggradients/fiqa", split="baseline")
    dataset = dataset.select(range(min(30, len(dataset))))
    print(f"Loaded {len(dataset)} samples")
    
    # Show structure
    sample = dataset[0]
    print("Fields:", sample.keys())
    print("Ground truths sample:", sample["ground_truths"][0][:100])
    
    # Process data  
    data = []
    for sample in dataset:
        processed = {
            "question": sample["question"],
            "answer": sample["answer"],
            "ground_truth": sample["ground_truths"][0] if sample["ground_truths"] else "",  # Take first ground truth
            "contexts": " ".join(sample["contexts"]) if sample["contexts"] else ""
        }
        data.append(processed)
    
    # Save
    output = {
        "dataset_name": "FIQA",
        "total_samples": len(data),
        "data": data
    }
    
    with open("answer_correctness/data/fiqa_data.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Fixed and saved {len(data)} samples with ground truth")

if __name__ == "__main__":
    fix_fiqa_dataset()