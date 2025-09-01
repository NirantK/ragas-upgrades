"""
Extract datasets with proper ground truth mapping
"""

import json
import os
from datasets import load_dataset

def process_amnesty_dataset():
    """Process AmnestyQA dataset"""
    print("Loading AmnestyQA dataset...")
    
    try:
        dataset = load_dataset("explodinggradients/amnesty_qa", split="eval")
        print(f"Loaded {len(dataset)} samples")
        
        # Show structure
        print("Fields:", dataset.features)
        sample = dataset[0]
        print("\nSample structure:")
        for k, v in sample.items():
            print(f"  {k}: {str(v)[:100]}...")
        
        # Process data
        data = []
        for sample in dataset:
            processed = {
                "question": sample["user_input"],  # Question
                "answer": sample["response"],      # Generated answer
                "ground_truth": sample["reference"], # Ground truth reference
                "contexts": " ".join(sample["retrieved_contexts"]) if sample.get("retrieved_contexts") else ""
            }
            data.append(processed)
        
        # Save
        output = {
            "dataset_name": "AmnestyQA",
            "total_samples": len(data),
            "data": data
        }
        
        os.makedirs("answer_correctness/data", exist_ok=True)
        with open("answer_correctness/data/amnesty_data.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved {len(data)} samples with ground truth")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def process_fiqa_dataset():
    """Process FIQA dataset"""
    print("\nLoading FIQA dataset...")
    
    try:
        # Try different splits
        for split in ["baseline", "train", "eval"]:
            try:
                print(f"Trying split: {split}")
                dataset = load_dataset("explodinggradients/fiqa", split=split)
                print(f"Loaded {len(dataset)} samples from split '{split}'")
                
                # Show structure  
                print("Fields:", dataset.features)
                if len(dataset) > 0:
                    sample = dataset[0]
                    print("\nSample structure:")
                    for k, v in sample.items():
                        print(f"  {k}: {str(v)[:100]}...")
                
                # Take first 30 samples
                dataset = dataset.select(range(min(30, len(dataset))))
                
                # Process data - need to figure out field mapping
                data = []
                for sample in dataset:
                    # Try to map fields based on what we find
                    processed = {}
                    
                    # Question
                    if "question" in sample:
                        processed["question"] = sample["question"]
                    elif "user_input" in sample:
                        processed["question"] = sample["user_input"]
                    elif "query" in sample:
                        processed["question"] = sample["query"]
                    
                    # Answer (generated)
                    if "answer" in sample:
                        processed["answer"] = sample["answer"]
                    elif "response" in sample:
                        processed["answer"] = sample["response"]
                    elif "prediction" in sample:
                        processed["answer"] = sample["prediction"]
                    
                    # Ground truth
                    if "ground_truth" in sample:
                        processed["ground_truth"] = sample["ground_truth"]
                    elif "reference" in sample:
                        processed["ground_truth"] = sample["reference"]
                    elif "correct_answer" in sample:
                        processed["ground_truth"] = sample["correct_answer"]
                    
                    # Contexts
                    if "contexts" in sample:
                        if isinstance(sample["contexts"], list):
                            processed["contexts"] = " ".join(sample["contexts"])
                        else:
                            processed["contexts"] = str(sample["contexts"])
                    elif "retrieved_contexts" in sample:
                        if isinstance(sample["retrieved_contexts"], list):
                            processed["contexts"] = " ".join(sample["retrieved_contexts"])
                        else:
                            processed["contexts"] = str(sample["retrieved_contexts"])
                    
                    # Only include if we have question and answer
                    if "question" in processed and "answer" in processed:
                        data.append(processed)
                
                if data:
                    # Save
                    output = {
                        "dataset_name": "FIQA",
                        "total_samples": len(data),
                        "data": data
                    }
                    
                    with open("answer_correctness/data/fiqa_data.json", "w") as f:
                        json.dump(output, f, indent=2)
                    
                    print(f"Saved {len(data)} samples")
                    return True
                    
            except Exception as e:
                print(f"Failed with split {split}: {e}")
                continue
                
        print("Could not load FIQA dataset from any split")
        return False
        
    except Exception as e:
        print(f"Error loading FIQA: {e}")
        return False

if __name__ == "__main__":
    success = process_amnesty_dataset()
    if success:
        process_fiqa_dataset()