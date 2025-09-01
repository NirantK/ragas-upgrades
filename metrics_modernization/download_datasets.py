"""
Download HuggingFace datasets and create proper data files with ground truth
"""

import json
import logging
import os
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_process_dataset(dataset_name: str, dataset_config: str = None, split: str = "eval", limit: int = None):
    """Download and process dataset from HuggingFace"""
    logger.info(f"Loading {dataset_name} dataset...")
    
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
            
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        logger.info(f"Dataset features: {dataset.features}")
        
        # Show first sample to understand structure
        if len(dataset) > 0:
            logger.info(f"Sample data structure:")
            sample = dataset[0]
            for key, value in sample.items():
                logger.info(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None


def create_data_file(dataset, dataset_name: str, output_path: str):
    """Create normalized data file with proper fields"""
    
    data = []
    for sample in dataset:
        # Try to identify the field mappings
        processed_sample = {}
        
        # Question field
        for question_field in ['question', 'query', 'input']:
            if question_field in sample:
                processed_sample['question'] = sample[question_field]
                break
        
        # Answer field (generated answer)
        for answer_field in ['answer', 'response', 'prediction', 'generated_text']:
            if answer_field in sample:
                processed_sample['answer'] = sample[answer_field]
                break
                
        # Ground truth field
        for gt_field in ['ground_truth', 'ground_truths', 'reference', 'references', 'target', 'correct_answer']:
            if gt_field in sample:
                gt_value = sample[gt_field]
                # Handle list vs string
                if isinstance(gt_value, list) and len(gt_value) > 0:
                    processed_sample['ground_truth'] = gt_value[0] if isinstance(gt_value[0], str) else str(gt_value[0])
                else:
                    processed_sample['ground_truth'] = str(gt_value)
                break
        
        # Context field
        for context_field in ['contexts', 'context', 'retrieved_contexts', 'passages']:
            if context_field in sample:
                context_value = sample[context_field]
                if isinstance(context_value, list):
                    processed_sample['contexts'] = ' '.join(context_value)
                else:
                    processed_sample['contexts'] = str(context_value)
                break
        
        # Only add if we have the essential fields
        if 'question' in processed_sample and 'answer' in processed_sample:
            data.append(processed_sample)
    
    # Create output structure
    output_data = {
        "dataset_name": dataset_name,
        "total_samples": len(data),
        "data": data
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} samples to {output_path}")
    return output_data


def main():
    """Main function to download datasets"""
    import os
    
    # AmnestyQA dataset
    logger.info("=" * 60)
    logger.info("Downloading AmnestyQA dataset")
    logger.info("=" * 60)
    
    amnesty_dataset = download_and_process_dataset("explodinggradients/amnesty_qa", split="eval", limit=20)
    if amnesty_dataset:
        output_path = "metrics_modernization/answer_correctness/data/amnesty_data.json"
        create_data_file(amnesty_dataset, "AmnestyQA", output_path)
    
    logger.info("=" * 60)
    logger.info("Downloading FIQA dataset")  
    logger.info("=" * 60)
    
    # FIQA dataset - try different configurations
    fiqa_dataset = None
    fiqa_configs = [
        ("explodinggradients/fiqa", None),
        ("ragas/fiqa", None),
        ("financial_phrasebank", None)
    ]
    
    for dataset_name, config in fiqa_configs:
        logger.info(f"Trying {dataset_name}...")
        fiqa_dataset = download_and_process_dataset(dataset_name, config, split="eval", limit=30)
        if fiqa_dataset:
            break
    
    if fiqa_dataset:
        output_path = "metrics_modernization/answer_correctness/data/fiqa_data.json"  
        create_data_file(fiqa_dataset, "FIQA", output_path)
    else:
        logger.error("Could not load FIQA dataset from any source")


if __name__ == "__main__":
    main()