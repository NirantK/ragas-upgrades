# Plan: Centralize Data Loading and Remove Sampling

## Current State Analysis

### Issues Identified
1. **Duplicated Data Loading Logic**: Each implementation has its own `load_dataset_by_name()` function
2. **Hard-coded Sampling**: All implementations default to `num_samples: int = 10` 
3. **Inconsistent Dataset Handling**: Each file handles dataset loading differently
4. **CLI Sampling Dependency**: CLI requires `--samples` parameter, forcing artificial limits

### Current Dataset Configurations
- **AmnestyQA**: `"explodinggradients/amnesty_qa", "english_v3"` → 20 samples in eval split
- **FIQA**: `"explodinggradients/fiqa", "ragas_eval"` → 30 samples in baseline split
- **No Malayalam/other configs found**: Only english_v3 is available in cache

## Plan Implementation

### 1. Centralize Data Loading in `run_evaluation.py`

**Create centralized data loader:**
```python
def load_dataset_by_name(
    dataset_name: str, 
    num_samples: Optional[int] = None, 
    random_seed: int = 42
) -> Dict[str, Any]:
    """Centralized dataset loading with optional sampling"""
    
    if dataset_name.lower() == "amnesty":
        hf_dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")["eval"]
        dataset_display_name = "AmnestyQA"
    elif dataset_name.lower() == "fiqa":
        hf_dataset = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]
        dataset_display_name = "FIQA"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Use full dataset by default, sample only if specified
    if num_samples is None:
        selected_samples = hf_dataset
        logger.info(f"Loading complete {dataset_display_name} dataset: {len(selected_samples)} samples")
    else:
        max_samples = min(num_samples, len(hf_dataset))
        selected_samples = hf_dataset.shuffle(seed=random_seed).select(range(max_samples))
        logger.info(f"Loading {dataset_display_name} dataset: {max_samples}/{len(hf_dataset)} samples")
    
    # Normalize data format for all implementations
    data = []
    for sample in selected_samples:
        if dataset_name.lower() == "fiqa":
            data.append({
                "question": sample["question"],
                "answer": sample["answer"], 
                "contexts": "\n".join(sample["contexts"])
            })
        else:  # amnesty
            data.append({
                "question": sample["user_input"],
                "answer": sample["response"],
                "contexts": "\n".join(sample["retrieved_contexts"])
            })
    
    return {
        "data": data,
        "dataset_name": dataset_display_name,
        "total_samples": len(selected_samples)
    }
```

### 2. Modify CLI Interface

**Remove sampling from required parameters:**
- Remove default `--samples 10` from Click options
- Add optional `--limit-samples` for testing with subsets
- Default behavior: process complete datasets (100%)

**Updated Click interface:**
```python
@click.option(
    "--limit-samples",
    type=int,
    help="Limit number of samples for testing (optional). Uses full dataset by default."
)
```

### 3. Update Implementation Files

**Files to modify:**
- `faithfulness/implementations/current_ragas_main.py`
- `faithfulness/implementations/modern_simplified.py`
- `faithfulness/implementations/modern_exact_replica.py`

**Changes per file:**
1. **Remove data loading functions**: Delete `load_dataset_by_name()` from each implementation
2. **Update CLI arguments**: Remove `--samples` argument, add `--limit-samples`
3. **Modify main functions**: Accept pre-loaded data instead of dataset names
4. **Update function signatures**: Change to receive processed data directly

**Example updated main function:**
```python
def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-file", required=True, help="Path to preprocessed data JSON")
    parser.add_argument("--output-dir", default="results")
    
    args = parser.parse_args()
    
    # Load preprocessed data from centralized loader
    with open(args.data_file, 'r') as f:
        dataset_info = json.load(f)
    
    data = dataset_info["data"]
    dataset_display_name = dataset_info["dataset_name"]
    # ... rest of evaluation logic
```

### 4. Update Unified Driver Logic

**Modify `UnifiedEvaluationDriver.execute_job()`:**
1. **Pre-load data**: Use centralized loader before running implementations
2. **Pass data to implementations**: Write temporary JSON files with preprocessed data
3. **Update command construction**: Pass data file path instead of dataset parameters

**Updated job execution flow:**
```python
def execute_job(self, job: EvaluationJob) -> EvaluationResult:
    # 1. Load data centrally
    dataset_info = load_dataset_by_name(
        job.dataset,
        num_samples=job.samples if hasattr(job, 'samples') and job.samples else None,
        random_seed=job.seed
    )
    
    # 2. Write temporary data file
    temp_data_file = Path(job.output_dir) / f"{job.dataset}_data.json"
    with open(temp_data_file, 'w') as f:
        json.dump(dataset_info, f)
    
    # 3. Build command with data file
    cmd = [
        sys.executable,
        job.implementation_info.file_path,
        "--dataset", job.dataset,
        "--data-file", str(temp_data_file),
        "--output-dir", job.output_dir,
    ]
    # ... rest of execution
```

### 5. Update Documentation

**Update README.md:**
- Document new default behavior (full dataset processing)
- Update CLI examples to remove `--samples`
- Add `--limit-samples` option documentation
- Update architecture section to mention centralized data loading

## Expected Benefits

### 1. Consistency
- Single source of truth for dataset loading logic
- Consistent data format across all implementations
- Unified error handling and logging

### 2. Maintainability
- Remove code duplication across 3+ files
- Single place to add new datasets or modify existing ones
- Easier to update dataset configurations

### 3. Performance
- Process complete datasets by default (more accurate results)
- Optional sampling only for testing/development
- Better resource utilization

### 4. User Experience
- Simplified CLI: `uv run run_evaluation.py --metrics faithfulness --datasets amnesty`
- No need to specify sample counts for full evaluation
- Optional `--limit-samples 5` for quick testing

## Migration Path

1. **Phase 1**: Implement centralized data loader in `run_evaluation.py`
2. **Phase 2**: Update CLI interface to remove sampling requirement
3. **Phase 3**: Modify all implementation files to accept preprocessed data
4. **Phase 4**: Update driver to use new data flow
5. **Phase 5**: Test with existing implementations
6. **Phase 6**: Update documentation and examples

## Validation

**Test cases to verify:**
- Full dataset evaluation: `uv run run_evaluation.py --metrics faithfulness --datasets amnesty`
- Limited sampling: `uv run run_evaluation.py --metrics faithfulness --datasets amnesty --limit-samples 5`
- Multi-dataset: `uv run run_evaluation.py --metrics faithfulness --datasets amnesty --datasets fiqa`
- All implementations work with centralized data loading