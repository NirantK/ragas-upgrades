# Metrics Modernization

A unified evaluation framework for running faithfulness metrics across multiple implementations and datasets with centralized data loading.

## Key Features

- **Centralized Data Loading**: Single source of truth for dataset loading logic
- **Full Dataset Processing**: Uses complete datasets by default (20 samples for AmnestyQA, 30 for FIQA) instead of limiting to 10 samples
- **Optional Sample Limiting**: Test with smaller subsets using `--limit-samples` when needed
- **Unified CLI**: Run all implementations through a single interface
- **Concurrent Execution**: Parallel processing for faster evaluations

## Usage

### Basic Usage (Full Datasets)

```bash
# Run all faithfulness implementations on all datasets (recommended)
uv run python run_evaluation.py --metrics faithfulness

# Run on specific dataset only
uv run python run_evaluation.py --metrics faithfulness --datasets amnesty

# Run specific implementation only  
uv run python run_evaluation.py --metrics faithfulness --implementations modern_simplified
```

### Limited Sample Testing

```bash
# Test with 5 samples for quick validation
uv run python run_evaluation.py --metrics faithfulness --limit-samples 5

# Test with 2 samples on single dataset
uv run python run_evaluation.py --metrics faithfulness --datasets fiqa --limit-samples 2
```

### Other Options

```bash
# List all available implementations
uv run python run_evaluation.py --list-implementations

# Save detailed results report
uv run python run_evaluation.py --metrics faithfulness --output-summary results_summary.json

# Control parallel workers
uv run python run_evaluation.py --metrics faithfulness --workers 2
```

## Default Behavior Changes

### Before (Old System)
- **Sample Limit**: Hard-coded to 10 samples per dataset
- **CLI Requirement**: `--samples` parameter was required
- **Data Loading**: Each implementation loaded data independently
- **Inconsistency**: Different data loading logic across files

### After (New System)  
- **Full Datasets**: Uses complete datasets by default (20-30 samples)
- **Optional Limiting**: `--limit-samples` is optional for testing only
- **Centralized Loading**: Single data loader ensures consistency
- **Better Performance**: Process all available data for accurate results

## Available Implementations

- **modern_simplified**: Direct OpenAI calls with simplified prompting
- **current_ragas_main**: Official Ragas library implementation
- **modern_exact_replica**: Exact replica of Ragas main using two-step process

## Available Datasets  

- **amnesty**: AmnestyQA dataset (20 samples in eval split)
- **fiqa**: FIQA dataset (30 samples in baseline split)

## Architecture

The system uses a centralized data loading approach:

1. **Centralized Loader**: `load_dataset_by_name()` in `run_evaluation.py` handles all data loading
2. **Data Normalization**: Converts different dataset formats to consistent structure
3. **Temporary Files**: Preprocessed data passed to implementations via JSON files
4. **Implementation Updates**: All implementations now accept `--data-file` instead of loading data themselves

This ensures consistency, reduces duplication, and makes it easier to add new datasets or modify existing ones.