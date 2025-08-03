# LongBench v2 Integration

Minimal integration of LongBench v2 evaluation into LlamaFactory.

## Quick Start

### Option 1: Use Local Dataset (Recommended)
```bash
# Setup symlink to your existing dataset
./setup_longbench.sh

# Or specify custom path
./setup_longbench.sh /your/custom/path/to/LongBench-v2

# Run evaluation with local dataset
llamafactory-cli eval longbench_local.yaml
```

### Option 2: Download from HuggingFace
```bash
# Run LongBench evaluation (requires internet)
llamafactory-cli eval longbench_test.yaml
```

## Configuration

```yaml
# Basic settings
model_name_or_path: your-model
task: longbench
save_dir: saves/longbench

# LongBench settings
longbench_max_length: 32768   # Max context length
longbench_max_samples: 10     # Number of samples to evaluate
longbench_use_local: true     # Use local dataset (requires setup)
```

## Setup Script Usage

```bash
# Use default path (/exp/data/eval_data/LongBench-v2)
./setup_longbench.sh

# Use custom path
./setup_longbench.sh /path/to/your/LongBench-v2

# Show help
./setup_longbench.sh --help
```

## Features

- ✅ Simple, minimal implementation
- ✅ Automatic context truncation (middle strategy)
- ✅ Support for all LongBench v2 questions
- ✅ HuggingFace dataset integration
- ✅ Compatible with existing LlamaFactory models

## Results

Results are saved in the specified `save_dir`:
- `predictions.jsonl`: Detailed predictions for each sample
- `summary.json`: Overall accuracy and metrics

## Implementation Details

The evaluator:
1. Loads the model using LlamaFactory's standard pipeline
2. Downloads LongBench v2 from HuggingFace
3. Truncates long contexts using middle truncation
4. Generates answers and extracts A/B/C/D choices
5. Calculates accuracy and saves results

This is a clean, minimal implementation focused on simplicity and reliability.