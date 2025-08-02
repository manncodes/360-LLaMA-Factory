# HELMET Long Context Benchmark Integration

This document provides comprehensive guidance for using the HELMET (How to Evaluate Long-context Models Effectively and Thoroughly) benchmark within LlamaFactory.

## Overview

HELMET is a comprehensive benchmark for long-context language models covering **seven diverse task categories**:

1. **Recall** - JSON KV retrieval, NIAH variants (multikey, multivalue)
2. **RAG** - Retrieval-Augmented Generation tasks
3. **Passage Re-ranking** - Document ranking and retrieval
4. **Citation** - ALCE citation generation tasks
5. **Long QA** - Long-form question answering
6. **Summarization** - Document summarization tasks
7. **In-Context Learning (ICL)** - Few-shot learning evaluation

**Reference**: [HELMET GitHub Repository](https://github.com/princeton-nlp/HELMET)

## Prerequisites

### 1. HELMET Repository Setup

First, ensure the HELMET repository is available:

```bash
# Clone HELMET repository (one level up from this project)
cd /path/to/long-context/
git clone https://github.com/princeton-nlp/HELMET.git
```

The directory structure should be:
```
long-context/
├── 360-LLaMA-Factory/  (this project)
└── HELMET/             (HELMET repository)
```

### 2. HELMET Data Download

Download HELMET benchmark data:

```bash
cd HELMET/
bash scripts/download_data.sh
```

This downloads ~34GB of preprocessed data to the `data/` directory.

### 3. Dependencies

Install HELMET dependencies:

```bash
cd HELMET/
pip install -r requirements.txt
pip install flash-attn  # For GPU acceleration
```

## Configuration

### Available HELMET Tasks

| Task Category | Task Names | Description |
|---------------|------------|-------------|
| **Recall** | `json_kv`, `ruler_niah_mk_2`, `ruler_niah_mk_3`, `ruler_niah_mv` | Key-value retrieval and multi-needle tasks |
| **RAG** | `rag` | Retrieval-augmented generation |
| **Re-ranking** | `rerank` | Passage re-ranking tasks |
| **Citation** | `cite`, `alce_nocite` | Citation generation and verification |
| **Long QA** | `longqa` | Long-form question answering |
| **Summarization** | `summ` | Document summarization |
| **ICL** | `icl` | In-context learning evaluation |

### Configuration Parameters

Add these parameters to your YAML configuration:

```yaml
# Basic LlamaFactory settings
model_name_or_path: your/model/path
task: helmet_[task_type]  # e.g., helmet_recall, helmet_comprehensive
save_dir: saves/helmet_evaluation

# HELMET-specific parameters
helmet_tasks: "json_kv,ruler_niah_s_2"  # Comma-separated task list
helmet_input_max_length: 131072         # Max input tokens (128k)
helmet_generation_max_length: 100       # Max generation tokens
helmet_shots: 2                         # ICL examples
helmet_max_test_samples: 100            # Limit test samples (null = all)
helmet_output_dir: "saves/helmet_results"
```

## Usage Examples

### 1. Quick Demo Evaluation

Test HELMET integration with a small subset:

```bash
llamafactory-cli eval helmet_quick_demo.yaml
```

**Configuration**: `helmet_quick_demo.yaml`
- Single task (`json_kv`)
- 10 test samples
- 8k context length
- Fast execution

### 2. Recall Tasks Evaluation

Evaluate memory and retrieval capabilities:

```bash
llamafactory-cli eval helmet_recall_config.yaml
```

**Configuration**: `helmet_recall_config.yaml`
- Tasks: JSON KV, NIAH multikey/multivalue
- 100 samples per task
- 131k context length

### 3. Long QA Evaluation

Evaluate long-form question answering:

```bash
llamafactory-cli eval helmet_longqa_config.yaml
```

**Configuration**: `helmet_longqa_config.yaml`
- Long QA tasks
- 256 max generation tokens
- 50 test samples

### 4. Comprehensive Benchmark

Run multiple HELMET task categories:

```bash
llamafactory-cli eval helmet_comprehensive_config.yaml
```

**Configuration**: `helmet_comprehensive_config.yaml`
- Multiple tasks: JSON KV, NIAH, RAG, Re-ranking
- 25 samples per task
- Full evaluation suite

### 5. Custom Task Configuration

Create custom HELMET evaluations:

```yaml
model_name_or_path: your/model
task: helmet_custom
save_dir: saves/custom_helmet_eval

# Select specific tasks
helmet_tasks: "rag,rerank,cite"
helmet_input_max_length: 65536
helmet_generation_max_length: 200
helmet_shots: 3
helmet_max_test_samples: 50
```

## Direct Python Execution

Run HELMET evaluation without CLI:

```python
from llamafactory.eval.helmet_evaluator import run_helmet_evaluation
from llamafactory.hparams import get_eval_args

# Load configuration
model_args, data_args, eval_args, finetuning_args = get_eval_args("helmet_config.yaml")

# Run evaluation
results = run_helmet_evaluation(model_args, data_args, eval_args, finetuning_args)

print(f"Overall metrics: {results['overall_metrics']}")
print(f"Successful tasks: {results['successful_tasks']}")
```

## Output and Results

### Result Files

HELMET evaluation generates multiple output files:

```
saves/helmet_evaluation/
├── helmet_evaluation_summary.json    # Comprehensive results
├── helmet_metrics.json              # Aggregated metrics
├── [task]_eval_[details].json       # Individual task results
└── [task]_eval_[details].json.score # Task-specific scores
```

### Metrics Interpretation

**Key Metrics by Task Category**:

| Task | Primary Metrics | Description |
|------|----------------|-------------|
| **Recall** | `exact_match`, `f1` | Exact retrieval accuracy |
| **RAG** | `em`, `f1`, `recall` | Answer quality and relevance |
| **Re-ranking** | `map`, `mrr`, `ndcg` | Ranking quality |
| **Citation** | `citation_precision`, `citation_recall` | Citation accuracy |
| **Long QA** | `rouge`, `bertscore` | Answer quality |
| **Summarization** | `rouge-1`, `rouge-2`, `rouge-l` | Summary quality |
| **ICL** | `accuracy` | Few-shot learning performance |

### Example Results

```json
{
  "overall_metrics": {
    "avg_exact_match": 0.78,
    "avg_f1": 0.82,
    "avg_rouge_l": 0.45
  },
  "successful_tasks": ["json_kv", "ruler_niah_s_2"],
  "task_results": {
    "json_kv": {
      "averaged_metrics": {
        "exact_match": 85.0,
        "f1": 87.3
      },
      "num_samples": 100
    }
  }
}
```

## Advanced Configuration

### Task-Specific Settings

Different tasks may require different configurations:

```yaml
# For memory-intensive tasks (Recall, NIAH)
helmet_input_max_length: 131072
helmet_generation_max_length: 50
helmet_shots: 2

# For generation tasks (Long QA, Summarization)
helmet_input_max_length: 65536
helmet_generation_max_length: 512
helmet_shots: 3

# For quick testing
helmet_max_test_samples: 10
helmet_input_max_length: 8192
```

### Multiple Length Evaluation

Evaluate across different context lengths:

```yaml
# Configure separate runs for different lengths
helmet_input_max_length: 32768   # Run 1: 32k tokens
# helmet_input_max_length: 65536  # Run 2: 64k tokens  
# helmet_input_max_length: 131072 # Run 3: 128k tokens
```

### Model-Specific Optimizations

```yaml
# For smaller models
helmet_max_test_samples: 50
helmet_input_max_length: 16384

# For API models
helmet_shots: 1  # Reduce costs
helmet_max_test_samples: 25

# For local large models
helmet_input_max_length: 131072
batch_size: 1    # Prevent OOM
```

## Troubleshooting

### Common Issues

1. **HELMET Repository Not Found**
   ```
   Error: HELMET directory not found
   ```
   **Solution**: Ensure HELMET is cloned in the correct location (one level up)

2. **Data Not Downloaded**
   ```
   Error: data/[task] directory not found
   ```
   **Solution**: Run `bash scripts/download_data.sh` in HELMET directory

3. **Memory Issues**
   ```
   Error: CUDA out of memory
   ```
   **Solution**: Reduce `helmet_input_max_length` or `batch_size`

4. **Missing Dependencies**
   ```
   Error: No module named 'flash_attn'
   ```
   **Solution**: Install HELMET requirements: `pip install -r HELMET/requirements.txt`

### Performance Optimization

- **Reduce samples**: Set `helmet_max_test_samples: 10` for quick testing
- **Shorter contexts**: Use `helmet_input_max_length: 8192` for faster evaluation
- **Fewer shots**: Set `helmet_shots: 1` to reduce memory usage
- **Single tasks**: Evaluate one task at a time instead of comprehensive mode

### Validation

Test your setup with the quick demo:

```bash
llamafactory-cli eval helmet_quick_demo.yaml
```

Expected output:
```
[INFO] Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
[INFO] Evaluating task: json_kv
[INFO] Completed task json_kv: {'exact_match': 75.0, 'f1': 78.5}
[INFO] HELMET evaluation completed successfully!
```

## Integration with Existing Workflows

### Combining with Other Evaluations

```bash
# Run needle haystack first
llamafactory-cli eval needle_haystack_config.yaml

# Then run HELMET
llamafactory-cli eval helmet_recall_config.yaml

# Compare results using plotting scripts
python plot_niah_comparison.py --experiments needle_haystack helmet_recall
```

### Batch Evaluation Script

```bash
#!/bin/bash
# evaluate_all_benchmarks.sh

echo "Running comprehensive long-context evaluation..."

# HELMET tasks
llamafactory-cli eval helmet_recall_config.yaml
llamafactory-cli eval helmet_longqa_config.yaml

# Custom needle haystack
llamafactory-cli eval needle_haystack_config.yaml

echo "Evaluation completed. Check saves/ directory for results."
```

## Comparison with Other Benchmarks

| Benchmark | Focus | Tasks | Context Length | LlamaFactory Integration |
|-----------|-------|-------|----------------|------------------------|
| **HELMET** | Comprehensive long-context | 7 categories, 15+ tasks | Up to 128k+ | ✅ Full integration |
| **Needle Haystack** | Information retrieval | Single task, configurable | Configurable | ✅ Built-in |
| **MMLU/CEVAL** | Knowledge & reasoning | Multiple choice | Standard | ✅ Built-in |

HELMET provides the most comprehensive evaluation for long-context capabilities, making it ideal for thorough model assessment.

## Citation

If you use HELMET in your research, please cite:

```bibtex
@inproceedings{yen2025helmet,
  title={HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly}, 
  author={Howard Yen and Tianyu Gao and Minmin Hou and Ke Ding and Daniel Fleischer and Peter Izsak and Moshe Wasserblat and Danqi Chen},
  year={2025},
  booktitle={International Conference on Learning Representations (ICLR)},
}
```