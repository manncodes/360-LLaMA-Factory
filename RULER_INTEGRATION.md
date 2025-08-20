# RULER Integration for LlamaFactory

This document describes the RULER (NVIDIA's comprehensive long-context evaluation benchmark) integration with LlamaFactory.

## Overview

RULER provides a comprehensive benchmark for evaluating language models on long-context tasks including:
- Needle-in-Haystack retrieval (single/multi-key/value/query)
- Variable Tracking
- Common Words Extraction
- Frequency Words Extraction
- Question Answering (SQuAD, HotpotQA)

## Installation

The RULER repository has been cloned to `third_party/RULER/`. Additional dependencies may be required:

```bash
pip install nltk wonderwords
```

## Usage

### Basic Evaluation

Run a quick RULER test with the provided configuration:

```bash
llamafactory-cli eval --config eval_configs/ruler_quick_test.yaml
```

### Available Configurations

Several pre-configured RULER evaluation setups are available:

1. **ruler_quick_test.yaml** - Minimal setup for testing (2K context, 2 samples)
2. **ruler_basic.yaml** - Basic needle-in-haystack at 4K context
3. **ruler_comprehensive.yaml** - All RULER tasks at multiple context lengths (4K-32K)
4. **ruler_rope_scaling.yaml** - Test with linear RoPE scaling (8K-64K)
5. **ruler_yarn.yaml** - Test with YARN RoPE scaling (16K-128K)

### Custom Configuration

Create a custom YAML configuration with RULER-specific parameters:

```yaml
# Model configuration
model_name_or_path: your-model-path

# RULER evaluation settings
task: ruler
ruler_tasks:
  - niah_single_1      # Single needle retrieval
  - niah_multikey_1    # Multi-key retrieval
  - vt                 # Variable tracking
  - cwe                # Common words extraction
ruler_context_lengths:
  - 4096
  - 8192
  - 16384
ruler_num_samples: 20
ruler_subset: validation
ruler_max_new_tokens: 50
output_dir: ./ruler_results/custom

# Optional RoPE scaling
rope_scaling_type: linear
rope_scaling_factor: 4.0
```

### RULER Task Types

Available RULER tasks (use in `ruler_tasks` list):

**Needle-in-Haystack variants:**
- `niah_single_1` - Single needle, noise haystack
- `niah_single_2` - Single needle, essay haystack
- `niah_single_3` - Single needle, UUID values
- `niah_multikey_1` - Multiple keys (4 keys)
- `niah_multikey_2` - Multiple keys, needle haystack
- `niah_multikey_3` - UUID keys and values
- `niah_multivalue` - Single key, multiple values
- `niah_multiquery` - Multiple queries

**Other tasks:**
- `vt` - Variable tracking (chain reasoning)
- `cwe` - Common words extraction
- `fwe` - Frequency words extraction
- `qa_1` - Question answering (SQuAD)
- `qa_2` - Question answering (HotpotQA)

### Output

Results are saved to the specified `output_dir` with:
- `{task}_{length}_predictions.jsonl` - Model predictions
- `{task}_{length}_results.json` - Evaluation scores
- `ruler_summary.json` - Overall summary

### Example Command

Evaluate Llama-2-7B with RULER on multiple context lengths:

```bash
llamafactory-cli eval \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task ruler \
    --ruler_tasks niah_single_1 niah_multikey_1 vt \
    --ruler_context_lengths 4096 8192 16384 \
    --ruler_num_samples 50 \
    --output_dir ./ruler_results/llama2_test
```

## Testing

Run the integration test to verify everything is set up correctly:

```bash
python3 test_ruler_integration.py
```

## Notes

- RULER data is generated dynamically based on the specified context length and task parameters
- The evaluation uses greedy decoding by default (temperature=0.0)
- For very long contexts, consider using quantization (`quantization_bit: 4`) to reduce memory usage
- FlashAttention-2 (`flash_attn: fa2`) can significantly speed up evaluation for long sequences