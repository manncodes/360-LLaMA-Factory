# Long Context Methods Benchmark for 360-LLaMA-Factory

## Overview

This document provides a comprehensive analysis of inference-time long context improvement methods available in 360-LLaMA-Factory and benchmarking infrastructure to evaluate them.

## Available Methods

### 1. RoPE Scaling Methods

#### Linear Scaling
- **Configuration**: `rope_scaling: linear`
- **Description**: Linearly interpolates position embeddings to extend context length
- **Use Case**: Simple and effective for moderate context extensions (2-4x)
- **File**: `src/llamafactory/model/model_utils/rope.py`

#### Dynamic NTK Scaling
- **Configuration**: `rope_scaling: dynamic`
- **Description**: Dynamically adjusts the base frequency of RoPE embeddings
- **Use Case**: Better for larger context extensions but may affect fine-tuning
- **Warning**: May not work well with fine-tuning as noted in the codebase

### 2. LongLoRA S²-Attn (Shift Short Attention)

- **Configuration**: `shift_attn: true`
- **Description**: Implements shifted window attention from LongLoRA paper
- **Group Size Ratio**: 0.25 (1/4)
- **File**: `src/llamafactory/model/model_utils/longlora.py`
- **Requirements**: Must be used with training or compatible with specific model types

### 3. FlashAttention Optimizations

- **Options**:
  - `flash_attn: fa2` - FlashAttention-2
  - `flash_attn: sdpa` - PyTorch Scaled Dot Product Attention
  - `flash_attn: auto` - Automatic selection
  - `flash_attn: disabled` - No optimization
- **File**: `src/llamafactory/model/model_utils/attention.py`

### 4. Sequence Parallel (SP Branch Feature)

- **Configuration**: `sequence_parallel_size: N`
- **Modes**: 
  - `zigzag-ring` (default)
  - `llama3`
  - `ulysses`
- **Description**: Distributes sequence processing across multiple GPUs
- **File**: `src/llamafactory/hparams/model_args.py`

## Methods NOT Implemented

The following popular long context methods were searched but NOT found in the codebase:
- **YaRN** (Yet another RoPE extension)
- **LongRope**
- **NoPE** (No Position Encoding)
- **PI** (Position Interpolation) - partially covered by linear scaling
- **Advanced NTK variants**

## Benchmark Setup

### Quick Test

```bash
# Test basic functionality
python test_long_context.py --test all

# Run quick benchmark
python benchmark_long_context.py --quick
```

### Full Benchmark

```bash
# Run comprehensive benchmark
./run_benchmarks.sh

# Or with Python script for detailed analysis
python benchmark_long_context.py \
    --contexts 2048 4096 8192 16384 \
    --rope-methods none linear dynamic \
    --attention-types auto sdpa fa2 \
    --num-samples 10
```

### Configuration Examples

#### Baseline (No Scaling)
```yaml
rope_scaling: none
model_max_length: 8192
flash_attn: auto
```

#### Linear RoPE Scaling
```yaml
rope_scaling: linear
model_max_length: 32768
flash_attn: fa2
```

#### Dynamic NTK Scaling
```yaml
rope_scaling: dynamic
model_max_length: 32768
flash_attn: fa2
```

#### LongLoRA with S²-Attn
```yaml
rope_scaling: linear
shift_attn: true
model_max_length: 32768
flash_attn: fa2
```

## Benchmark Results Structure

Results are saved in JSON format with the following structure:

```json
{
  "config": {
    "model": "model_name",
    "context_lengths": [2048, 4096, 8192],
    "rope_methods": ["none", "linear", "dynamic"]
  },
  "results": [
    {
      "context_length": 4096,
      "rope_scaling": "linear",
      "attention_type": "fa2",
      "shift_attn": false,
      "avg_inference_time": 2.5,
      "accuracy_rate": 0.95,
      "avg_memory_gb": 8.2
    }
  ]
}
```

## Performance Expectations

Based on the implementation analysis:

1. **Best Speed**: FlashAttention-2 (`fa2`) with linear RoPE scaling
2. **Best Accuracy**: Dynamic NTK scaling for moderate extensions
3. **Best Memory**: SDPA with appropriate context chunking
4. **Most Stable**: Linear scaling without shift attention

## Usage Recommendations

### For Inference Only (No Training)
1. Use `rope_scaling: linear` for 2-4x context extension
2. Enable `flash_attn: fa2` if available
3. Keep `shift_attn: false` for inference

### For Fine-tuning
1. Use `rope_scaling: linear` (avoid dynamic)
2. Consider `shift_attn: true` for LongLoRA-style training
3. Enable gradient checkpointing for memory efficiency

### For Multi-GPU Setup
1. Enable sequence parallel with `sequence_parallel_size > 1`
2. Choose appropriate mode based on model architecture
3. Use `zigzag-ring` as default

## Running on SP Branch

Since we're on the SP branch with sequence parallel support:

```bash
# For multi-GPU setup
python -m llamafactory.cli train \
    --config benchmark_configs/rope_linear.yaml \
    --sequence_parallel_size 2 \
    --sequence_parallel_mode zigzag-ring
```

## Monitoring and Analysis

The benchmark suite provides:
- Inference time measurements
- Memory usage tracking
- Accuracy evaluation (needle-in-haystack style)
- Comparative analysis across methods

## Next Steps

1. Run baseline benchmarks to establish performance metrics
2. Test each method with increasing context lengths
3. Compare accuracy vs speed trade-offs
4. Identify optimal configurations for specific use cases
5. Consider implementing missing methods (YaRN, LongRope) if needed

## Notes

- The SP branch includes additional optimizations for sequence parallelism
- FlashAttention-2 requires separate installation via `pip install flash-attn`
- Dynamic NTK scaling should be used carefully with fine-tuning
- S²-Attn is primarily designed for training, not inference