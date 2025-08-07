# Sequence Parallelism Configuration Examples

This directory contains example YAML configuration files demonstrating how to enable and configure sequence parallelism in 360-LLaMA-Factory.

## Overview

Sequence Parallelism (SP) allows you to train models with longer context lengths by distributing the sequence dimension across multiple GPUs. This is particularly useful for long-context training where memory constraints would otherwise limit sequence lengths.

## Key Benefits

- **Extended Context Length**: Train with much longer sequences than single-GPU setups
- **Memory Efficiency**: Distribute sequence processing across multiple GPUs
- **Minimal Code Changes**: Enable SP with just a few configuration parameters

## Configuration Files

### Basic Examples

1. **`sequence_parallel_demo.yaml`** - Basic SFT with sequence parallelism
   - `sequence_parallel_size: 4`
   - `sequence_parallel_mode: zigzag-ring`
   - `cutoff_len: 128000`
   - Full parameter fine-tuning setup

2. **`sequence_parallel_dpo_demo.yaml`** - DPO training with sequence parallelism
   - Optimized for Direct Preference Optimization
   - LoRA configuration for efficiency
   - `cutoff_len: 84000` (typical for DPO)

3. **`sequence_parallel_ulysses.yaml`** - Alternative Ulysses mode
   - Uses `sequence_parallel_mode: ulysses`
   - LoRA fine-tuning configuration
   - Good for experimenting with different SP algorithms

### Advanced Example

4. **`sequence_parallel_advanced.yaml`** - Advanced configuration
   - Large model (72B parameters)
   - `sequence_parallel_size: 8`
   - `cutoff_len: 210000` (very long context)
   - Multiple memory optimization techniques

## Key Configuration Parameters

### Sequence Parallelism Settings

```yaml
# Number of GPUs to process one sequence (must divide total GPU count evenly)
sequence_parallel_size: 4

# Mode of sequence parallel implementation
sequence_parallel_mode: zigzag-ring  # Options: zigzag-ring, ulysses, llama3

# Sequence length for training (set to your data's maximum length)
cutoff_len: 128000
```

### Recommended Complementary Settings

```yaml
# Memory optimizations
gradient_checkpointing: true
flash_attn: fa2
packing: true
bf16: true

# DeepSpeed for additional memory savings
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```

## Sequence Parallel Modes

### zigzag-ring (Default, Recommended)
- Most stable and well-tested
- Based on ring-flash-attention
- Good performance across different model sizes
- Requires `ring_flash_attn` package

### ulysses
- Alternative implementation
- May have different performance characteristics
- Good for experimentation

### llama3
- Specialized mode for LLaMA 3 models
- May offer optimizations specific to LLaMA architecture

## Performance Guidelines

### GPU Requirements
- `sequence_parallel_size` must evenly divide your total GPU count
- Common values: 2, 4, 8, 16
- Start with `sequence_parallel_size: 2` for initial testing

### Memory Scaling
According to the project documentation, with 8 x A800 GPUs and sequence parallelism:
- **Full SFT**: Up to 210k context length (7B model), 128k (72B model)
- **DPO**: Up to 84k context length (7B model), 46k (72B model)

### Typical Use Cases
- 32k context: `sequence_parallel_size: 2`
- 64k-128k context: `sequence_parallel_size: 4`
- 128k+ context: `sequence_parallel_size: 8` or higher

## Usage Examples

### Command Line
```bash
# Single node, 8 GPUs, SP size of 4
llamafactory-cli train examples/sequence_parallel_demo.yaml

# Multi-node with DeepSpeed
deepspeed --hostfile=8nodes.host src/train.py \
    --config examples/sequence_parallel_advanced.yaml \
    --sequence_parallel_size 8 \
    --cutoff_len 128000
```

### Alternative Command Line Syntax
```bash
deepspeed src/train.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset identity,alpaca_en_demo \
    --template llama3 \
    --sequence_parallel_size 4 \
    --sequence_parallel_mode zigzag-ring \
    --cutoff_len 128000 \
    --flash_attn fa2 \
    --gradient_checkpointing true \
    --bf16 true \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json
```

## Troubleshooting

### Common Issues
1. **Import Error**: Ensure `ring_flash_attn` is installed for zigzag-ring mode
2. **GPU Count Mismatch**: Verify `sequence_parallel_size` divides total GPU count evenly
3. **Memory Issues**: Try enabling `gradient_checkpointing` and reducing `per_device_train_batch_size`

### Performance Tips
1. Start with `sequence_parallel_size: 2` for initial testing
2. Use `flash_attn: fa2` for optimal performance
3. Enable `packing: true` for better throughput
4. Consider `enable_liger_kernel: true` for additional memory savings

## Dependencies

- `ring_flash_attn` (for zigzag-ring mode)
- `flash_attn` (recommended)
- DeepSpeed (for multi-GPU memory optimization)

For more details, see the main README and the 360-LLaMA-Factory technical report.