# ROPE Scaling Configuration Guide

This guide provides comprehensive documentation for all RoPE (Rotary Position Embedding) scaling configuration options available in 360-LLaMA-Factory.

## Overview

RoPE scaling extends the context length capabilities of transformer models beyond their original training limits. This system supports multiple scaling methods and both flat and nested YAML configuration formats.

## Configuration Formats

The system supports two YAML configuration formats that produce identical results:

### Flat Format (Traditional)
```yaml
model_name_or_path: meta-llama/Llama-2-7b-hf
rope_scaling_type: linear
rope_scaling_factor: 2.0
cutoff_len: 4096
```

### Nested Format (HuggingFace Compatible)
```yaml
model_name_or_path: meta-llama/Llama-2-7b-hf
rope_scaling:
  type: linear
  factor: 2.0
cutoff_len: 4096
```

## RoPE Scaling Methods

### 1. Linear Scaling

Simple position interpolation scaling for moderate context extension.

**Flat Format:**
```yaml
rope_scaling_type: linear
rope_scaling_factor: 2.0  # Doubles context length
```

**Nested Format:**
```yaml
rope_scaling:
  type: linear
  factor: 2.0
```

**Use Case:** Basic context extension (2x-4x), general fine-tuning
**Recommendation:** Start here for most applications

### 2. Dynamic Scaling (NTK)

Neural Tangent Kernel aware scaling that adapts frequency base dynamically.

**Flat Format:**
```yaml
rope_scaling_type: dynamic
rope_scaling_factor: 4.0
```

**Nested Format:**
```yaml
rope_scaling:
  type: dynamic
  factor: 4.0
```

**Use Case:** Moderate context extension with better frequency adaptation
**Warning:** May not work well with fine-tuning

### 3. YARN Scaling

Yet Another RoPE extensioN - advanced scaling with attention and position interpolation.

**Flat Format:**
```yaml
rope_scaling_type: yarn
rope_scaling_factor: 4.0
yarn_alpha: 1.0      # Attention scaling parameter
yarn_beta: 32.0      # Position interpolation parameter
```

**Nested Format:**
```yaml
rope_scaling:
  type: yarn
  factor: 4.0
  alpha: 1.0
  beta: 32.0
```

**Use Case:** High-quality context extension for continual pretraining
**Best For:** 4x-8x extension with minimal performance degradation

#### YARN Parameters
- `factor`: Base scaling factor (required)
- `alpha`: Attention scaling parameter (default: 1.0)
- `beta`: Position interpolation parameter (default: 32.0)
- `original_max_position_embeddings`: Original context size (optional)

### 4. LongRoPE Scaling

Advanced multi-frequency scaling for extreme context extension.

**Flat Format:**
```yaml
rope_scaling_type: longrope
rope_scaling_factor: 8.0
longrope_short_factor: [1.0, 1.5, 2.0, 2.5]
longrope_long_factor: [1.0, 2.0, 4.0, 8.0]
```

**Nested Format:**
```yaml
rope_scaling:
  type: longrope
  factor: 8.0
  short_factor: [1.0, 1.5, 2.0, 2.5]
  long_factor: [1.0, 2.0, 4.0, 8.0]
```

**Use Case:** Extreme context extension (8x-32x) with frequency component optimization
**Best For:** Applications requiring very long contexts

#### LongRoPE Parameters
- `factor`: Overall scaling factor (required)
- `short_factor`: High-frequency component scaling factors (default: [1.0, 1.5, 2.0])
- `long_factor`: Low-frequency component scaling factors (default: [1.0, 2.0, 4.0])

## Additional Parameters

### RoPE Theta
Frequency base parameter that can be used with any scaling method:

**Flat Format:**
```yaml
rope_scaling_type: linear
rope_scaling_factor: 2.0
rope_theta: 1000000.0  # Higher values help with long contexts
```

**Nested Format:**
```yaml
rope_scaling:
  type: linear
  factor: 2.0
  rope_theta: 1000000.0
```

**Mixed Format (Supported):**
```yaml
rope_scaling:
  type: yarn
  factor: 4.0
  alpha: 2.0
  beta: 32.0
rope_theta: 1000000.0  # Can be specified outside nested config
```

## Complete Examples

### Basic Linear Extension
```yaml
# Double the context length with linear scaling
model_name_or_path: meta-llama/Llama-2-7b-hf
template: llama2
cutoff_len: 8192

rope_scaling_type: linear
rope_scaling_factor: 2.0

output_dir: ./output
learning_rate: 2e-5
num_train_epochs: 3
```

### Advanced YARN Configuration
```yaml
# High-quality 4x context extension for continual pretraining
model_name_or_path: meta-llama/Llama-2-13b-hf
template: llama2
cutoff_len: 16384

rope_scaling:
  type: yarn
  factor: 4.0
  alpha: 1.0
  beta: 32.0
  original_max_position_embeddings: 4096

rope_theta: 1000000.0

output_dir: ./yarn_output
learning_rate: 1e-5
num_train_epochs: 1
```

### Extreme LongRoPE Extension
```yaml
# 8x context extension with frequency optimization
model_name_or_path: microsoft/Phi-3-mini-4k-instruct
template: phi
cutoff_len: 32768

rope_scaling:
  type: longrope
  factor: 8.0
  short_factor: [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.4, 1.4]
  long_factor: [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]

flash_attn: fa2
sequence_parallel_size: 4

output_dir: ./longrope_output
learning_rate: 5e-6
num_train_epochs: 1
```

## Legacy Format Support

The system maintains backward compatibility with legacy string-based configurations:

```yaml
# Legacy format (automatically converted)
rope_scaling: linear  # Equivalent to rope_scaling_type: linear
# rope_scaling_factor defaults to 2.0 if not specified
```

## Best Practices

### Choosing Scaling Methods
1. **Linear**: Start here for 2x-4x extensions
2. **YARN**: Use for high-quality 4x-8x extensions with continual pretraining
3. **Dynamic**: Consider for inference-only applications
4. **LongRoPE**: Use for extreme extensions (8x+) with careful tuning

### Parameter Guidelines
- **Scaling Factor**: Start conservative (2.0-4.0) and increase gradually
- **YARN Alpha**: Keep at 1.0 unless specific research indicates otherwise
- **YARN Beta**: 32.0 works well for most cases
- **RoPE Theta**: Increase (100K-1M) for very long contexts

### Performance Considerations
- Higher scaling factors require more memory and computation
- YARN and LongRoPE add computational overhead but provide better quality
- Test with your specific model and use case
- Monitor validation metrics when extending context

## Validation and Warnings

The system provides automatic validation:

### Automatic Warnings
- Non-positive scaling factors (≤ 0)
- Very large scaling factors (> 100)
- Missing required parameters
- Model compatibility issues

### Error Handling
- Invalid RoPE types result in clear error messages
- Missing required parameters are caught early
- Graceful fallback for malformed configurations

## Model Compatibility

RoPE scaling works with models that support rotary position embeddings:
- ✅ Llama, Llama 2, Code Llama
- ✅ Mistral, Mixtral
- ✅ Qwen, Qwen 2
- ✅ Phi-3
- ✅ Most recent transformer architectures
- ❌ Models without RoPE (GPT-2, BERT, etc.)

## Integration Examples

### Training Configuration
```yaml
# Complete training setup with RoPE scaling
model_name_or_path: meta-llama/Llama-2-7b-chat-hf
template: llama2

rope_scaling:
  type: yarn
  factor: 4.0
  alpha: 1.0
  beta: 32.0

dataset: long_context_dataset
cutoff_len: 16384
learning_rate: 1e-5
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

flash_attn: fa2
output_dir: ./long_context_model
```

### Inference Configuration
```yaml
# Inference with extended context
model_name_or_path: ./long_context_model
template: llama2

rope_scaling:
  type: yarn
  factor: 4.0
  alpha: 1.0
  beta: 32.0

infer_dtype: bfloat16
use_cache: true
temperature: 0.1
```

## Troubleshooting

### Common Issues

**YAML Parsing Errors:**
- Ensure proper indentation for nested format
- Use consistent spacing (2 or 4 spaces)
- Verify all required fields are present

**Model Loading Failures:**
- Check model compatibility with RoPE
- Verify scaling factors are reasonable
- Ensure adequate GPU memory

**Performance Issues:**
- Start with lower scaling factors
- Use Flash Attention when possible
- Consider gradient checkpointing for memory

### Debugging Commands

Test your configuration:
```bash
# Validate configuration
llamafactory-cli train --config your_config.yaml --dry_run

# Test inference
llamafactory-cli chat --config your_config.yaml
```

## Summary

The 360-LLaMA-Factory RoPE scaling system provides flexible, powerful context extension capabilities with:

- **Multiple scaling methods** for different use cases
- **Dual format support** (flat and nested YAML)
- **Complete parameter control** for advanced users
- **Robust validation** and error handling
- **Production-ready stability** with comprehensive testing

Choose the appropriate scaling method and parameters based on your specific requirements for context length, quality, and computational resources.