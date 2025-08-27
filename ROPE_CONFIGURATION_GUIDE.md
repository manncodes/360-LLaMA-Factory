# RoPE Configuration Guide for LlamaFactory

## Available RoPE Parameters in ModelArguments

### Basic RoPE Scaling
```yaml
# Linear or dynamic RoPE scaling (supported in current branch)
rope_scaling: linear    # Options: "linear" or "dynamic" 
rope_theta: 500000.0    # RoPE theta frequency base
```

### Parameter Descriptions

#### `rope_scaling`
- **Type**: `linear` or `dynamic` 
- **Purpose**: Method for extending context length
- **Linear**: Simple interpolation scaling
- **Dynamic**: Adaptive frequency scaling during training
- **Default**: None (uses model's original context length)

#### `rope_theta` 
- **Type**: Float
- **Purpose**: Base frequency for RoPE embeddings
- **LLaMA 2**: 10,000.0 (default)
- **LLaMA 3.2**: 500,000.0 (for better long context)
- **Ultra-long context**: 1,000,000.0+ (experimental)
- **Impact**: Higher values → better long context performance

## Example Configurations

### Standard Long Context (4K-8K tokens)
```yaml
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
rope_scaling: linear
rope_theta: 500000.0
cutoff_len: 4096
```

### Extended Long Context (8K-16K tokens)
```yaml
model_name_or_path: unsloth/Llama-3.2-1B-Instruct  
rope_scaling: dynamic
rope_theta: 1000000.0
cutoff_len: 8192
```

### Ultra Long Context (16K+ tokens)
```yaml
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
rope_scaling: dynamic
rope_theta: 2000000.0
cutoff_len: 16384
```

## Advanced RoPE (Available in EvaluationArguments only)

The following parameters are available for evaluation but NOT training in current branch:

```yaml
# Advanced RoPE types (evaluation only)
rope_scaling_type: yarn        # yarn, longrope, llama3
rope_scaling_factor: 4.0       # Context extension multiplier
yarn_alpha: 1.0               # YARN attention scaling
yarn_beta: 32.0               # YARN position interpolation
longrope_short_factor: [1.0]   # LongRoPE short factors
longrope_long_factor: [2.0]    # LongRoPE long factors
```

## Model-Specific Defaults

### LLaMA 3.2 1B (unsloth)
- Built-in RoPE config:
```json
{
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0, 
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "max_position_embeddings": 131072
}
```

### TinyLlama 1.1B
- Default: `rope_theta: 10000.0`
- Max positions: 2048
- Requires manual scaling for long context

## Best Practices

### Memory Optimization
- Use `bf16: true` for long contexts
- Start with `cutoff_len: 4096` and increase gradually
- Monitor GPU memory usage

### Context Length Guidelines
- **rope_theta: 500k** → Up to 32K tokens
- **rope_theta: 1M** → Up to 64K tokens  
- **rope_theta: 2M** → Up to 128K+ tokens

### Scaling Method Selection
- **Linear**: More stable, good for most use cases
- **Dynamic**: Better adaptation, use for very long contexts

## Testing Your Configuration

Use the provided `quick_rope_test.py` to validate:
```bash
python3 quick_rope_test.py
```

## Real-world Example

For a production long-context application:
```yaml
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
rope_scaling: linear
rope_theta: 1000000.0  # 1M for good long context
cutoff_len: 8192       # 8K context window
bf16: true             # Memory efficiency
template: llama3       # Proper tokenization
```

This configuration provides excellent long-context performance while maintaining training stability.