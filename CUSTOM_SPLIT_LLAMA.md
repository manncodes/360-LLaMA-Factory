# CustomSplitLLamaModel Integration

This document describes the CustomSplitLLamaModel architecture integration, which implements a LLaMA Pro-like model that combines layers from two different LLaMA models (8B and 70B) with an adapter layer.

## Architecture Overview

The CustomSplitLLamaModel creates a hybrid architecture by:

1. **First Stage**: Uses the first N layers from an 8B LLaMA model
2. **Adapter Layer**: Transforms hidden size from 8B model (h8) to 70B model (h70 = 2*h8)
3. **Second Stage**: Uses the last N layers from a 70B LLaMA model

```
Input → Embed → 8B Layers (N) → Adapter → 70B Layers (N) → Norm → Output
```

## Configuration Parameters

### Required Parameters

- `use_custom_split_llama: true` - Enable CustomSplitLLamaModel
- `path8b: <path>` - Path to the 8B model (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
- `path70b: <path>` - Path to the 70B model (e.g., "meta-llama/Meta-Llama-3-70B-Instruct")

### Optional Parameters

- `num_layers_8: <int>` - Number of layers to use from 8B model (default: 16)
- `num_layers_70: <int>` - Number of layers to use from 70B model (default: 16)
- `use_mlp_adapter: <bool>` - Use MLP adapter instead of linear (default: false)

## Usage Examples

### Training Configuration

```yaml
# examples/train_lora/custom_split_llama_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct  # Placeholder, will be ignored
use_custom_split_llama: true
path8b: meta-llama/Meta-Llama-3-8B-Instruct
path70b: meta-llama/Meta-Llama-3-70B-Instruct
num_layers_8: 16
num_layers_70: 16
use_mlp_adapter: false

stage: sft
finetuning_type: lora
lora_target: all
# ... other training parameters
```

### Command Line Usage

```bash
llamafactory-cli train \
    --stage sft \
    --use_custom_split_llama true \
    --path8b "meta-llama/Meta-Llama-3-8B-Instruct" \
    --path70b "meta-llama/Meta-Llama-3-70B-Instruct" \
    --num_layers_8 16 \
    --num_layers_70 16 \
    --finetuning_type lora \
    # ... other parameters
```

## Adapter Types

### Linear Adapter (Default)
- Single linear transformation: `h8 → h70`
- Parameter count: `h8 * h70`
- Faster and more parameter-efficient

### MLP Adapter
- Two-layer MLP with ReLU: `h8 → h70 → h70`
- Parameter count: `h8 * h70 + h70 * h70`
- More expressive but higher parameter count

Set `use_mlp_adapter: true` to use the MLP adapter.

## Memory and Performance Considerations

### Memory Usage
- Loads both 8B and 70B models during initialization
- Models are deleted after extracting required components
- Final model size depends on number of layers used from each model

### Training Considerations
- Supports LoRA fine-tuning on the combined architecture
- Gradient checkpointing is supported
- Compatible with DeepSpeed and other optimization strategies

### Inference
- Standard inference workflows work without modification
- Supports vLLM backend (experimental)

## Implementation Details

### File Structure
```
src/llamafactory/
├── model/
│   └── model_utils/
│       └── custom_split_llama.py          # Core implementation
├── hparams/
│   └── model_args.py                      # Configuration parameters
└── loader.py                             # Model loading integration

examples/
└── train_lora/
    └── custom_split_llama_sft.yaml        # Example configuration

test_custom_split_llama.py                 # Integration tests
```

### Key Components

1. **CustomSplitLLamaModel**: Main model class inheriting from LlamaModel
2. **load_custom_split_llama_model**: Factory function for model creation
3. **ModelArguments**: Extended to include CustomSplitLLamaModel parameters
4. **Integration**: Seamlessly integrated into LlamaFactory's loading pipeline

## Testing

Run the integration test:

```bash
python test_custom_split_llama.py
```

This verifies:
- Parameter validation
- Model loading logic
- Configuration handling

## Limitations and Notes

1. **Model Compatibility**: Requires LLaMA-family models with compatible architectures
2. **Hidden Size Constraint**: The 70B model's hidden size must be exactly 2x the 8B model's hidden size
3. **Memory Requirements**: Initial loading requires enough memory for both source models
4. **Fine-tuning**: Only certain layers (adapter and selected transformer layers) are typically trained

## Future Enhancements

Potential improvements:
- Support for more flexible hidden size ratios
- Dynamic layer selection strategies
- Optimized memory loading (streaming)
- Additional adapter architectures
- Integration with more model families

## Troubleshooting

### Common Issues

1. **Memory Errors**: Ensure sufficient GPU/CPU memory for loading both models
2. **Hidden Size Mismatch**: Verify that h70 = 2 * h8 for your chosen models
3. **Path Errors**: Ensure both model paths are valid and accessible

### Debug Mode

Enable detailed logging:
```yaml
logging_level: DEBUG
```

This will show model loading progress and architecture details.