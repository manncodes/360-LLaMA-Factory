# Advanced RoPE Scaling for Continual Pretraining - Research Summary

## Key Findings from Recent Research

### 1. YaRN (NeurIPS 2023)
**Paper**: "YaRN: Efficient Context Window Extension of Large Language Models"

**Key Innovation**: Apply RoPE scaling BEFORE continual pretraining
```python
# YaRN formula
theta_i = base_theta * (alpha * (i/d) + (1-alpha))^(-beta)
```

**Training Strategy**:
1. **Initialize** with pretrained model
2. **Apply YaRN scaling** to position embeddings
3. **Continual pretrain** on long documents (16K-128K tokens)
4. **Fine-tune** on downstream tasks

**Results**: 
- LLaMA models extended from 4K → 128K context
- Only 0.1% of original training compute needed
- Maintains performance on original tasks

### 2. LongLoRA (ICLR 2024)
**Paper**: "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models"

**Key Innovation**: Shifted Sparse Attention + LoRA
```python
# Apply RoPE modification before training
model.config.rope_scaling = {
    "type": "linear",
    "factor": context_extension_factor
}
# Then apply S²-Attn during training
```

**Training Strategy**:
1. **Modify RoPE scaling** in model config
2. **Apply S²-Attn** (Shifted Sparse Attention)
3. **Use LoRA** for parameter-efficient training
4. **Progressively increase** context during training

**Results**:
- Extended LLaMA-2 7B to 100K context
- 70% reduction in GPU memory usage
- Preserved original model capabilities

### 3. LongRoPE (2024)
**Paper**: "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"

**Key Innovation**: Different RoPE factors for different frequency components
```python
# LongRoPE approach
rope_config = {
    "type": "longrope",
    "short_factor": [1.0, 1.5, 2.0],  # For high-freq
    "long_factor": [1.0, 2.0, 4.0],   # For low-freq
    "original_max_position": 8192,
    "target_max_position": 2048000
}
```

**Training Strategy**:
1. **Identify critical dimensions** via search
2. **Apply non-uniform scaling** to RoPE
3. **Progressive training**:
   - Stage 1: 256K tokens
   - Stage 2: 1M tokens  
   - Stage 3: 2M tokens
4. **Minimal fine-tuning** (1000 steps per stage)

**Results**:
- Extended to 2M tokens context
- Only 1000 training steps needed
- 90%+ accuracy on long-context tasks

### 4. Scaling Laws for RoPE (2024)
**Paper**: "Scaling Laws for Rope-based Long Context Extension"

**Key Findings**:
```python
# Optimal theta scaling
theta_optimal = theta_base * (L_target / L_base)^0.5

# Memory-efficient training
gradient_checkpointing = True
mixed_precision = "bf16"
micro_batch_size = 1
```

**Best Practices**:
1. **Theta scaling**: Square root of context ratio
2. **Gradual warmup**: Start with shorter sequences
3. **Curriculum learning**: Progressive length increase
4. **Mixed precision**: Essential for long contexts

## Implementation Strategies for Continual Pretraining

### Strategy 1: Direct Config Modification (Most Common)

```python
from transformers import AutoModelForCausalLM, AutoConfig

# Load pretrained model
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Modify RoPE before loading weights
config.rope_scaling = {
    "type": "yarn",  # or "linear", "dynamic"
    "factor": 8.0,
    "original_max_position_embeddings": 4096
}
config.max_position_embeddings = 32768

# Load model with modified config
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.bfloat16
)

# Now continual pretrain on long sequences
```

### Strategy 2: Post-Loading Modification

```python
# Load model normally
model = AutoModelForCausalLM.from_pretrained(...)

# Modify RoPE embeddings directly
for layer in model.model.layers:
    # Modify rotary embedding
    layer.self_attn.rotary_emb = modified_rope_embedding
    
# Continue pretraining
```

### Strategy 3: Progressive Extension

```python
# Stage 1: 8K context
train_with_context(model, context_len=8192, steps=1000)

# Stage 2: 16K context  
model.config.rope_scaling["factor"] = 4.0
train_with_context(model, context_len=16384, steps=1000)

# Stage 3: 32K context
model.config.rope_scaling["factor"] = 8.0
train_with_context(model, context_len=32768, steps=1000)
```

## Critical Implementation Details

### 1. When to Apply RoPE Scaling

**BEFORE Continual Pretraining**: ✅
- Modify config before loading model
- Apply scaling to position embeddings
- Start training on long sequences immediately

**DURING Continual Pretraining**: ✅
- Progressive scaling (8K → 16K → 32K)
- Curriculum learning approach
- Adjust scaling factor between stages

**AFTER Pretraining**: ❌
- Less effective
- May cause distribution shift
- Requires more fine-tuning

### 2. Training Data Requirements

```python
# Minimum data for effective extension
data_requirements = {
    "4K_to_8K": "10M tokens",
    "4K_to_16K": "50M tokens",
    "4K_to_32K": "200M tokens",
    "4K_to_128K": "1B tokens"
}

# Data composition
data_mix = {
    "long_documents": 0.5,  # Books, papers
    "medium_length": 0.3,   # Articles
    "short_text": 0.2       # Original distribution
}
```

### 3. Hyperparameter Recommendations

```yaml
# For continual pretraining with RoPE
learning_rate: 2e-5  # Lower than original pretraining
warmup_steps: 500
batch_size: 1  # Due to memory constraints
gradient_accumulation: 32
gradient_checkpointing: true
mixed_precision: bf16
optimizer: adamw
weight_decay: 0.01
max_grad_norm: 1.0
```

### 4. Evaluation Metrics

```python
# Key metrics to track
metrics = {
    "perplexity_short": "PPL on <4K sequences",
    "perplexity_long": "PPL on >16K sequences", 
    "needle_retrieval": "Accuracy at different depths",
    "passkey_retrieval": "Random token retrieval",
    "long_range_qa": "Question answering on long docs"
}
```

## Practical Implementation in LlamaFactory

### Current Limitation
The current branch only supports basic `linear` and `dynamic` in ModelArguments. To use advanced RoPE for continual pretraining, we need to:

### Solution 1: Extend ModelArguments

```python
# In src/llamafactory/hparams/model_args.py
@dataclass
class ModelArguments:
    rope_scaling_type: Optional[str] = field(
        default=None,
        metadata={"help": "RoPE scaling type: linear, dynamic, yarn, longrope"}
    )
    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={"help": "RoPE scaling factor"}
    )
    yarn_alpha: Optional[float] = field(default=None)
    yarn_beta: Optional[float] = field(default=None)
    # ... other parameters
```

### Solution 2: Direct Config Manipulation

```python
# In src/llamafactory/model/loader.py
def load_model(...):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    
    # Apply advanced RoPE config
    if model_args.rope_scaling_type == "yarn":
        config.rope_scaling = {
            "type": "yarn",
            "factor": model_args.rope_scaling_factor,
            "alpha": model_args.yarn_alpha,
            "beta": model_args.yarn_beta
        }
    # ... handle other types
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ...
    )
```

### Solution 3: Use Pretrained Models with RoPE

Many models already have advanced RoPE built-in:
- `NousResearch/Yarn-Llama-2-*`: YaRN applied
- `Yukang/LongAlpaca-*`: Linear scaling applied
- `togethercomputer/LLaMA-2-7B-32K`: Extended context
- `unsloth/llama-3-*`: LLaMA3 RoPE built-in

## Recommended Workflow for Continual Pretraining

1. **Choose Base Model**: Select model with existing long context support if possible
2. **Apply RoPE Scaling**: Modify config before training
3. **Prepare Data**: Long documents (>8K tokens)
4. **Progressive Training**:
   ```bash
   # Stage 1: 8K
   llamafactory-cli train stage1_8k.yaml
   
   # Stage 2: 16K
   llamafactory-cli train stage2_16k.yaml
   
   # Stage 3: 32K+
   llamafactory-cli train stage3_32k.yaml
   ```
5. **Evaluate**: Test on long-context benchmarks

## Key Insights

1. **Advanced RoPE MUST be applied before/during continual pretraining**, not after
2. **Progressive extension** works better than direct extreme extension
3. **Minimal compute needed**: Often <1% of original pretraining
4. **Data quality matters**: Need genuinely long documents, not concatenated short texts
5. **Curriculum learning**: Start with shorter sequences, gradually increase

## References

1. YaRN: https://arxiv.org/abs/2309.00071
2. LongLoRA: https://arxiv.org/abs/2309.12307
3. LongRoPE: https://arxiv.org/abs/2402.13753
4. Extending Context Window: https://arxiv.org/abs/2401.02415
5. RoPE Scaling Laws: https://arxiv.org/abs/2404.07146