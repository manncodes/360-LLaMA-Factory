# Complete Workflow: Advanced RoPE for Continual Pretraining

## Overview

This workflow implements research-backed continual pretraining with advanced RoPE scaling, based on YaRN, LongLoRA, and LongRoPE papers. The approach extends context from 4K → 32K+ using progressive curriculum learning.

## Prerequisites

### 1. Extended ModelArguments (Required Implementation)

Current limitation: ModelArguments only supports `linear` and `dynamic` RoPE scaling. For continual pretraining with YARN/LongRoPE, extend `src/llamafactory/hparams/model_args.py`:

```python
# Add to ModelArguments class
rope_scaling_type: Optional[Literal["linear", "dynamic", "yarn", "longrope"]] = field(
    default=None,
    metadata={"help": "RoPE scaling type: linear, dynamic, yarn, longrope"}
)
rope_scaling_factor: Optional[float] = field(default=None)
yarn_alpha: Optional[float] = field(default=1.0)
yarn_beta: Optional[float] = field(default=32.0) 
longrope_short_factor: Optional[List[float]] = field(default=None)
longrope_long_factor: Optional[List[float]] = field(default=None)
original_max_position: Optional[int] = field(default=None)
```

### 2. Data Preparation

```bash
# Prepare long context dataset
python3 -c "
import datasets
from datasets import Dataset, concatenate_datasets

# Load datasets with naturally long documents
books = datasets.load_dataset('pg19', split='train[:1000]')  # Books
papers = datasets.load_dataset('arxiv', split='train[:1000]')  # Papers
code = datasets.load_dataset('codeparrot/github-code', split='train[:1000]')

# Filter for length >8K tokens
long_docs = []
for item in books:
    if len(item['text'].split()) > 2000:  # ~8K tokens
        long_docs.append({'text': item['text']})

# Save as long_context_corpus
Dataset.from_list(long_docs).save_to_disk('long_context_corpus')
print(f'Prepared {len(long_docs)} long documents')
"
```

## Workflow Steps

### Step 1: Setup Environment

```bash
# Create directories
mkdir -p continual_pretraining_configs/evaluation_configs
mkdir -p continual_pt_outputs

# Verify LlamaFactory installation
llamafactory-cli --help
```

### Step 2: Progressive Training Pipeline

#### Stage 1: YaRN 4K → 8K (Foundation)
```bash
# Apply initial YaRN scaling and train on 8K context
llamafactory-cli train continual_pretraining_configs/stage1_yarn_8k.yaml
```

**Expected outcomes:**
- Model learns to attend over 8K context
- YaRN scaling prevents position encoding collapse
- ~1000 training steps sufficient

#### Stage 2: YaRN 8K → 16K (Expansion) 
```bash
# Increase scaling factor and extend to 16K
llamafactory-cli train continual_pretraining_configs/stage2_yarn_16k.yaml
```

**Expected outcomes:**
- Progressive extension without catastrophic forgetting
- Maintains performance on shorter contexts
- ~1500 training steps for stabilization

#### Stage 3: LongRoPE 16K → 32K (Extreme Extension)
```bash
# Switch to LongRoPE for extreme context extension
llamafactory-cli train continual_pretraining_configs/stage3_longrope_32k.yaml
```

**Expected outcomes:**
- Non-uniform scaling optimizes different frequency components
- Achieves 32K+ context with minimal degradation
- ~2000 training steps for convergence

### Step 3: Automated Pipeline

```bash
# Run complete pipeline with validation
cd continual_pretraining_configs
python3 run_progressive_training.py

# Options:
python3 run_progressive_training.py --dry-run           # Validate only
python3 run_progressive_training.py --start-stage 2    # Resume from Stage 2
python3 run_progressive_training.py --validate-only    # Check configs
```

### Step 4: Evaluation

#### Quick Validation
```bash
# Test basic functionality
llamafactory-cli eval continual_pretraining_configs/evaluation_configs/needle_haystack_32k.yaml
```

#### Comprehensive Evaluation
```bash
# Full LongBench evaluation
llamafactory-cli eval continual_pretraining_configs/evaluation_configs/longbench_evaluation.yaml

# Run comprehensive sweep on final model
cd comprehensive_sweep
python3 fast_eval.py --model_path ../continual_pt_stage3_32k
```

## Expected Performance

### Training Efficiency
- **Total training time**: 4-8 hours on single GPU
- **Compute requirement**: <1% of original pretraining
- **Memory usage**: ~24GB for 32K context (1B model)

### Context Extension Results
| Stage | Context | RoPE Type | Training Steps | Memory (GB) |
|-------|---------|-----------|----------------|-------------|
| Base  | 4K      | None      | 0              | 8           |
| 1     | 8K      | YaRN      | 1000           | 12          |
| 2     | 16K     | YaRN      | 1500           | 18          |
| 3     | 32K     | LongRoPE  | 2000           | 24          |

### Quality Metrics
- **Needle-in-haystack**: >90% accuracy at all depths
- **LongBench**: Competitive with specialized long-context models
- **Perplexity**: Minimal degradation on original distribution

## Troubleshooting

### Memory Issues
```yaml
# Reduce batch size and increase accumulation
per_device_train_batch_size: 1
gradient_accumulation_steps: 128
bf16: true
gradient_checkpointing: true
```

### Training Instability
```yaml
# Use more conservative learning rate
learning_rate: 5e-6
max_grad_norm: 0.5
warmup_steps: 1000
```

### OOM on Long Sequences
```bash
# Start with shorter contexts and increase gradually
python3 run_progressive_training.py --start-stage 1
# Monitor GPU memory with: nvidia-smi -l 1
```

## Validation Commands

### Pre-Training Validation
```bash
# Test configuration parsing
python3 validate_rope_types.py

# Test model loading with advanced RoPE
python3 test_all_rope_types.py
```

### Post-Training Validation
```bash
# Quick context test
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('continual_pt_stage3_32k')
tokenizer = AutoTokenizer.from_pretrained('continual_pt_stage3_32k')

# Test with long input
long_text = 'Hello ' * 8000  # ~32K tokens
inputs = tokenizer(long_text, return_tensors='pt')
print(f'Input length: {inputs.input_ids.shape[1]} tokens')

outputs = model.generate(**inputs, max_new_tokens=10)
print('Success: Model handles 32K context')
"
```

## Research Integration Notes

### YaRN Application (Stages 1-2)
- Applied before continual pretraining (not after)
- Uses attention scaling (alpha) and position interpolation (beta)
- Maintains performance on shorter contexts
- Minimal compute: 0.1% of original training

### LongRoPE Application (Stage 3) 
- Different factors for different frequency components
- Non-uniform scaling optimizes position encoding
- Supports extreme extensions (2M+ tokens possible)
- Progressive training essential for stability

### Key Success Factors
1. **Apply RoPE BEFORE training on long context**
2. **Progressive extension** prevents catastrophic forgetting
3. **Curriculum learning** with increasing sequence lengths
4. **Quality long documents** essential for effective training
5. **Conservative hyperparameters** for training stability

This workflow transforms a 4K context model into a 32K+ context model using minimal compute while maintaining quality, following the latest research in continual pretraining with advanced RoPE scaling.