# Advanced RoPE for Continual Pretraining - Implementation Plan

## Current State Analysis

### Training Pipeline (ModelArguments)
```python
rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None)
rope_theta: Optional[float] = field(default=None)
```

### Evaluation Pipeline (EvaluationArguments) 
```python
rope_scaling_type: Optional[str] = field(default=None)  # Supports: linear, dynamic, yarn, longrope, llama3
rope_scaling_factor: Optional[float] = field(default=None)
yarn_alpha: Optional[float] = field(default=None)
yarn_beta: Optional[float] = field(default=None)
longrope_short_factor: Optional[List[float]] = field(default=None)
longrope_long_factor: Optional[List[float]] = field(default=None)
```

## Implementation Strategy for Continual Pretraining

### Phase 1: Extend ModelArguments for Advanced RoPE

**File: src/llamafactory/hparams/model_args.py**

Add these parameters to ModelArguments:
```python
# Replace existing rope_scaling with more comprehensive version
rope_scaling_type: Optional[Literal["linear", "dynamic", "yarn", "longrope"]] = field(
    default=None,
    metadata={"help": "RoPE scaling type for continual pretraining: linear, dynamic, yarn, longrope"}
)
rope_scaling_factor: Optional[float] = field(
    default=None,
    metadata={"help": "RoPE scaling factor for context extension"}
)
# YARN parameters
yarn_alpha: Optional[float] = field(
    default=1.0,
    metadata={"help": "YARN alpha parameter for attention scaling"}
)
yarn_beta: Optional[float] = field(
    default=32.0,
    metadata={"help": "YARN beta parameter for position interpolation"}
)
# LongRoPE parameters
longrope_short_factor: Optional[List[float]] = field(
    default=None,
    metadata={"help": "LongRoPE short range factors for high-frequency components"}
)
longrope_long_factor: Optional[List[float]] = field(
    default=None,
    metadata={"help": "LongRoPE long range factors for low-frequency components"}
)
# Original max position for scaling
original_max_position: Optional[int] = field(
    default=None,
    metadata={"help": "Original max position embeddings before scaling"}
)
```

### Phase 2: Modify RoPE Configuration Logic

**File: src/llamafactory/model/model_utils/rope.py**

Extend the configure_rope function:
```python
def configure_rope(config, model_args, is_trainable: bool) -> None:
    if model_args.rope_scaling_type is not None:
        if model_args.rope_scaling_type in ["yarn", "longrope"]:
            # Build comprehensive RoPE scaling config
            rope_config = {
                "type": model_args.rope_scaling_type,
                "factor": model_args.rope_scaling_factor or 4.0
            }
            
            if model_args.rope_scaling_type == "yarn":
                rope_config.update({
                    "alpha": model_args.yarn_alpha or 1.0,
                    "beta": model_args.yarn_beta or 32.0
                })
            
            elif model_args.rope_scaling_type == "longrope":
                rope_config.update({
                    "short_factor": model_args.longrope_short_factor or [1.0, 1.5, 2.0],
                    "long_factor": model_args.longrope_long_factor or [1.0, 2.0, 4.0]
                })
            
            if model_args.original_max_position is not None:
                rope_config["original_max_position_embeddings"] = model_args.original_max_position
            
            setattr(config, "rope_scaling", rope_config)
            logger.info_rank0(f"Applied {model_args.rope_scaling_type} RoPE scaling: {rope_config}")
```

### Phase 3: Continual Pretraining Configuration Templates

**Template 1: Progressive YaRN Extension**
```yaml
# Stage 1: 4K -> 8K with YaRN
model_name_or_path: meta-llama/Llama-2-7b-hf
rope_scaling_type: yarn
rope_scaling_factor: 2.0
yarn_alpha: 1.0
yarn_beta: 32.0
rope_theta: 500000.0
original_max_position: 4096
cutoff_len: 8192

# Continual pretraining settings
stage: pt
do_train: true
dataset: long_context_corpus
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 2e-5
num_train_epochs: 1
warmup_steps: 500
bf16: true
gradient_checkpointing: true
dataloader_drop_last: true
```

**Template 2: LongRoPE for Extreme Extension**
```yaml
# Stage 1: 8K -> 32K with LongRoPE
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
rope_scaling_type: longrope
rope_scaling_factor: 4.0
longrope_short_factor: [1.0, 1.5, 2.0]
longrope_long_factor: [1.0, 2.0, 4.0]
rope_theta: 1000000.0
original_max_position: 8192
cutoff_len: 32768

# Progressive training curriculum
max_samples: 10000  # Start small for Stage 1
```

### Phase 4: Progressive Training Workflow

**Script: progressive_rope_training.py**
```python
#!/usr/bin/env python3
"""Progressive RoPE scaling for continual pretraining."""

import subprocess
import yaml
from pathlib import Path

def create_stage_config(base_config, stage_info):
    """Create configuration for a training stage."""
    config = base_config.copy()
    config.update(stage_info)
    return config

def run_continual_pretraining(model_path, target_context=32768):
    """Run progressive continual pretraining with RoPE scaling."""
    
    base_config = {
        "model_name_or_path": model_path,
        "stage": "pt",
        "do_train": True,
        "dataset": "long_context_corpus",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 2e-5,
        "warmup_steps": 500,
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 1000,
        "logging_steps": 100
    }
    
    # Define progressive stages
    stages = [
        {
            "name": "stage1_8k",
            "rope_scaling_type": "yarn",
            "rope_scaling_factor": 2.0,
            "yarn_alpha": 1.0,
            "yarn_beta": 32.0,
            "rope_theta": 500000.0,
            "cutoff_len": 8192,
            "num_train_epochs": 0.5,
            "max_samples": 5000
        },
        {
            "name": "stage2_16k", 
            "rope_scaling_type": "yarn",
            "rope_scaling_factor": 4.0,
            "yarn_alpha": 1.0,
            "yarn_beta": 32.0,
            "rope_theta": 1000000.0,
            "cutoff_len": 16384,
            "num_train_epochs": 0.5,
            "max_samples": 10000
        },
        {
            "name": "stage3_32k",
            "rope_scaling_type": "longrope",
            "rope_scaling_factor": 8.0,
            "longrope_short_factor": [1.0, 1.5, 2.0],
            "longrope_long_factor": [1.0, 2.0, 4.0],
            "rope_theta": 2000000.0,
            "cutoff_len": 32768,
            "num_train_epochs": 1.0,
            "max_samples": 20000
        }
    ]
    
    for stage in stages:
        print(f"Starting {stage['name']}")
        
        # Create stage config
        stage_config = create_stage_config(base_config, stage)
        stage_config["output_dir"] = f"continual_pt_{stage['name']}"
        
        # Save config
        config_file = f"{stage['name']}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(stage_config, f)
        
        # Run training
        result = subprocess.run([
            "llamafactory-cli", "train", config_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Stage {stage['name']} failed: {result.stderr}")
            break
        
        print(f"Stage {stage['name']} completed successfully")
        
        # Update model path for next stage
        base_config["model_name_or_path"] = stage_config["output_dir"]

if __name__ == "__main__":
    run_continual_pretraining("unsloth/Llama-3.2-1B-Instruct", 32768)
```

## Critical Implementation Notes

### 1. Backward Compatibility
- Keep existing `rope_scaling` parameter as alias for `rope_scaling_type`
- Maintain support for simple string values ("linear", "dynamic")
- Add deprecation warning for old parameter

### 2. Validation Logic
```python
def __post_init__(self):
    # Validate RoPE parameters
    if self.rope_scaling_type == "yarn":
        if self.rope_scaling_factor is None:
            raise ValueError("rope_scaling_factor required for YARN")
        if self.yarn_alpha is None:
            self.yarn_alpha = 1.0
        if self.yarn_beta is None:
            self.yarn_beta = 32.0
    
    elif self.rope_scaling_type == "longrope":
        if self.rope_scaling_factor is None:
            raise ValueError("rope_scaling_factor required for LongRoPE")
        if self.longrope_short_factor is None:
            self.longrope_short_factor = [1.0, 1.5, 2.0]
        if self.longrope_long_factor is None:
            self.longrope_long_factor = [1.0, 2.0, 4.0]
```

### 3. Memory and Performance Considerations

For continual pretraining with long contexts:
```yaml
# Essential memory optimizations
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
gradient_checkpointing: true
bf16: true
dataloader_drop_last: true

# Long context specific
rope_scaling_type: yarn  # or longrope
cutoff_len: 16384
max_samples: 10000  # Limit data for memory

# Learning rate scheduling
learning_rate: 2e-5  # Lower than original pretraining
warmup_steps: 500
lr_scheduler_type: cosine
```

## Testing Strategy

### 1. Configuration Validation
```bash
# Test all new configurations
python3 validate_advanced_rope.py
```

### 2. Progressive Training Test
```bash
# Run 1-step test for each stage
python3 test_progressive_rope.py --max_steps 1
```

### 3. Memory Profiling
```bash
# Profile memory usage at different context lengths
python3 profile_rope_memory.py --contexts 4096,8192,16384
```

## Production Workflow

### Step 1: Prepare Long Context Data
```bash
# Prepare dataset with documents >8K tokens
python3 prepare_long_context_dataset.py \
  --input_dir raw_documents \
  --output_dir long_context_corpus \
  --min_tokens 8192
```

### Step 2: Run Progressive Training
```bash
# Stage 1: Apply RoPE and train on 8K
llamafactory-cli train stage1_yarn_8k.yaml

# Stage 2: Extend to 16K
llamafactory-cli train stage2_yarn_16k.yaml

# Stage 3: Final extension to 32K with LongRoPE
llamafactory-cli train stage3_longrope_32k.yaml
```

### Step 3: Evaluation
```bash
# Evaluate on long context benchmarks
llamafactory-cli eval longbench_eval.yaml
llamafactory-cli eval needle_haystack_eval.yaml
```

## Key Benefits of This Implementation

1. **Research-Backed**: Based on YaRN, LongLoRA, LongRoPE papers
2. **Progressive**: Curriculum learning prevents catastrophic forgetting  
3. **Memory-Efficient**: Optimized for practical training constraints
4. **Production-Ready**: Comprehensive error handling and validation
5. **Flexible**: Supports all major RoPE scaling methods

## Minimal Compute Requirements

Based on research findings:
- **YaRN**: 0.1% of original pretraining compute
- **LongRoPE**: 1000 steps per extension stage
- **Progressive**: 500-1000 steps per doubling of context

This makes continual pretraining feasible on modest hardware while achieving significant context extensions.