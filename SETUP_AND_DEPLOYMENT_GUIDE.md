# Setup and Deployment Guide for RoPE Hyperparameter Sweeps

## Quick Setup on New Node

### 1. Clone and Setup Repository

```bash
# Clone your fork
git clone https://github.com/manncodes/360-LLaMA-Factory.git
cd 360-LLaMA-Factory

# Switch to the RoPE branch
git checkout llamafactory-native-rope-eval

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation

```bash
# Test basic installation
llamafactory-cli --help

# Quick configuration test
python3 quick_rope_test.py
```

### 3. Test RoPE Parameters

```bash
# Test basic RoPE configuration (should complete in ~2 minutes)
llamafactory-cli train final_rope_example.yaml
```

## Comprehensive Hyperparameter Sweep Setup

### 1. Use Existing Sweep System

The repository includes a complete hyperparameter sweep system:

```bash
# Navigate to comprehensive sweep directory
cd comprehensive_sweep

# Run the full sweep (66 configurations)
python3 fast_eval.py

# Or run focused sweep
python3 run_focused_sweep.py
```

### 2. Generate Custom Configurations

```bash
# Generate new configurations
cd hyperparam_sweep
python3 generate_configs.py

# Run sweep
python3 run_sweep.py
```

## Manual RoPE Testing

### 1. Single Configuration Test

Create a test YAML file:

```yaml
# test_config.yaml
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
rope_scaling: linear
rope_theta: 1000000.0
cutoff_len: 8192
dataset: alpaca_en_demo
template: llama3
stage: sft
do_train: true
finetuning_type: full
output_dir: test_output
max_steps: 1
bf16: true
```

Run test:
```bash
llamafactory-cli train test_config.yaml
```

### 2. Parameter Sweep Script

Create custom sweep script:

```python
#!/usr/bin/env python3
"""Custom RoPE parameter sweep."""

import yaml
import subprocess
from pathlib import Path

# Parameter combinations to test
rope_configs = [
    {"rope_scaling": "linear", "rope_theta": 500000.0, "cutoff_len": 4096},
    {"rope_scaling": "linear", "rope_theta": 1000000.0, "cutoff_len": 8192},  
    {"rope_scaling": "dynamic", "rope_theta": 2000000.0, "cutoff_len": 16384},
]

base_config = {
    "model_name_or_path": "unsloth/Llama-3.2-1B-Instruct",
    "dataset": "alpaca_en_demo",
    "template": "llama3",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "max_steps": 1,
    "bf16": True,
}

for i, rope_config in enumerate(rope_configs):
    config = {**base_config, **rope_config}
    config["output_dir"] = f"sweep_output_{i}"
    
    # Save configuration
    config_file = f"sweep_config_{i}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    print(f"Testing configuration {i+1}/{len(rope_configs)}")
    result = subprocess.run(['llamafactory-cli', 'train', config_file])
    
    if result.returncode == 0:
        print(f"✅ Configuration {i} succeeded")
    else:
        print(f"❌ Configuration {i} failed")
```

## Advanced Sweep Setup

### 1. Multi-Node Distributed Setup

For multiple nodes, use the existing distributed system:

```bash
# On each node, set up the environment
export MASTER_ADDR="your_master_node_ip"
export MASTER_PORT="29500" 
export WORLD_SIZE=4  # Total number of GPUs
export RANK=0        # Node rank (0, 1, 2, 3...)

# Run distributed sweep
python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=4 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    comprehensive_sweep/run_distributed_sweep.py
```

### 2. Slurm Integration

Create slurm job script:

```bash
#!/bin/bash
#SBATCH --job-name=rope-sweep
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load modules
module load cuda/11.8
module load python/3.9

# Setup environment
source ~/venv/bin/activate
cd 360-LLaMA-Factory

# Run sweep
srun python3 comprehensive_sweep/fast_eval.py
```

## Available Sweep Systems

### 1. Fast Evaluation (`comprehensive_sweep/fast_eval.py`)
- **Purpose**: Quick RoPE parameter testing
- **Configurations**: 66 predefined configs
- **Time**: ~2-4 hours
- **Use**: Initial parameter exploration

### 2. Comprehensive Sweep (`hyperparam_sweep/run_sweep.py`)  
- **Purpose**: Full hyperparameter optimization
- **Configurations**: Customizable grid search
- **Time**: 4-24 hours depending on scope
- **Use**: Production parameter tuning

### 3. Focused Sweep (`comprehensive_sweep/run_focused_sweep.py`)
- **Purpose**: Targeted parameter ranges
- **Configurations**: 20-50 focused configs
- **Time**: 1-3 hours
- **Use**: Refinement of known good ranges

## Key Configuration Files

### Ready-to-Use Configurations

1. **`final_rope_example.yaml`** - Optimal production config
2. **`rope_theta_test.yaml`** - High theta testing
3. **`llama32_rope_test.yaml`** - LLaMA 3.2 specific
4. **`comprehensive_rope_config.yaml`** - Advanced settings

### Sweep Configuration Directories

1. **`comprehensive_sweep/focused/`** - Pre-generated configs
2. **`hyperparam_sweep/configs/`** - Grid search configs
3. **`eval_configs/`** - Evaluation-specific configs

## Resource Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (for 1B models)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space

### Recommended for Sweeps
- **GPU**: 24GB+ VRAM (for larger models)
- **RAM**: 32GB+ system RAM  
- **Storage**: 200GB+ free space
- **Network**: Fast internet for model downloads

## Monitoring and Results

### 1. Real-time Monitoring
```bash
# Monitor sweep progress
tail -f comprehensive_sweep/logs/sweep.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### 2. Results Analysis
```bash
# Analyze completed sweep
cd comprehensive_sweep
python3 analyze_comprehensive_results.py

# View results
ls analysis/
# -> comprehensive_report.md
# -> memory_and_perplexity_by_type.png
# -> scaling_factor_effects.png
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `cutoff_len` or use `bf16: true`
2. **Model Download Fails**: Check internet connection, try smaller models first
3. **Permission Errors**: Ensure write access to output directories
4. **CUDA Errors**: Verify CUDA compatibility with PyTorch version

### Quick Fixes

```bash
# Reset environment
pip install --upgrade torch transformers
pip install -e .

# Clear cache
rm -rf ~/.cache/huggingface/

# Test basic functionality
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Production Deployment

### 1. Recommended Workflow

1. **Initial Testing**: Use `final_rope_example.yaml`
2. **Parameter Search**: Run `comprehensive_sweep/fast_eval.py`
3. **Refinement**: Use best configs for focused sweep
4. **Production**: Deploy optimal configuration

### 2. Configuration Management

```bash
# Save successful configurations
mkdir production_configs
cp final_rope_example.yaml production_configs/
cp best_sweep_config.yaml production_configs/

# Version control
git add production_configs/
git commit -m "Add validated production RoPE configurations"
```

This guide provides everything needed to set up and run RoPE hyperparameter sweeps on any node or cluster.