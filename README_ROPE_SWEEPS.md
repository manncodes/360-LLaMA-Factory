# RoPE Hyperparameter Sweeps - Quick Start Guide

## One-Command Setup on New Node

```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/manncodes/360-LLaMA-Factory/llamafactory-native-rope-eval/quick_node_setup.sh | bash
```

Or manual setup:
```bash
git clone https://github.com/manncodes/360-LLaMA-Factory.git
cd 360-LLaMA-Factory
git checkout llamafactory-native-rope-eval
./quick_node_setup.sh
```

## Quick Test (2 minutes)

```bash
# Activate environment
source venv/bin/activate

# Quick RoPE test
python3 sweep_runner.py --preset quick
```

## Available Sweep Types

### 1. Quick Sweep (5 minutes)
```bash
python3 sweep_runner.py --preset quick
# Tests: 2 configurations, 2 steps each
# Good for: Initial testing, CI/CD
```

### 2. Standard Sweep (15-30 minutes)  
```bash
python3 sweep_runner.py --preset standard
# Tests: 4 configurations, 5 steps each
# Good for: Development, parameter exploration
```

### 3. Comprehensive Sweep (1-3 hours)
```bash
python3 sweep_runner.py --preset comprehensive  
# Tests: 12 configurations, 10 steps each
# Good for: Production optimization
```

### 4. Use Existing Full Sweep (2-4 hours)
```bash
python3 comprehensive_sweep/fast_eval.py
# Tests: 66 pre-configured combinations
# Good for: Complete parameter space exploration
```

## Custom Configuration

Create `my_sweep.yaml`:
```yaml
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
dataset: alpaca_en_demo
template: llama3
max_steps: 5
bf16: true
results_dir: my_sweep_results

rope_configurations:
  - rope_scaling: linear
    rope_theta: 500000.0
    cutoff_len: 4096
  - rope_scaling: linear  
    rope_theta: 1000000.0
    cutoff_len: 8192
  - rope_scaling: dynamic
    rope_theta: 2000000.0
    cutoff_len: 16384
```

Run custom sweep:
```bash
python3 sweep_runner.py --config my_sweep.yaml
```

## Results Analysis

```bash
# View results
ls sweep_results/
# -> sweep_summary.json  (main results)
# -> config_*.yaml       (individual configs)
# -> run_*/               (training outputs)

# Analyze comprehensive results  
cd comprehensive_sweep
python3 analyze_comprehensive_results.py
ls analysis/
# -> comprehensive_report.md
# -> *.png (visualizations)
```

## Single Configuration Test

Test specific RoPE settings:
```bash
# Use pre-made configs
llamafactory-cli train final_rope_example.yaml      # Production ready
llamafactory-cli train rope_theta_test.yaml         # High theta test  
llamafactory-cli train llama32_rope_test.yaml       # LLaMA 3.2 specific

# Validate config first
python3 quick_rope_test.py
```

## Multi-Node Setup

### SLURM Cluster
```bash
#!/bin/bash
#SBATCH --job-name=rope-sweep
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load cuda python
source ~/venv/bin/activate
cd 360-LLaMA-Factory

# Run distributed sweep
srun python3 sweep_runner.py --preset comprehensive
```

### Manual Distributed
```bash
# On each node:
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 comprehensive_sweep/fast_eval.py
```

## Monitoring

```bash
# Watch progress
tail -f sweep_results/sweep_summary.json

# Monitor GPU
watch -n 1 nvidia-smi

# Check specific run
tail -f sweep_results/run_0/training_log.txt
```

## Key Features

### RoPE Parameters Available
- **`rope_scaling`**: `"linear"` or `"dynamic"`  
- **`rope_theta`**: `500000.0`, `1000000.0`, `2000000.0`
- **`cutoff_len`**: `4096`, `8192`, `16384`, `32768`

### Model Support
- ✅ `unsloth/Llama-3.2-1B-Instruct` (recommended)
- ✅ `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 
- ✅ Any LLaMA-architecture model with RoPE support

### Memory Requirements
- **1B models**: 8GB+ VRAM
- **3B models**: 16GB+ VRAM  
- **7B+ models**: 24GB+ VRAM

### Automatic Features
- Configuration validation
- Error handling and timeouts
- Progress tracking
- Results aggregation
- GPU memory optimization

## Troubleshooting

### Common Issues
```bash
# Out of memory -> reduce context
cutoff_len: 2048  # instead of 8192

# Model download fails -> check connection
curl -I https://huggingface.co

# CUDA issues -> verify installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Permission errors -> check directories
chmod -R 755 360-LLaMA-Factory/
```

### Quick Fixes
```bash
# Reset environment
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Clear caches
rm -rf ~/.cache/huggingface/
rm -rf sweep_results/
```

## Production Recommendations

1. **Start with**: `python3 sweep_runner.py --preset quick`
2. **Scale to**: `python3 sweep_runner.py --preset standard`  
3. **Optimize with**: Custom configuration based on best results
4. **Deploy**: Use `final_rope_example.yaml` as template

## Support Files

- **`SETUP_AND_DEPLOYMENT_GUIDE.md`**: Detailed setup instructions
- **`ROPE_CONFIGURATION_GUIDE.md`**: Parameter documentation
- **`quick_node_setup.sh`**: Automated setup script
- **`sweep_runner.py`**: Unified sweep execution
- **`quick_rope_test.py`**: Configuration validation

This system provides everything needed to run RoPE hyperparameter sweeps on any node or cluster efficiently.