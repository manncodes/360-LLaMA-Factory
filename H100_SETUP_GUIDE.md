# RoPE Evaluation with 8x H100 GPUs - Maximum Performance Guide

This guide shows how to exploit the full potential of 8x NVIDIA H100 GPUs for RoPE needle-in-haystack evaluation with extreme long contexts.

## System Requirements

### Hardware
- **8x NVIDIA H100 GPUs** (80GB VRAM each = 640GB total)
- **High-speed GPU interconnect** (NVLink 4.0 preferred)
- **CPU**: 64+ cores (Intel Xeon Platinum 8480+ or AMD EPYC 9654)
- **RAM**: 512GB+ DDR5
- **Storage**: NVMe SSD with 10TB+ free space
- **Network**: InfiniBand or 100GbE for multi-node (optional)

### Software Requirements
- **CUDA**: 12.2+
- **Driver**: 535.54+
- **Python**: 3.10+
- **PyTorch**: 2.1.0+ with CUDA 12.2
- **DeepSpeed**: 0.12.0+
- **Transformers**: 4.49.0+

## Installation

### 1. Environment Setup

```bash
# Create isolated environment
conda create -n h100_rope python=3.10
conda activate h100_rope

# Install PyTorch with CUDA 12.2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers>=4.49.0 datasets accelerate deepspeed
pip install flash-attn ring-flash-attn --no-build-isolation
pip install triton xformers

# Clone and setup 360-LLaMA-Factory
git clone https://github.com/manncodes/360-LLaMA-Factory.git
cd 360-LLaMA-Factory
git checkout rope-needle-eval-feature
pip install -e .
```

### 2. Verify H100 Setup

```bash
# Check GPU visibility
nvidia-smi

# Verify CUDA and PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test sequence parallelism dependencies
python -c "from ring_flash_attn import zigzag_ring_flash_attn_func; print('Ring Flash Attention: OK')"
```

## Quick Start Commands

### 1. Maximum Performance (8x H100)

```bash
# 70B model with extreme contexts (up to 256k tokens)
python run_rope_needle_eval.py \
    --config examples/rope_eval_h100_8gpu.yaml \
    --model "meta-llama/Llama-2-70b-hf" \
    --sequence-parallel-size 8 \
    --sequence-parallel-mode zigzag-ring
```

### 2. Balanced Setup (4x H100)

```bash
# 13B model with balanced contexts (up to 128k tokens)
python run_rope_needle_eval.py \
    --config examples/rope_eval_h100_4gpu.yaml \
    --model "meta-llama/Llama-2-13b-hf" \
    --sequence-parallel-size 4
```

### 3. Extreme Evaluation (Push Limits)

```bash
# Absolutely maximum contexts (up to 1M tokens)
python run_rope_needle_eval.py \
    --config examples/rope_eval_h100_extreme.yaml \
    --sequence-parallel-size 8 \
    --contexts 32768 65536 131072 262144 524288 1048576
```

## Configuration Examples

### High-Performance Configuration (8x H100)

```yaml
# examples/rope_eval_h100_8gpu.yaml
model_name: "meta-llama/Llama-2-70b-hf"

rope_techniques:
  - baseline
  - linear
  - dynamic
  - yarn
  - longrope

context_lengths:
  - 8192
  - 16384
  - 32768
  - 65536
  - 131072
  - 262144

sequence_parallel_config:
  sequence_parallel_size: 8
  sequence_parallel_mode: "zigzag-ring"
  flash_attn: "fa2"
  gradient_checkpointing: true
  bf16: true
  deepspeed: "examples/deepspeed/ds_z3_offload_config.json"
```

### Memory-Efficient Configuration

```yaml
# For maximum context lengths
sequence_parallel_config:
  sequence_parallel_size: 8
  sequence_parallel_mode: "zigzag-ring"
  deepspeed: "examples/deepspeed/ds_z3_offload_config.json"

memory_optimizations:
  offload_optimizer: true
  offload_params: true
  cpu_offload: true
  zero_stage: 3
  activation_checkpointing: true
```

## Performance Optimization Tips

### 1. Memory Management

```bash
# Set optimal environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
```

### 2. Launch with Torchrun (Recommended)

```bash
# Multi-GPU launch
torchrun --standalone --nproc_per_node=8 run_rope_needle_eval.py \
    --config examples/rope_eval_h100_8gpu.yaml \
    --sequence-parallel-size 8
```

### 3. DeepSpeed Launch

```bash
# Alternative launch method
deepspeed --num_gpus=8 run_rope_needle_eval.py \
    --config examples/rope_eval_h100_8gpu.yaml \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json
```

## Context Length Scaling Guide

| Model Size | SP Size | Max Context | Memory Usage | Time Est. |
|------------|---------|-------------|--------------|-----------|
| 7B         | 2       | 64K         | 20GB/GPU     | 30 min    |
| 13B        | 4       | 128K        | 40GB/GPU     | 1 hour    |
| 30B        | 4       | 128K        | 60GB/GPU     | 2 hours   |
| 70B        | 8       | 256K        | 75GB/GPU     | 4 hours   |
| 70B        | 8       | 512K        | 79GB/GPU     | 8 hours   |
| 70B        | 8       | 1M          | 79GB/GPU     | 16 hours  |

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
batch_size: 1

# Enable more aggressive offloading
sequence_parallel_config:
  deepspeed: "examples/deepspeed/ds_z3_offload_config.json"

memory_optimizations:
  offload_optimizer: true
  offload_params: true
  cpu_offload: true
```

### Slow Performance

```bash
# Verify high-speed interconnect
nvidia-smi topo -m

# Check NCCL settings
export NCCL_DEBUG=INFO

# Use faster sequence parallel mode
sequence_parallel_mode: "zigzag-ring"  # Fastest for ring topology
```

### CUDA Errors

```bash
# Reset GPU state
nvidia-smi --gpu-reset

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Check temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv
```

## Advanced Techniques

### 1. Mixed Model Evaluation

```python
# Compare different model families
models = [
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mixtral-8x7B-v0.1", 
    "01-ai/Yi-34B"
]

for model in models:
    run_evaluation(model, context_lengths=[32768, 65536, 131072])
```

### 2. Custom RoPE Parameters

```yaml
# Fine-tune RoPE scaling
rope_theta_analysis:
  enabled: true
  base_values: [10000, 50000, 500000, 1000000, 16000000]

extreme_rope_configs:
  max_scaling_factor: 256
  adaptive_scaling: true
  theta_adjustment: true
```

### 3. Performance Monitoring

```yaml
monitoring:
  gpu_utilization: true
  memory_usage: true
  temperature_monitoring: true
  power_consumption: true

safety:
  memory_threshold: 0.95
  temperature_threshold: 85
  auto_recovery: true
```

## Expected Results

### Performance Benchmarks (8x H100)

- **Context 32K**: ~15 min/technique for 70B model
- **Context 64K**: ~30 min/technique 
- **Context 128K**: ~1 hour/technique
- **Context 256K**: ~2 hours/technique
- **Context 512K**: ~4 hours/technique (memory-limited)

### Memory Utilization

- **Optimal**: 75-80% per GPU
- **Peak**: Up to 95% for extreme contexts
- **Efficiency**: >90% GPU utilization with proper tuning

### Throughput

- **70B model**: ~50-100 tokens/second at 64K context
- **13B model**: ~200-400 tokens/second at 64K context
- **Scaling**: Near-linear with sequence parallelism

## Cost Optimization

### Cloud Instances (Estimated Costs)

- **AWS p5.48xlarge**: $98/hour (8x H100)
- **GCP a3-megagpu-8g**: $90/hour  
- **Azure ND96amsr_A100_v4**: $85/hour

### Recommended Evaluation Strategy

1. **Start small**: Test with 4 GPUs, shorter contexts
2. **Scale gradually**: Increase to 8 GPUs for extreme contexts
3. **Batch efficiently**: Group similar context lengths
4. **Monitor costs**: Set time limits and automatic shutdowns

## Contact & Support

- **Issues**: https://github.com/manncodes/360-LLaMA-Factory/issues
- **Branch**: `rope-needle-eval-feature`
- **Discord**: Join 360 AI community

## Citation

```bibtex
@misc{h100_rope_eval_2024,
  title={Large-Scale RoPE Evaluation on H100 Systems},
  author={360-LLaMA-Factory Team},
  year={2024},
  url={https://github.com/manncodes/360-LLaMA-Factory}
}
```