# RoPE Needle-in-Haystack Evaluation Setup Guide

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for models and datasets
- **OS**: Linux (Ubuntu 20.04+) or Windows with WSL2

### Software Requirements
- Python 3.9+
- CUDA 11.8+ with cuDNN
- Git

## Installation Steps

### 1. Clone the Repository

```bash
# Clone the fork with the rope-needle-eval feature
git clone https://github.com/manncodes/360-LLaMA-Factory.git
cd 360-LLaMA-Factory

# Checkout the feature branch
git checkout rope-needle-eval-feature
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv rope_eval_env
source rope_eval_env/bin/activate  # On Windows: rope_eval_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Install additional dependencies for RoPE evaluation
pip install transformers>=4.49.0 datasets accelerate matplotlib seaborn pandas pyyaml tqdm
```

### 4. Download PaulGraham Essays (Optional)

The evaluation uses PaulGraham essays as the haystack. If not present, it will use generic text.

```bash
# Clone the needle-in-haystack repo for PaulGraham essays
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git
# The evaluator will automatically find the essays
```

## Running Evaluations

### Quick Test (Small Model)

```bash
# Test with TinyLlama (1.1B parameters, ~2GB VRAM)
python run_rope_needle_eval.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --techniques baseline linear \
    --contexts 1024 2048 \
    --output-dir ./test_results
```

### Standard Evaluation (7B Model)

```bash
# Run with Llama-2-7B (requires ~14GB VRAM)
python run_rope_needle_eval.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --techniques baseline linear dynamic yarn \
    --contexts 2048 4096 8192 16384 \
    --output-dir ./llama2_results
```

### Using Configuration Files

```bash
# Basic configuration
python run_rope_needle_eval.py --config examples/rope_eval_config.yaml

# Advanced configuration with more techniques
python run_rope_needle_eval.py --config examples/rope_eval_advanced.yaml
```

### Custom Configuration

Create your own YAML config:

```yaml
# my_config.yaml
model_name: "mistralai/Mistral-7B-v0.1"  # Your model choice

rope_techniques:
  - baseline
  - linear
  - dynamic
  - yarn

context_lengths:
  - 4096
  - 8192
  - 16384
  - 32768

base_context_length: 4096

depth_percents:
  - 0.1
  - 0.5
  - 0.9

samples_per_needle: 3  # More samples for better statistics

device: "cuda"  # or "cpu" for testing

output_dir: "./my_results"
```

Run with:
```bash
python run_rope_needle_eval.py --config my_config.yaml
```

## Memory Optimization

### For Limited VRAM (4-8GB)

1. **Use smaller models**:
   - TinyLlama-1.1B (~2GB)
   - Phi-2 (~3GB)
   - Qwen-1.8B (~4GB)

2. **Reduce context lengths**:
   ```bash
   --contexts 512 1024 2048
   ```

3. **Use CPU offloading** (slower but works):
   ```bash
   --device cpu
   ```

4. **Use 8-bit quantization**:
   ```python
   # Modify in code: dtype=torch.int8 instead of torch.float16
   ```

### For High-End GPUs (24GB+ VRAM)

Run larger models and contexts:

```bash
python run_rope_needle_eval.py \
    --model "meta-llama/Llama-2-13b-hf" \
    --techniques baseline linear dynamic yarn longrope \
    --contexts 4096 8192 16384 32768 65536 \
    --output-dir ./large_model_results
```

## Interpreting Results

Results are saved in JSON format:

```
output_dir/
├── results_YYYYMMDD_HHMMSS.json    # Detailed results
└── summary_YYYYMMDD_HHMMSS.json    # Aggregated summary
```

### Summary Structure

```json
{
  "by_technique": {
    "linear": {"avg_accuracy": 0.85},
    "dynamic": {"avg_accuracy": 0.82}
  },
  "by_context_length": {
    "8192": {
      "best_technique": "linear",
      "best_accuracy": 0.90,
      "all_techniques": {...}
    }
  }
}
```

### Key Metrics

- **Accuracy**: Percentage of times the needle was successfully retrieved
- **Best Technique**: Which RoPE method works best for each context length
- **Performance by Depth**: How well the model finds needles at different positions

## Visualization

Create plots from results:

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open('output_dir/summary_*.json') as f:
    data = json.load(f)

# Plot accuracy by technique
techniques = list(data['by_technique'].keys())
accuracies = [data['by_technique'][t]['avg_accuracy'] for t in techniques]

plt.bar(techniques, accuracies)
plt.xlabel('RoPE Technique')
plt.ylabel('Average Accuracy')
plt.title('RoPE Technique Performance')
plt.show()
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size (in code)
# Reduce context lengths
--contexts 512 1024

# Use smaller model
--model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Model Not Found

```bash
# Login to HuggingFace for gated models
huggingface-cli login

# Or use public models
--model "NousResearch/Llama-2-7b-hf"  # Public Llama-2
```

### Slow Performance

- Ensure CUDA is properly installed: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Use faster storage (SSD) for model cache

## Advanced Usage

### Testing Custom RoPE Parameters

Edit `src/llamafactory/rope_eval/rope_config.py` to add custom RoPE configurations:

```python
elif technique == "custom":
    return RoPEConfig(
        rope_type="linear",
        factor=3.5,  # Custom factor
        rope_theta=100000  # Custom theta
    )
```

### Batch Processing Multiple Models

Create a script:

```bash
#!/bin/bash
models=("TinyLlama/TinyLlama-1.1B-Chat-v1.0" "mistralai/Mistral-7B-v0.1")

for model in "${models[@]}"; do
    echo "Evaluating $model"
    python run_rope_needle_eval.py \
        --model "$model" \
        --techniques baseline linear dynamic \
        --contexts 2048 4096 8192 \
        --output-dir "./results/${model##*/}"
done
```

## Support

- **Issues**: Report at https://github.com/manncodes/360-LLaMA-Factory/issues
- **Branch**: `rope-needle-eval-feature`
- **Main Script**: `run_rope_needle_eval.py`

## Citation

If you use this evaluation framework:

```bibtex
@misc{rope_needle_eval_360llama,
  title={RoPE Needle-in-Haystack Evaluation for 360-LLaMA-Factory},
  author={Your Name},
  year={2024},
  url={https://github.com/manncodes/360-LLaMA-Factory/tree/rope-needle-eval-feature}
}
```