# RoPE Hyperparameter Sweep

A comprehensive system for finding optimal RoPE scaling configurations for long context extension using perplexity and LongPPL metrics.

## Quick Start

### 1. Generate Configurations

```bash
# Generate basic sweep (fast model, short+medium contexts)
python hyperparam_sweep/run_sweep.py --generate --model fast --contexts short medium

# Generate comprehensive sweep (all rope types, longer contexts)  
python hyperparam_sweep/run_sweep.py --generate --model medium --contexts short medium long --rope-types linear yarn llama3

# Check generated configurations
ls configs/
cat configs/manifest.yaml
```

### 2. Run Evaluation

```bash
# Single GPU evaluation
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 1

# Multi-GPU parallel evaluation  
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 4

# Resume interrupted sweep
python hyperparam_sweep/run_sweep.py --evaluate --resume --num-gpus 4
```

### 3. Analyze Results

```bash
# Generate analysis and visualizations
python hyperparam_sweep/run_sweep.py --analyze

# Create detailed visualizations
python hyperparam_sweep/visualize_results.py --results-dir results --output-dir analysis
```

## Detailed Usage

### Configuration Generation

The system supports comprehensive parameter sweeps across multiple dimensions:

```bash
python hyperparam_sweep/generate_configs.py --help

# Examples:
# Fast iteration (GPT-2, short contexts)
python hyperparam_sweep/generate_configs.py --model fast --contexts short medium

# Production sweep (Llama-3-8B, all contexts)  
python hyperparam_sweep/generate_configs.py --model large --contexts short medium long very_long

# Specific RoPE types only
python hyperparam_sweep/generate_configs.py --rope-types yarn llama3 --estimate
```

**Configuration Space:**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `rope_scaling_type` | linear, dynamic, yarn, longrope, llama3 | RoPE scaling method |
| `rope_scaling_factor` | 2, 4, 8, 16, 32 | Context extension multiplier |
| `yarn_alpha` | 0.5, 1.0, 2.0, 4.0 | YARN attention scaling |
| `yarn_beta` | 8.0, 16.0, 32.0, 64.0 | YARN position interpolation |
| `context_length` | 2K, 4K, 8K, 16K, 32K, 64K, 128K | Target context length |

### Evaluation Pipeline

The evaluation system measures multiple metrics:

```bash
python hyperparam_sweep/evaluate_perplexity.py config.yaml --dataset wikitext

# Evaluate entire directory
python hyperparam_sweep/evaluate_perplexity.py configs/ --dataset c4
```

**Metrics Computed:**
- **Perplexity**: Standard language modeling quality
- **LongPPL**: Perplexity on long sequences using sliding window
- **Memory Usage**: Peak GPU memory consumption  
- **Throughput**: Tokens processed per second
- **Needle Accuracy**: Optional retrieval accuracy test

### Parallel Execution

Efficient multi-GPU evaluation with automatic work distribution:

```bash
# 4 GPUs in parallel
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 4 --max-workers 4

# Resume from checkpoint
python hyperparam_sweep/run_sweep.py --evaluate --resume

# Custom config directory
python hyperparam_sweep/run_sweep.py --evaluate --config-dir my_configs
```

### Results Analysis

Comprehensive analysis with visualizations:

```bash
# Basic analysis
python hyperparam_sweep/run_sweep.py --analyze

# Detailed visualizations  
python hyperparam_sweep/visualize_results.py --results-dir results

# Custom output directory
python hyperparam_sweep/visualize_results.py --output-dir my_analysis
```

**Generated Visualizations:**
1. **PPL vs Context Length** - Performance across context sizes
2. **Scaling Factor Analysis** - Impact of rope_scaling_factor  
3. **YARN Hyperparameters** - Alpha/beta parameter effects
4. **Pareto Frontier** - Quality vs compute cost tradeoffs
5. **Summary Statistics** - Comprehensive performance table

## Example Workflows

### 1. Quick Exploration (1-2 hours)
```bash
# Generate small sweep
python hyperparam_sweep/run_sweep.py --generate --model fast --contexts short

# Evaluate on single GPU  
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 1

# Analyze results
python hyperparam_sweep/run_sweep.py --analyze
```

### 2. Comprehensive Research (1-2 days)
```bash
# Generate full sweep
python hyperparam_sweep/run_sweep.py --generate --model medium --contexts short medium long --rope-types linear dynamic yarn llama3

# Parallel evaluation
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 8 --max-workers 8

# Full analysis
python hyperparam_sweep/visualize_results.py
```

### 3. Production Optimization (3-5 days)
```bash
# Large model comprehensive sweep
python hyperparam_sweep/run_sweep.py --generate --model large --contexts short medium long very_long extreme

# Distributed evaluation  
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 16 --max-workers 16

# Detailed analysis
python hyperparam_sweep/visualize_results.py --show-plots
```

## Expected Results

### Performance Characteristics

**Linear Scaling:**
- Simple and stable
- Good for moderate context extensions (2-8x)
- Linear memory scaling

**YARN Scaling:**
- Best for extreme context lengths (16x+)
- Tunable alpha/beta parameters
- Higher memory usage but better quality

**LLaMA 3 Scaling:**
- Optimized for LLaMA architecture
- Good balance of quality and efficiency
- Recommended for LLaMA-based models

### Typical Outcomes

| Method | 4K Context | 16K Context | 64K Context |
|--------|------------|-------------|-------------|
| Baseline | PPL: 10.5 | PPL: ∞ | PPL: ∞ |
| Linear | PPL: 11.2 | PPL: 13.8 | PPL: 18.5 |
| YARN | PPL: 10.8 | PPL: 12.1 | PPL: 14.2 |
| LLaMA 3 | PPL: 10.9 | PPL: 12.5 | PPL: 15.8 |

## File Structure

```
hyperparam_sweep/
├── README.md                    # This file
├── generate_configs.py          # Configuration generator
├── evaluate_perplexity.py       # Single config evaluation
├── run_sweep.py                # Main orchestration script
└── visualize_results.py        # Results analysis and visualization

configs/                        # Generated configurations
├── manifest.yaml               # Configuration metadata
├── baseline_short_abc123.yaml  # Individual config files
└── yarn_medium_def456.yaml

results/                        # Evaluation outputs
├── eval_baseline_short.json    # Individual results
├── eval_yarn_medium.json
└── sweep_summary.json          # Aggregated analysis

analysis/                       # Visualization outputs  
├── summary_statistics.csv      # Performance table
├── ppl_vs_context.png          # Main results plot
├── scaling_analysis.png        # Parameter sensitivity
└── pareto_frontier.png         # Efficiency analysis
```

## Compute Requirements

### Resource Estimation

| Sweep Size | Configs | GPU Hours | Memory (per GPU) |
|------------|---------|-----------|------------------|
| Quick | 20-50 | 10-25 | 16-32 GB |
| Medium | 100-200 | 50-100 | 32-48 GB |  
| Comprehensive | 500+ | 250+ | 48-80 GB |

### Optimization Tips

1. **Start Small**: Use `--model fast` for initial exploration
2. **Parallel Processing**: Use all available GPUs with `--num-gpus`
3. **Resume Capability**: Use `--resume` for interrupted runs
4. **Memory Management**: Monitor with `nvidia-smi` during evaluation
5. **Early Stopping**: Implement custom logic for poor configurations

## Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size or context length
# Use smaller model for initial sweep
python hyperparam_sweep/generate_configs.py --model fast
```

**Slow Evaluation:**
```bash  
# Increase parallelism
python hyperparam_sweep/run_sweep.py --evaluate --num-gpus 4 --max-workers 4

# Reduce max_samples in configs
```

**Failed Configurations:**
```bash
# Resume with error handling
python hyperparam_sweep/run_sweep.py --evaluate --resume
```

### Support

For issues or questions:
1. Check configuration files in `configs/manifest.yaml`  
2. Review evaluation logs for specific errors
3. Verify GPU memory availability with `nvidia-smi`
4. Test single configuration before full sweep