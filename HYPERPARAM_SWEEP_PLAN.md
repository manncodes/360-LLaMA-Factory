# Hyperparameter Sweep for Optimal Long Context Extension

## Objective
Find optimal RoPE scaling configurations for extending context length while maintaining model quality, using perplexity (PPL) and LongPPL as primary metrics.

## 1. Search Space Definition

### RoPE Scaling Types
```yaml
rope_scaling_types:
  - linear       # Simple linear interpolation
  - dynamic      # Dynamic scaling
  - yarn         # Yet Another RoPE extensioN
  - longrope     # LongRoPE method
  - llama3       # LLaMA 3 style scaling
```

### Parameter Ranges

#### Universal Parameters
```yaml
rope_scaling_factor: [2, 4, 8, 16, 32]  # Context extension multipliers
context_lengths: [2048, 4096, 8192, 16384, 32768, 65536, 131072]
```

#### YARN-Specific Parameters
```yaml
yarn_alpha: [0.5, 1.0, 2.0, 4.0]  # Attention scaling
yarn_beta: [8.0, 16.0, 32.0, 64.0]  # Position interpolation
```

#### LongRoPE-Specific Parameters
```yaml
longrope_short_factor: [1.0, 1.5, 2.0]
longrope_long_factor: [1.0, 2.0, 4.0]
```

## 2. Evaluation Metrics

### Primary Metrics
1. **Perplexity (PPL)**
   - Standard perplexity on validation set
   - Measures general language modeling quality

2. **LongPPL**
   - Perplexity specifically on long sequences
   - Sliding window evaluation for sequences > original context

### Secondary Metrics
3. **Needle-in-Haystack Accuracy**
   - Retrieval accuracy at different depths
   - Tests positional understanding

4. **Memory Usage**
   - Peak GPU memory per configuration
   - Throughput (tokens/second)

## 3. Evaluation Datasets

### For Perplexity
- **PG19** - Long-form books dataset
- **ArXiv** - Scientific papers (long technical content)
- **BookCorpus** - General long-form text

### For LongPPL
- **SCROLLS** - Long context understanding
- **LongBench** - Comprehensive long context benchmark
- **Custom sliding window on C4**

## 4. Experiment Configuration

### Base Models (Choose based on resources)
```yaml
models:
  fast_iteration:
    - pythia-1.4b     # Quick experiments
    - gpt2-large      # Baseline
  
  production:
    - llama-3-8b      # Production quality
    - mistral-7b      # Alternative architecture
```

### Sweep Strategy

#### Phase 1: Coarse Grid Search
- Test all scaling types with default parameters
- Context lengths: [4K, 16K, 64K]
- Measure PPL and LongPPL

#### Phase 2: Fine-Grained Search
- Top 3 configurations from Phase 1
- Refined parameter ranges
- Extended evaluation on all metrics

#### Phase 3: Ablation Studies
- Individual parameter impact
- Interaction effects
- Compute vs quality tradeoffs

## 5. Implementation Plan

### Directory Structure
```
hyperparam_sweep/
├── configs/           # Generated configurations
├── scripts/           # Evaluation scripts
├── results/           # Raw results
├── analysis/          # Processed results & plots
└── logs/             # Experiment logs
```

### Tracking System
```python
experiment_id: {model}_{rope_type}_{factor}_{context}_{timestamp}
results_format:
  - config: Full configuration
  - metrics: {ppl, long_ppl, needle_acc, memory_gb, throughput}
  - artifacts: Model checkpoints if fine-tuning
```

## 6. Computational Estimates

### Per Configuration
- Evaluation time: ~30-60 minutes (depending on context)
- GPU memory: 16-80GB (model and context dependent)

### Total Grid
- Configurations: ~200-500 (depending on granularity)
- Total compute: 100-500 GPU hours

### Optimization Strategies
1. **Early stopping** - Skip if PPL > threshold
2. **Cascade evaluation** - Start with shortest contexts
3. **Parallel execution** - Multiple GPUs/nodes
4. **Caching** - Reuse computed activations

## 7. Expected Outcomes

### Deliverables
1. **Optimal configuration per context length**
2. **Pareto frontier plots** (quality vs compute)
3. **Scaling laws** for each method
4. **Best practices guide**

### Success Metrics
- Find configuration with:
  - < 10% PPL degradation at 4x context
  - < 25% PPL degradation at 16x context
  - Maintain > 80% needle accuracy

## 8. Next Steps

1. [ ] Select base model based on available compute
2. [ ] Implement evaluation pipeline
3. [ ] Generate configuration files
4. [ ] Set up distributed execution
5. [ ] Create visualization dashboard