# Pretokenization Guide for Long-Context Training

## Overview

When training with extremely long contexts (>128K tokens), preprocessing can become a major bottleneck. Pretokenization allows you to tokenize your dataset once and reuse it across multiple training runs, saving significant time.

## Why Use Pretokenization?

### Performance Impact by Context Length

| Context Length | Tokenization Speed | Recommended Approach |
|---------------|-------------------|---------------------|
| < 8K tokens   | Fast (~100 samples/sec) | On-the-fly tokenization |
| 8K - 32K      | Moderate (~10 samples/sec) | Optional pretokenization |
| 32K - 128K    | Slow (~1 sample/sec) | Recommended pretokenization |
| > 128K        | Very Slow (<0.1 samples/sec) | **Required pretokenization** |

For 512K context datasets like NExtLong-512K, pretokenization can reduce preprocessing time from hours to seconds!

## Quick Start

### Step 1: Pretokenize Your Dataset

```bash
# Basic usage for NExtLong-512K dataset
python pretokenize_dataset.py \
  --dataset caskcsg/NExtLong-512K-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 524288 \
  --output ./tokenized_data/nextlong_512k

# With validation split
python pretokenize_dataset.py \
  --dataset caskcsg/NExtLong-512K-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 524288 \
  --val-size 0.1 \
  --output ./tokenized_data/nextlong_512k_with_val

# With packing for efficient training
python pretokenize_dataset.py \
  --dataset your-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 32768 \
  --packing \
  --output ./tokenized_data/packed_32k
```

### Step 2: Use in Training YAML

```yaml
# train_pretokenized.yaml
model_name_or_path: meta-llama/Llama-2-7b-hf
template: default

# Use pretokenized dataset - THIS IS THE KEY LINE
tokenized_path: ./tokenized_data/nextlong_512k

# Training configuration
stage: pt
do_train: true
cutoff_len: 524288  # Must match pretokenization cutoff

# No need to specify dataset when using tokenized_path
# dataset: caskcsg/NExtLong-512K-dataset  # Not needed!

# Rest of your configuration
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-5
num_train_epochs: 1
output_dir: ./outputs/nextlong_pretokenized
```

## Profiling Your Dataset

Before deciding on pretokenization, profile your dataset:

```bash
# Run profiling script
python profile_preprocessing.py

# Or test with the profiling YAML
llamafactory-cli train profile_nextlong_512k.yaml
```

## Advanced Usage

### Multiple Datasets

Pretokenize multiple datasets and combine them:

```bash
# Tokenize each dataset
for dataset in dataset1 dataset2 dataset3; do
  python pretokenize_dataset.py \
    --dataset $dataset \
    --model meta-llama/Llama-2-7b-hf \
    --cutoff 131072 \
    --output ./tokenized_data/${dataset}_128k
done
```

### Different Context Lengths

Prepare datasets with various context lengths for curriculum learning:

```bash
# Stage 1: 8K context
python pretokenize_dataset.py \
  --dataset your-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 8192 \
  --output ./tokenized_data/stage1_8k

# Stage 2: 32K context  
python pretokenize_dataset.py \
  --dataset your-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 32768 \
  --output ./tokenized_data/stage2_32k

# Stage 3: 128K context
python pretokenize_dataset.py \
  --dataset your-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 131072 \
  --output ./tokenized_data/stage3_128k
```

## Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset name or local file path | Required |
| `--model` | Model for tokenizer | Required |
| `--cutoff` | Maximum sequence length | 8192 |
| `--output` | Output directory | Required |
| `--max-samples` | Limit number of samples | None (all) |
| `--packing` | Enable sequence packing | False |
| `--template` | Chat template (default/llama3/gemma) | default |
| `--num-proc` | Parallel processes | 4 |
| `--batch-size` | Tokenization batch size | 1000 |
| `--val-size` | Validation split ratio | 0.0 |
| `--streaming` | Use streaming mode | False |

## Important Notes

1. **Consistency**: The model tokenizer used for pretokenization MUST match the model used for training
2. **Cutoff Length**: The cutoff_len in training YAML should match the pretokenization cutoff
3. **Memory**: Pretokenizing very long sequences requires significant RAM (32GB+ for 512K contexts)
4. **Storage**: Pretokenized datasets can be large (10-100GB for large datasets)

## Troubleshooting

### Out of Memory During Pretokenization

- Reduce `--batch-size` (e.g., to 100 or 10)
- Use `--streaming` mode for very large datasets
- Process in chunks with `--max-samples`

### Slow Tokenization

- Increase `--num-proc` for more parallel workers
- Ensure you're using a fast tokenizer (`use_fast=True`)
- Consider reducing `--cutoff` if appropriate

### Training Fails with Pretokenized Data

- Verify tokenizer compatibility between pretokenization and training
- Check that cutoff_len matches in both steps
- Ensure tokenized_path points to the correct directory

## Examples

### Complete Example: NExtLong-512K Training

```bash
# Step 1: Pretokenize (do this once)
python pretokenize_dataset.py \
  --dataset caskcsg/NExtLong-512K-dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 524288 \
  --max-samples 1000 \
  --val-size 0.1 \
  --output ./tokenized_data/nextlong_512k_1k_samples

# Step 2: Create training config
cat > train_nextlong.yaml << EOF
model_name_or_path: meta-llama/Llama-2-7b-hf
tokenized_path: ./tokenized_data/nextlong_512k_1k_samples
stage: pt
do_train: true
cutoff_len: 524288
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
bf16: true
learning_rate: 1e-5
num_train_epochs: 1
output_dir: ./outputs/nextlong_pretokenized
logging_steps: 10
save_steps: 100
eval_strategy: steps
eval_steps: 50
EOF

# Step 3: Train (fast startup, no preprocessing delay!)
llamafactory-cli train train_nextlong.yaml
```

## Performance Benchmarks

Based on our testing with NExtLong-512K dataset:

| Method | Context | Preprocessing Time | Training Startup |
|--------|---------|-------------------|------------------|
| On-the-fly | 512K | N/A | >2 hours |
| Pretokenized | 512K | 30 min (once) | <1 minute |

The pretokenization investment pays off after just 2-3 training runs!