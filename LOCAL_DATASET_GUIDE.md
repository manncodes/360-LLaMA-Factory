# Local Dataset Guide for LlamaFactory

## Overview

You can use locally downloaded datasets with LlamaFactory in multiple ways. This is especially useful for:
- Offline training environments
- Large datasets you don't want to re-download
- Custom preprocessed datasets
- Datasets stored on fast local storage

## Quick Start: Using Local Datasets

### Option 1: Download Dataset Locally First

```bash
# Download NExtLong dataset to local directory
python prepare_local_dataset.py \
  --dataset caskcsg/NExtLong-512K-dataset \
  --output ./data/nextlong_local \
  --format jsonl \
  --max-samples 1000
```

This creates:
- `./data/nextlong_local/caskcsg_NExtLong-512K-dataset.jsonl` - The dataset
- `./data/nextlong_local/caskcsg_NExtLong-512K-dataset_info.json` - Dataset info entry
- `./data/nextlong_local/train_caskcsg_NExtLong-512K-dataset.yaml` - Example config

### Option 2: Use Existing Local Directory

If you already have the dataset downloaded (e.g., from HuggingFace cache or manual download):

#### For Arrow/HuggingFace Dataset Format:
```bash
# If your dataset is in Arrow format (saved with dataset.save_to_disk())
# Structure: /path/to/dataset/
#   ├── dataset_info.json
#   ├── data-00000-of-00001.arrow
#   └── state.json

# Pretokenize directly from local directory
python pretokenize_dataset.py \
  --dataset /path/to/your/local/dataset \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 131072 \
  --output ./tokenized_data/local_dataset
```

#### For JSON/JSONL/Parquet Files:
```bash
# Single file
python pretokenize_dataset.py \
  --dataset /path/to/dataset.jsonl \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 131072 \
  --output ./tokenized_data/local_dataset

# Directory with multiple files
python pretokenize_dataset.py \
  --dataset /path/to/dataset_dir/ \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 131072 \
  --output ./tokenized_data/local_dataset
```

## Setting Up dataset_info.json

Add your local dataset to `data/dataset_info.json`:

```json
{
  "nextlong_local": {
    "file_name": "nextlong_dataset",
    "columns": {
      "prompt": "text"
    }
  },
  "nextlong_local_jsonl": {
    "file_name": "datasets/nextlong.jsonl",
    "columns": {
      "prompt": "text"
    }
  },
  "nextlong_local_dir": {
    "file_name": "datasets/nextlong_chunks/",
    "columns": {
      "prompt": "text"
    }
  }
}
```

## Training Configuration

### Using Local Dataset in YAML:

```yaml
# train_local.yaml
model_name_or_path: meta-llama/Llama-2-7b-hf

# Method 1: Using dataset name from dataset_info.json
dataset: nextlong_local
dataset_dir: ./data  # Where to find the file specified in dataset_info.json

# Method 2: Using pretokenized local dataset
tokenized_path: ./tokenized_data/local_dataset

stage: pt
cutoff_len: 131072
per_device_train_batch_size: 1
output_dir: ./outputs/local_training
```

## Complete Workflow Example

### Step 1: Download Dataset Locally

```bash
# Download with HuggingFace datasets
from datasets import load_dataset

dataset = load_dataset("caskcsg/NExtLong-512K-dataset")
dataset.save_to_disk("./data/nextlong_arrow")

# Or save as JSONL
dataset["train"].to_json("./data/nextlong.jsonl", lines=True)
```

### Step 2: Pretokenize (Optional but Recommended for Long Context)

```bash
# From Arrow format
python pretokenize_dataset.py \
  --dataset ./data/nextlong_arrow \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 524288 \
  --output ./tokenized_data/nextlong_512k

# From JSONL
python pretokenize_dataset.py \
  --dataset ./data/nextlong.jsonl \
  --model meta-llama/Llama-2-7b-hf \
  --cutoff 524288 \
  --output ./tokenized_data/nextlong_512k
```

### Step 3: Train

```yaml
# train_config.yaml
model_name_or_path: meta-llama/Llama-2-7b-hf
tokenized_path: ./tokenized_data/nextlong_512k
stage: pt
cutoff_len: 524288
# ... rest of config
```

```bash
llamafactory-cli train train_config.yaml
```

## Directory Structure Examples

### Example 1: Single File
```
data/
├── dataset_info.json
└── nextlong.jsonl
```

### Example 2: Directory with Multiple Files
```
data/
├── dataset_info.json
└── nextlong_chunks/
    ├── chunk_001.jsonl
    ├── chunk_002.jsonl
    └── chunk_003.jsonl
```

### Example 3: Arrow Format
```
data/
├── dataset_info.json
└── nextlong_arrow/
    ├── dataset_info.json
    ├── data-00000-of-00001.arrow
    └── state.json
```

### Example 4: Pretokenized
```
tokenized_data/
└── nextlong_512k/
    ├── train/
    │   ├── data-00000-of-00001.arrow
    │   └── dataset_info.json
    ├── validation/
    │   ├── data-00000-of-00001.arrow
    │   └── dataset_info.json
    └── metadata.json
```

## Performance Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **HuggingFace Hub** | Always latest, no storage | Slow download, needs internet | Small datasets, testing |
| **Local Files** | Fast access, offline | Storage space, manual updates | Production, large datasets |
| **Pretokenized Local** | Fastest loading, offline | Large storage, fixed tokenization | Long-context (>128K), production |

## Tips and Best Practices

1. **For Large Datasets**: Always use local + pretokenized
2. **Storage Location**: Use fast SSDs for dataset storage
3. **Format Choice**: 
   - JSONL: Good for streaming, human-readable
   - Parquet: Efficient compression, fast loading
   - Arrow: Fastest for random access
4. **Multiple Datasets**: Can combine multiple local datasets in training

## Troubleshooting

### Dataset Not Found
```bash
# Check dataset_dir path
ls -la ./data/

# Verify dataset_info.json entry
cat ./data/dataset_info.json | grep your_dataset_name
```

### Wrong Column Names
```python
# Check dataset columns
from datasets import load_dataset
ds = load_dataset("json", data_files="./data/your_dataset.jsonl")
print(ds["train"].column_names)
```

### Memory Issues with Large Local Files
```bash
# Use streaming mode
python pretokenize_dataset.py \
  --dataset ./data/large_dataset.jsonl \
  --streaming \
  --model meta-llama/Llama-2-7b-hf \
  --output ./tokenized_data/large
```

## Advanced: Custom Dataset Formats

If you have a custom format, create a converter:

```python
# convert_custom_to_jsonl.py
import json

def convert_custom_format(input_file, output_file):
    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
        # Your custom parsing logic
        for line in inf:
            data = your_custom_parser(line)
            json_line = {"text": data["content"]}
            outf.write(json.dumps(json_line) + '\n')

convert_custom_format("custom_data.txt", "data/custom.jsonl")
```

Then use the JSONL file with LlamaFactory!