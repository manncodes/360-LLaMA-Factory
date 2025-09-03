#!/usr/bin/env python3
"""
Script to download datasets locally and prepare them for LlamaFactory training.
Supports HuggingFace datasets and converts them to local formats.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset
import pandas as pd

def download_and_save_dataset(
    dataset_name: str,
    output_dir: str,
    output_format: str = "json",
    split: str = "train",
    max_samples: Optional[int] = None,
    text_column: str = "text"
):
    """Download dataset and save locally in specified format."""
    
    print(f"Downloading {dataset_name}...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if max_samples:
        dataset = load_dataset(
            dataset_name, 
            split=f"{split}[:{max_samples}]",
            trust_remote_code=True
        )
    else:
        dataset = load_dataset(
            dataset_name, 
            split=split,
            trust_remote_code=True
        )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Determine output file
    dataset_basename = dataset_name.replace("/", "_")
    
    if output_format == "json":
        output_file = output_path / f"{dataset_basename}.json"
        dataset.to_json(output_file)
    elif output_format == "jsonl":
        output_file = output_path / f"{dataset_basename}.jsonl"
        dataset.to_json(output_file, lines=True)
    elif output_format == "parquet":
        output_file = output_path / f"{dataset_basename}.parquet"
        dataset.to_parquet(output_file)
    elif output_format == "arrow":
        output_file = output_path / f"{dataset_basename}"
        dataset.save_to_disk(output_file)
    elif output_format == "csv":
        output_file = output_path / f"{dataset_basename}.csv"
        dataset.to_csv(output_file)
    else:
        raise ValueError(f"Unsupported format: {output_format}")
    
    print(f"✓ Dataset saved to: {output_file}")
    
    # Create dataset_info entry
    dataset_info_entry = {
        dataset_basename: {
            "file_name": output_file.name if output_format != "arrow" else str(output_file.relative_to(output_path.parent)),
            "columns": {
                "prompt": text_column
            }
        }
    }
    
    # Save dataset_info snippet
    info_file = output_path / f"{dataset_basename}_info.json"
    with open(info_file, "w") as f:
        json.dump(dataset_info_entry, f, indent=2)
    
    print(f"\nAdd this to your data/dataset_info.json:")
    print(json.dumps(dataset_info_entry, indent=2))
    
    # Create example YAML config
    yaml_content = f"""# Training config for local {dataset_basename} dataset
model_name_or_path: meta-llama/Llama-2-7b-hf
dataset: {dataset_basename}
dataset_dir: {output_path.parent}
stage: pt
cutoff_len: 131072
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-5
num_train_epochs: 1
output_dir: ./outputs/{dataset_basename}
"""
    
    yaml_file = output_path / f"train_{dataset_basename}.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    
    print(f"\n✓ Example training config saved to: {yaml_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets locally for LlamaFactory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download NExtLong dataset to local directory
  python prepare_local_dataset.py \\
    --dataset caskcsg/NExtLong-512K-dataset \\
    --output ./data/nextlong_local \\
    --format jsonl
    
  # Download with sample limit
  python prepare_local_dataset.py \\
    --dataset caskcsg/NExtLong-512K-dataset \\
    --output ./data/nextlong_local \\
    --format json \\
    --max-samples 100
    
  # Download and save as Arrow format (most efficient)
  python prepare_local_dataset.py \\
    --dataset caskcsg/NExtLong-512K-dataset \\
    --output ./data/datasets \\
    --format arrow
"""
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory for local dataset"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl", "parquet", "arrow", "csv"],
        default="jsonl",
        help="Output format (jsonl recommended for large datasets)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the dataset"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("DATASET LOCAL PREPARATION")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    if args.max_samples:
        print(f"Samples: {args.max_samples}")
    
    try:
        output_file = download_and_save_dataset(
            dataset_name=args.dataset,
            output_dir=args.output,
            output_format=args.format,
            split=args.split,
            max_samples=args.max_samples,
            text_column=args.text_column
        )
        
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("1. Add the dataset info entry to data/dataset_info.json")
        print("2. Use the dataset name in your training YAML")
        print(f"3. Set dataset_dir to: {Path(args.output).parent}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()