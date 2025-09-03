#!/usr/bin/env python3
"""
Pretokenize long-context datasets for efficient training.
This script tokenizes datasets offline following LlamaFactory's preprocessing rules.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from itertools import chain

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message: str, color: str = Colors.OKGREEN):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], 
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int,
    packing: bool = False,
    template: str = "default",
    add_eos: bool = True
) -> Dict[str, List[Any]]:
    """
    Preprocess pretrain dataset following LlamaFactory's rules.
    Adapted from src/llamafactory/data/processors/pretrain.py
    """
    # Determine EOS token based on template
    if template == "llama3":
        eos_token = "<|end_of_text|>"
    else:
        eos_token = tokenizer.eos_token if add_eos else ""
    
    # Extract text content
    if "text" in examples:
        text_examples = [text + eos_token for text in examples["text"]]
    elif "_prompt" in examples:
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]
    else:
        # Try to find text column
        text_col = None
        for col in examples.keys():
            if isinstance(examples[col][0], str):
                text_col = col
                break
        if text_col:
            text_examples = [text + eos_token for text in examples[text_col]]
        else:
            raise ValueError("Could not find text column in dataset")
    
    if not packing:
        # Simple truncation
        if template == "gemma":
            text_examples = [tokenizer.bos_token + example for example in text_examples]
        
        result = tokenizer(
            text_examples,
            add_special_tokens=False,
            truncation=True,
            max_length=cutoff_len,
            padding=False
        )
    else:
        # Packing: concatenate and split into blocks
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {
            k: list(chain(*tokenized_examples[k])) 
            for k in tokenized_examples.keys()
        }
        
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        total_length = (total_length // cutoff_len) * cutoff_len
        
        result = {
            k: [t[i : i + cutoff_len] for i in range(0, total_length, cutoff_len)]
            for k, t in concatenated_examples.items()
        }
        
        if template == "gemma":
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id
    
    return result

def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int,
    max_samples: Optional[int] = None,
    packing: bool = False,
    template: str = "default",
    num_proc: int = 4,
    batch_size: int = 1000,
    split: str = "train",
    streaming: bool = False
) -> Dataset:
    """Load and tokenize a dataset"""
    
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"Loading dataset: {dataset_name}", Colors.OKBLUE)
    print_colored(f"{'='*60}", Colors.HEADER)
    
    # Load dataset
    dataset_path = Path(dataset_name)
    
    # Check if it's a local path
    if dataset_path.exists():
        print(f"Loading from local path: {dataset_path}")
        
        if dataset_path.is_dir():
            # Check if it's an Arrow dataset directory
            if (dataset_path / "dataset_info.json").exists():
                # Arrow format dataset
                from datasets import load_from_disk
                dataset = load_from_disk(str(dataset_path))
                if split in dataset:
                    dataset = dataset[split]
            else:
                # Directory with data files
                data_files = []
                for ext in [".json", ".jsonl", ".parquet", ".csv", ".txt"]:
                    data_files.extend(dataset_path.glob(f"*{ext}"))
                
                if not data_files:
                    raise ValueError(f"No data files found in {dataset_path}")
                
                # Determine file type from first file
                first_file = str(data_files[0])
                file_ext = data_files[0].suffix[1:]  # Remove the dot
                
                file_type_map = {
                    "json": "json",
                    "jsonl": "json",
                    "parquet": "parquet",
                    "csv": "csv",
                    "txt": "text"
                }
                
                dataset = load_dataset(
                    file_type_map.get(file_ext, "text"),
                    data_files=[str(f) for f in data_files],
                    split=split
                )
        else:
            # Single file
            if dataset_path.suffix == ".json":
                dataset = Dataset.from_json(str(dataset_path))
            elif dataset_path.suffix == ".jsonl":
                dataset = Dataset.from_json(str(dataset_path), lines=True)
            elif dataset_path.suffix == ".parquet":
                dataset = Dataset.from_parquet(str(dataset_path))
            elif dataset_path.suffix == ".csv":
                dataset = Dataset.from_csv(str(dataset_path))
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
        if max_samples and not streaming:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    elif "/" in dataset_name:
        # HuggingFace dataset
        print(f"Loading from HuggingFace: {dataset_name}")
        if max_samples:
            dataset = load_dataset(
                dataset_name,
                split=f"{split}[:{max_samples}]",
                streaming=streaming,
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                trust_remote_code=True
            )
    else:
        raise ValueError(f"Dataset not found: {dataset_name}")
    
    print(f"Dataset loaded: {len(dataset) if not streaming else 'streaming'} samples")
    
    # Tokenize dataset
    print_colored(f"\nTokenizing with cutoff_len={cutoff_len}...", Colors.OKCYAN)
    
    def tokenize_function(examples):
        return preprocess_pretrain_dataset(
            examples,
            tokenizer,
            cutoff_len=cutoff_len,
            packing=packing,
            template=template
        )
    
    start_time = time.time()
    
    if streaming:
        # For streaming datasets, we need to iterate
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names
        )
    else:
        # Regular dataset with progress bar
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc if not packing else 1,  # Packing needs sequential processing
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            load_from_cache_file=False
        )
    
    elapsed = time.time() - start_time
    
    if not streaming:
        print_colored(f"✓ Tokenization complete: {len(tokenized_dataset)} samples in {elapsed:.2f}s", Colors.OKGREEN)
        
        # Print statistics
        if len(tokenized_dataset) > 0:
            sample_lengths = [len(ids) for ids in tokenized_dataset["input_ids"][:100]]
            avg_length = sum(sample_lengths) / len(sample_lengths)
            print(f"  Average sequence length (first 100): {avg_length:.0f} tokens")
            print(f"  Min length: {min(sample_lengths)} tokens")
            print(f"  Max length: {max(sample_lengths)} tokens")
    else:
        print_colored(f"✓ Tokenization pipeline created for streaming dataset", Colors.OKGREEN)
    
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize datasets for LlamaFactory training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize NExtLong dataset with 512K context
  python pretokenize_dataset.py \\
    --dataset caskcsg/NExtLong-512K-dataset \\
    --model meta-llama/Llama-2-7b-hf \\
    --cutoff 524288 \\
    --output ./tokenized_data/nextlong_512k
    
  # Tokenize with packing for efficient training
  python pretokenize_dataset.py \\
    --dataset c4 \\
    --model meta-llama/Llama-2-7b-hf \\
    --cutoff 8192 \\
    --packing \\
    --output ./tokenized_data/c4_packed_8k
    
  # Tokenize local JSON file
  python pretokenize_dataset.py \\
    --dataset ./data/my_dataset.json \\
    --model gpt2 \\
    --cutoff 2048 \\
    --output ./tokenized_data/my_dataset
"""
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset name (HuggingFace) or path to local file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model name or path for tokenizer"
    )
    parser.add_argument(
        "--cutoff", 
        type=int, 
        default=8192,
        help="Maximum sequence length (default: 8192)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output directory for tokenized dataset"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--packing", 
        action="store_true",
        help="Enable packing for efficient training"
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="default",
        choices=["default", "llama3", "gemma"],
        help="Chat template to use"
    )
    parser.add_argument(
        "--num-proc", 
        type=int, 
        default=4,
        help="Number of processes for tokenization"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1000,
        help="Batch size for tokenization"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--val-size", 
        type=float, 
        default=0.0,
        help="Validation set size (0.0-1.0)"
    )
    parser.add_argument(
        "--streaming", 
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    parser.add_argument(
        "--no-eos", 
        action="store_true",
        help="Don't add EOS token"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print_colored(f"\n{'='*60}", Colors.BOLD)
    print_colored("DATASET PRETOKENIZATION", Colors.BOLD)
    print_colored(f"{'='*60}", Colors.BOLD)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Cutoff length: {args.cutoff:,} tokens")
    print(f"Output directory: {args.output}")
    print(f"Packing: {'Enabled' if args.packing else 'Disabled'}")
    print(f"Template: {args.template}")
    
    # Load tokenizer
    print_colored("\nLoading tokenizer...", Colors.OKCYAN)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        cutoff_len=args.cutoff,
        max_samples=args.max_samples,
        packing=args.packing,
        template=args.template,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        split=args.split,
        streaming=args.streaming
    )
    
    # Handle validation split
    if args.val_size > 0 and not args.streaming:
        print_colored(f"\nCreating validation split ({args.val_size*100:.0f}%)...", Colors.OKCYAN)
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(
            test_size=args.val_size,
            seed=42
        )
        
        dataset_dict = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
        
        print(f"✓ Train samples: {len(dataset_dict['train'])}")
        print(f"✓ Validation samples: {len(dataset_dict['validation'])}")
    else:
        dataset_dict = DatasetDict({"train": tokenized_dataset})
    
    # Save dataset
    print_colored(f"\nSaving tokenized dataset to {args.output}...", Colors.OKCYAN)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not args.streaming:
        dataset_dict.save_to_disk(str(output_path))
        
        # Save metadata
        metadata = {
            "dataset": args.dataset,
            "model": args.model,
            "cutoff_len": args.cutoff,
            "packing": args.packing,
            "template": args.template,
            "timestamp": datetime.now().isoformat(),
            "num_samples": {
                split: len(dataset) 
                for split, dataset in dataset_dict.items()
            }
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print_colored(f"✓ Dataset saved successfully!", Colors.OKGREEN)
        
        # Print usage instructions
        print_colored(f"\n{'='*60}", Colors.HEADER)
        print_colored("HOW TO USE THIS TOKENIZED DATASET", Colors.HEADER)
        print_colored(f"{'='*60}", Colors.HEADER)
        
        print("\nAdd this to your training YAML configuration:")
        print(f"\n{Colors.OKCYAN}tokenized_path: {output_path}{Colors.ENDC}")
        print(f"\n{Colors.WARNING}Note: When using tokenized_path, the dataset will be loaded")
        print(f"directly without preprocessing, significantly speeding up training!{Colors.ENDC}")
    else:
        print_colored("Streaming mode: Dataset pipeline created but not saved", Colors.WARNING)
        print("For streaming datasets, use the dataset directly in training")

if __name__ == "__main__":
    main()