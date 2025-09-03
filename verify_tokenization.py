#!/usr/bin/env python3
"""
Verify that our pretokenized data matches what LlamaFactory generates during training.
"""

import json
import sys
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

# Import LlamaFactory components
sys.path.insert(0, ".")
from llamafactory.data.processors.pretrain import preprocess_pretrain_dataset
from llamafactory.hparams import DataArguments

def load_pretokenized(path: str):
    """Load our pretokenized dataset"""
    dataset = load_from_disk(path)
    if "train" in dataset:
        return dataset["train"]
    return dataset

def tokenize_with_llamafactory(json_file: str, model_name: str, cutoff_len: int, max_samples: int = 5):
    """Tokenize using LlamaFactory's method"""
    
    # Load data
    with open(json_file, "r") as f:
        data = json.load(f)[:max_samples]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data args
    data_args = DataArguments(
        cutoff_len=cutoff_len,
        template="default",
        packing=False
    )
    
    # Format as LlamaFactory expects for pretrain
    examples = {
        "_prompt": [[{"content": item["text"] if isinstance(item, dict) else str(item)}] for item in data]
    }
    
    # Tokenize
    result = preprocess_pretrain_dataset(examples, tokenizer, data_args)
    
    return result

def compare_datasets(our_data, llama_data):
    """Compare two tokenized datasets"""
    print("\n" + "="*60)
    print("TOKENIZATION VERIFICATION")
    print("="*60)
    
    # Check number of samples
    our_samples = len(our_data)
    llama_samples = len(llama_data["input_ids"])
    
    print(f"\nSample counts:")
    print(f"  Our pretokenized: {our_samples}")
    print(f"  LlamaFactory: {llama_samples}")
    
    if our_samples != llama_samples:
        print(f"❌ Sample count mismatch!")
        return False
    
    # Compare each sample
    all_match = True
    for i in range(min(our_samples, llama_samples)):
        our_tokens = our_data[i]["input_ids"]
        llama_tokens = llama_data["input_ids"][i]
        
        if our_tokens == llama_tokens:
            print(f"✅ Sample {i}: IDENTICAL ({len(our_tokens)} tokens)")
        else:
            print(f"❌ Sample {i}: MISMATCH")
            print(f"   Our length: {len(our_tokens)}")
            print(f"   LlamaFactory length: {len(llama_tokens)}")
            
            # Find first difference
            min_len = min(len(our_tokens), len(llama_tokens))
            for j in range(min_len):
                if our_tokens[j] != llama_tokens[j]:
                    print(f"   First difference at position {j}:")
                    print(f"     Our token: {our_tokens[j]}")
                    print(f"     LlamaFactory token: {llama_tokens[j]}")
                    break
            
            # Show sample of tokens
            print(f"   Our first 10: {our_tokens[:10]}")
            print(f"   LlamaFactory first 10: {llama_tokens[:10]}")
            
            all_match = False
    
    return all_match

def main():
    print("🔍 Verifying pretokenization consistency with LlamaFactory\n")
    
    # Configuration
    pretokenized_path = "./test_pretok"
    source_data = "./data/c4_demo.json"
    model = "gpt2"
    cutoff = 2048
    max_samples = 5
    
    print(f"Configuration:")
    print(f"  Pretokenized path: {pretokenized_path}")
    print(f"  Source data: {source_data}")
    print(f"  Model: {model}")
    print(f"  Cutoff: {cutoff}")
    print(f"  Samples: {max_samples}")
    
    # Load our pretokenized data
    print(f"\nLoading pretokenized data from {pretokenized_path}...")
    our_data = load_pretokenized(pretokenized_path)
    print(f"✓ Loaded {len(our_data)} samples")
    
    # Tokenize with LlamaFactory
    print(f"\nTokenizing with LlamaFactory method...")
    llama_data = tokenize_with_llamafactory(source_data, model, cutoff, max_samples)
    print(f"✓ Tokenized {len(llama_data['input_ids'])} samples")
    
    # Compare
    match = compare_datasets(our_data, llama_data)
    
    print("\n" + "="*60)
    if match:
        print("✅ SUCCESS: Pretokenization is IDENTICAL to LlamaFactory!")
        print("Your pretokenized data is ready for training.")
    else:
        print("⚠️  WARNING: Differences found in tokenization")
        print("Please review the differences above.")
    print("="*60)
    
    return 0 if match else 1

if __name__ == "__main__":
    sys.exit(main())