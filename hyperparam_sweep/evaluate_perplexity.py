#!/usr/bin/env python3
"""Evaluate perplexity and LongPPL for RoPE scaling configurations."""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


@dataclass
class EvalMetrics:
    """Store evaluation metrics."""
    perplexity: float
    long_ppl: float
    loss: float
    tokens_processed: int
    context_length: int
    memory_gb: float
    throughput_tps: float  # tokens per second
    


def load_model_and_tokenizer(config: Dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with RoPE configuration."""
    model_name = config["model_name_or_path"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config and apply RoPE settings
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", False)
    )
    
    # Apply RoPE scaling if specified
    if "rope_scaling_type" in config:
        rope_config = {
            "type": config["rope_scaling_type"],
            "factor": config.get("rope_scaling_factor", 2.0)
        }
        
        # Add type-specific parameters
        if config["rope_scaling_type"] == "yarn":
            rope_config["original_max_position_embeddings"] = model_config.max_position_embeddings
            rope_config["yarn_alpha"] = config.get("yarn_alpha", 1.0)
            rope_config["yarn_beta"] = config.get("yarn_beta", 32.0)
        elif config["rope_scaling_type"] == "longrope":
            rope_config["short_factor"] = config.get("longrope_short_factor", 1.0)
            rope_config["long_factor"] = config.get("longrope_long_factor", 1.0)
        
        model_config.rope_scaling = rope_config
        print(f"Applied RoPE scaling: {rope_config}")
    
    # Update max position embeddings based on context length
    if "cutoff_len" in config:
        model_config.max_position_embeddings = config["cutoff_len"]
    
    # Load model
    dtype = getattr(torch, config.get("infer_dtype", "float32").replace("bfloat16", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=dtype,
        trust_remote_code=config.get("trust_remote_code", False),
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=config.get("low_cpu_mem_usage", True)
    )
    
    model.eval()
    return model, tokenizer


def load_eval_dataset(dataset_name: str, tokenizer, max_samples: int = 1000) -> DataLoader:
    """Load evaluation dataset."""
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
        dataset = dataset.take(max_samples)
    elif dataset_name == "pg19":
        dataset = load_dataset("emozilla/pg19", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"] if "text" in examples else examples["content"],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=False
    )
    
    return dataloader


def compute_perplexity(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    context_length: int,
    device: str = "cuda"
) -> Tuple[float, float, int]:
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Truncate to context length
            if input_ids.shape[1] > context_length:
                input_ids = input_ids[:, :context_length]
                attention_mask = attention_mask[:, :context_length]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            
            # Count actual tokens (excluding padding)
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Early stopping for quick evaluation
            if total_tokens > 100000:  # Process at least 100k tokens
                break
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss, total_tokens


def compute_long_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    context_length: int,
    stride: int = 512,
    device: str = "cuda"
) -> float:
    """Compute perplexity on long sequences using sliding window."""
    model.eval()
    
    # Tokenize entire text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    seq_len = input_ids.size(1)
    
    # If sequence is shorter than context, compute regular perplexity
    if seq_len <= context_length:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            return np.exp(outputs.loss.item())
    
    # Sliding window evaluation
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + context_length, seq_len)
        trg_len = end_loc - prev_end_loc  # How many tokens to evaluate
        
        input_ids_window = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_window.clone()
        target_ids[:, :-trg_len] = -100  # Only compute loss on new tokens
        
        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    # Compute average perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (end_loc - 0))
    return ppl.item()


def evaluate_config(config_path: str, dataset: str = "wikitext") -> Dict:
    """Evaluate a single configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nEvaluating: {config_path}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Track start time
    start_time = datetime.now()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    device = next(model.parameters()).device
    
    # Get context length
    context_length = config.get("cutoff_len", 2048)
    
    # Load dataset
    dataloader = load_eval_dataset(
        dataset, 
        tokenizer, 
        max_samples=config.get("max_samples", 1000)
    )
    
    # Compute standard perplexity
    ppl, loss, tokens = compute_perplexity(model, dataloader, context_length, device)
    
    # Compute long-context perplexity (on a sample long text)
    long_text = " ".join(["This is a test sentence." * 100] * 50)  # ~25k tokens
    long_ppl = compute_long_ppl(model, tokenizer, long_text, context_length, device=device)
    
    # Track memory usage
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory_gb = 0
    
    # Calculate throughput
    elapsed_time = (datetime.now() - start_time).total_seconds()
    throughput = tokens / elapsed_time
    
    # Create results
    results = {
        "config_path": config_path,
        "config": config,
        "metrics": {
            "perplexity": ppl,
            "long_ppl": long_ppl,
            "loss": loss,
            "tokens_processed": tokens,
            "context_length": context_length,
            "memory_gb": memory_gb,
            "throughput_tps": throughput,
        },
        "runtime": {
            "elapsed_seconds": elapsed_time,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Save results
    result_dir = Path(config.get("save_dir", "results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = result_dir / f"eval_{Path(config_path).stem}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Long PPL: {long_ppl:.2f}")
    print(f"  Memory: {memory_gb:.2f} GB")
    print(f"  Throughput: {throughput:.1f} tokens/sec")
    print(f"  Saved to: {result_file}")
    
    return results


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RoPE configurations")
    parser.add_argument("config", help="Path to config file or directory")
    parser.add_argument("--dataset", default="wikitext", 
                       choices=["wikitext", "c4", "pg19"],
                       help="Dataset for evaluation")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if config_path.is_file():
        # Evaluate single configuration
        results = evaluate_config(str(config_path), args.dataset)
    elif config_path.is_dir():
        # Evaluate all configs in directory
        config_files = list(config_path.glob("*.yaml"))
        
        # Skip manifest file
        config_files = [f for f in config_files if f.name != "manifest.yaml"]
        
        print(f"Found {len(config_files)} configurations")
        
        all_results = []
        for config_file in config_files:
            try:
                results = evaluate_config(str(config_file), args.dataset)
                all_results.append(results)
            except Exception as e:
                print(f"Error evaluating {config_file}: {e}")
                continue
        
        # Save aggregated results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
    else:
        print(f"Error: {config_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()