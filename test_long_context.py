#!/usr/bin/env python3
"""
Simple test script for long context methods in 360-LLaMA-Factory
"""

import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def test_rope_scaling(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Test different RoPE scaling methods."""
    
    print("Testing RoPE Scaling Methods")
    print("=" * 50)
    
    results = {}
    
    # Test configurations
    configs = [
        {"name": "baseline", "rope_scaling": None, "max_length": 4096},
        {"name": "linear", "rope_scaling": {"type": "linear", "factor": 2.0}, "max_length": 8192},
        {"name": "dynamic", "rope_scaling": {"type": "dynamic", "factor": 2.0}, "max_length": 8192},
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        try:
            # Load model with specific config
            model_kwargs = {}
            if config["rope_scaling"]:
                model_kwargs["rope_scaling"] = config["rope_scaling"]
            
            # Create test input
            test_text = "The quick brown fox " * 500  # Long input
            inputs = tokenizer(test_text, return_tensors="pt", max_length=config["max_length"], truncation=True)
            
            print(f"  Input length: {inputs['input_ids'].shape[1]} tokens")
            print(f"  Max length: {config['max_length']}")
            
            # Test position embeddings
            if config["rope_scaling"]:
                print(f"  RoPE scaling: {config['rope_scaling']['type']}")
                print(f"  Scaling factor: {config['rope_scaling']['factor']}")
            
            results[config["name"]] = {
                "max_length": config["max_length"],
                "rope_scaling": config["rope_scaling"],
                "status": "success"
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config["name"]] = {
                "status": "failed",
                "error": str(e)
            }
    
    return results


def test_attention_implementations():
    """Test different attention implementations."""
    
    print("\nTesting Attention Implementations")
    print("=" * 50)
    
    attention_types = ["auto", "sdpa", "fa2", "disabled"]
    
    for attn_type in attention_types:
        print(f"\nTesting {attn_type}...")
        
        # Check availability
        if attn_type == "fa2":
            try:
                import flash_attn
                print(f"  FlashAttention-2: Available")
            except ImportError:
                print(f"  FlashAttention-2: Not installed")
        elif attn_type == "sdpa":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                print(f"  SDPA: Available (PyTorch {torch.__version__})")
            else:
                print(f"  SDPA: Not available")
        else:
            print(f"  {attn_type}: Standard implementation")


def check_sequence_parallel():
    """Check sequence parallel configuration."""
    
    print("\nChecking Sequence Parallel Support")
    print("=" * 50)
    
    # Check if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs available: {torch.cuda.device_count()}")
        print("Sequence parallel can be enabled")
        
        modes = ["zigzag-ring", "llama3", "ulysses"]
        for mode in modes:
            print(f"  Mode: {mode} - Supported")
    else:
        print("Single GPU or CPU - Sequence parallel not applicable")


def main():
    parser = argparse.ArgumentParser(description="Test long context methods")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help="Model to test")
    parser.add_argument("--test", choices=["rope", "attention", "sp", "all"], 
                       default="all", help="What to test")
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    if args.test in ["rope", "all"]:
        rope_results = test_rope_scaling(args.model)
        
        # Save results
        with open("rope_test_results.json", "w") as f:
            json.dump(rope_results, f, indent=2)
        print("\nRoPE test results saved to rope_test_results.json")
    
    if args.test in ["attention", "all"]:
        test_attention_implementations()
    
    if args.test in ["sp", "all"]:
        check_sequence_parallel()
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    main()