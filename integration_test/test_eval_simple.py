#!/usr/bin/env python3
"""Simplified evaluation for integration testing."""

import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

def simple_evaluation(config_path: str):
    """Run a very simple evaluation test."""
    print(f"🧪 Running integration test: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    start_time = datetime.now()
    
    try:
        # Load model and tokenizer
        model_name = config["model_name_or_path"]
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        
        # Get device
        device = next(model.parameters()).device
        
        print(f"✅ Model loaded successfully on {device}")
        
        # Test with a simple text
        test_text = "The quick brown fox jumps over the lazy dog. " * 10
        
        # Tokenize
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            max_length=config.get("cutoff_len", 512),
            truncation=True
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            perplexity = np.exp(loss)
        
        # Mock additional metrics for demonstration
        long_ppl = perplexity * 1.2  # Simulate slightly worse long context
        
        # Track memory
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            memory_gb = 0.1  # Mock CPU memory
        
        # Calculate runtime
        elapsed = (datetime.now() - start_time).total_seconds()
        throughput = inputs["input_ids"].numel() / elapsed
        
        # Create results
        results = {
            "config_path": config_path,
            "config": config,
            "metrics": {
                "perplexity": perplexity,
                "long_ppl": long_ppl,
                "loss": loss,
                "tokens_processed": inputs["input_ids"].numel(),
                "context_length": config.get("cutoff_len", 512),
                "memory_gb": memory_gb,
                "throughput_tps": throughput,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            },
            "status": "success"
        }
        
        # Save results
        save_dir = Path(config.get("save_dir", "integration_test/results"))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = save_dir / f"eval_{Path(config_path).stem}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Evaluation completed successfully!")
        print(f"   Perplexity: {perplexity:.2f}")
        print(f"   LongPPL: {long_ppl:.2f}")
        print(f"   Memory: {memory_gb:.2f} GB")
        print(f"   Runtime: {elapsed:.1f}s")
        print(f"   Results saved: {result_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {
            "config_path": config_path,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_eval_simple.py <config.yaml>")
        sys.exit(1)
    
    result = simple_evaluation(sys.argv[1])
    if result["status"] == "success":
        print("🎉 Integration test PASSED")
    else:
        print("💥 Integration test FAILED")
        sys.exit(1)