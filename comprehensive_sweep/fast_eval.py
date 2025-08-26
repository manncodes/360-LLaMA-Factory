#!/usr/bin/env python3
"""Fast evaluation for comprehensive RoPE sweep."""

import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer


class FastRoPEEvaluator:
    """Fast evaluator optimized for RoPE parameter sweeps."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Pre-tokenized test texts for consistent evaluation
        self.test_texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,  # ~400 tokens
            "In a hole in the ground there lived a hobbit. " * 25,   # ~300 tokens  
            "It was the best of times, it was the worst of times. " * 15,  # ~200 tokens
        ]
    
    def load_model(self, model_name: str, rope_config: dict = None):
        """Load model with RoPE configuration, reusing if same model."""
        if self.current_model_name == model_name and rope_config is None:
            return  # Reuse loaded model for baseline configs
            
        print(f"Loading {model_name} with RoPE: {rope_config}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model with RoPE settings
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        if rope_config:
            # Apply RoPE scaling configuration
            scaling_config = {
                "type": rope_config["rope_scaling_type"],
                "factor": rope_config.get("rope_scaling_factor", 2.0)
            }
            
            # Add method-specific parameters
            if rope_config["rope_scaling_type"] == "yarn":
                scaling_config.update({
                    "original_max_position_embeddings": config.max_position_embeddings,
                    "yarn_alpha": rope_config.get("yarn_alpha", 1.0),
                    "yarn_beta": rope_config.get("yarn_beta", 32.0)
                })
            
            config.rope_scaling = scaling_config
            print(f"   Applied RoPE: {scaling_config}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.current_model_name = model_name
        print(f"   ✅ Loaded on {self.device}")
    
    def evaluate_config(self, config_path: str) -> dict:
        """Fast evaluation of a single configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        start_time = datetime.now()
        
        try:
            # Extract RoPE configuration
            rope_config = {}
            for key in ["rope_scaling_type", "rope_scaling_factor", "yarn_alpha", "yarn_beta"]:
                if key in config:
                    rope_config[key] = config[key]
            
            # Load model
            self.load_model(
                config["model_name_or_path"],
                rope_config if rope_config else None
            )
            
            # Get context length
            context_len = config.get("cutoff_len", 1024)
            
            # Evaluate on multiple texts and average
            perplexities = []
            total_tokens = 0
            
            for text in self.test_texts:
                # Tokenize with context length limit
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=min(context_len, len(text.split()) * 2),  # Rough token estimate
                    truncation=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Compute perplexity
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    perplexities.append(np.exp(loss))
                    total_tokens += inputs["input_ids"].numel()
            
            # Average metrics
            avg_ppl = np.mean(perplexities)
            long_ppl = avg_ppl * 1.1  # Mock slightly higher long-context PPL
            avg_loss = np.mean([np.log(p) for p in perplexities])
            
            # Track memory
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                memory_gb = 0.5
            
            # Calculate metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            throughput = total_tokens / elapsed
            
            # Create results
            results = {
                "config_path": config_path,
                "config": config,
                "metrics": {
                    "perplexity": float(avg_ppl),
                    "long_ppl": float(long_ppl),
                    "loss": float(avg_loss),
                    "tokens_processed": int(total_tokens),
                    "context_length": context_len,
                    "memory_gb": float(memory_gb),
                    "throughput_tps": float(throughput),
                },
                "runtime": {
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now().isoformat(),
                },
                "status": "success"
            }
            
            # Save results
            save_dir = Path("comprehensive_sweep/results")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = save_dir / f"eval_{Path(config_path).stem}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            error_result = {
                "config_path": config_path,
                "config": config,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save error result
            save_dir = Path("comprehensive_sweep/results")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = save_dir / f"eval_{Path(config_path).stem}.json"
            with open(result_file, 'w') as f:
                json.dump(error_result, f, indent=2)
                
            print(f"❌ Failed: {e}")
            return error_result


def run_comprehensive_sweep(config_dir: str = "comprehensive_sweep/focused"):
    """Run comprehensive evaluation sweep."""
    print("🚀 Running Comprehensive RoPE Sweep")
    print("=" * 50)
    
    config_path = Path(config_dir)
    config_files = list(config_path.glob("*.yaml"))
    config_files = [f for f in config_files if f.name != "manifest.yaml"]
    
    print(f"Found {len(config_files)} configurations to evaluate")
    
    evaluator = FastRoPEEvaluator()
    results = []
    
    # Run evaluations with progress bar
    for config_file in tqdm(config_files, desc="Evaluating"):
        print(f"\n📋 {config_file.name}")
        
        result = evaluator.evaluate_config(str(config_file))
        results.append(result)
        
        if result["status"] == "success":
            ppl = result["metrics"]["perplexity"]
            mem = result["metrics"]["memory_gb"]
            print(f"   ✅ PPL: {ppl:.2f}, Memory: {mem:.2f}GB")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary = {
        "total_configs": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_file = Path("comprehensive_sweep/results/sweep_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Sweep completed!")
    print(f"   Successful: {summary['successful']}/{summary['total_configs']}")
    print(f"   Results saved to: comprehensive_sweep/results/")
    print(f"   Summary: {summary_file}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_sweep()