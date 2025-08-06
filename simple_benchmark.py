#!/usr/bin/env python3
"""
Simple benchmark for testing existing long context methods
"""

import json
import time
import os
from pathlib import Path
import subprocess
from datetime import datetime

# Model to use (TinyLlama for testing, change to your model)
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = Path(f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test configurations
CONFIGS = [
    {
        "name": "baseline",
        "rope_scaling": "none",
        "context_length": 2048,
        "flash_attn": "auto",
        "shift_attn": False
    },
    {
        "name": "linear_4k",
        "rope_scaling": "linear", 
        "context_length": 4096,
        "flash_attn": "auto",
        "shift_attn": False
    },
    {
        "name": "linear_8k",
        "rope_scaling": "linear",
        "context_length": 8192,
        "flash_attn": "auto",
        "shift_attn": False
    },
    {
        "name": "dynamic_4k",
        "rope_scaling": "dynamic",
        "context_length": 4096,
        "flash_attn": "auto",
        "shift_attn": False
    },
    {
        "name": "dynamic_8k",
        "rope_scaling": "dynamic",
        "context_length": 8192,
        "flash_attn": "auto",
        "shift_attn": False
    },
]

def create_test_prompt(length):
    """Create a test prompt of approximately the desired token length."""
    # Rough estimate: 1 token ≈ 4 characters
    words = ["word"] * (length // 4)
    prompt = " ".join(words[:length//4])
    return f"Summarize the following text in one sentence: {prompt}"

def run_single_test(config):
    """Run a single benchmark test."""
    print(f"\nTesting: {config['name']}")
    print("-" * 40)
    
    # Create config file
    config_file = OUTPUT_DIR / f"config_{config['name']}.yaml"
    with open(config_file, 'w') as f:
        f.write(f"""model_name_or_path: {MODEL}
template: tinyllama
rope_scaling: {config['rope_scaling']}
model_max_length: {config['context_length']}
flash_attn: {config['flash_attn']}
shift_attn: {config['shift_attn']}
do_sample: false
temperature: 0.1
max_new_tokens: 50
""")
    
    # Create test prompt
    prompt = create_test_prompt(config['context_length'] // 2)
    prompt_file = OUTPUT_DIR / f"prompt_{config['name']}.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    
    # Run inference
    start_time = time.time()
    
    cmd = [
        "python3", "-m", "llamafactory.cli", "chat",
        "--config", str(config_file),
        "--no-stream"
    ]
    
    try:
        with open(prompt_file, 'r') as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
                timeout=60
            )
        
        elapsed_time = time.time() - start_time
        
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        
        # Save result
        result_data = {
            "config": config,
            "elapsed_time": elapsed_time,
            "success": success,
            "output": output[:500]  # First 500 chars
        }
        
        # Print summary
        status = "✓" if success else "✗"
        print(f"  Status: {status}")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Context: {config['context_length']} tokens")
        print(f"  RoPE: {config['rope_scaling']}")
        
        return result_data
        
    except subprocess.TimeoutExpired:
        print(f"  Status: ✗ (timeout)")
        return {
            "config": config,
            "elapsed_time": 60.0,
            "success": False,
            "output": "Timeout"
        }
    except Exception as e:
        print(f"  Status: ✗ ({str(e)})")
        return {
            "config": config,
            "elapsed_time": 0,
            "success": False,
            "output": str(e)
        }

def main():
    print("=" * 60)
    print("Simple Long Context Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Configs to test: {len(CONFIGS)}")
    
    results = []
    
    for config in CONFIGS:
        result = run_single_test(config)
        results.append(result)
        
        # Save intermediate results
        with open(OUTPUT_DIR / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for result in results:
        if result['success']:
            print(f"{result['config']['name']}: {result['elapsed_time']:.2f}s")
        else:
            print(f"{result['config']['name']}: FAILED")
    
    print(f"\nResults saved to: {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()