#!/usr/bin/env python3

import sys
sys.path.append('scripts/methods')
from benchmark_runner import MethodBenchmarkRunner

def test_baseline_only():
    print("Testing baseline method only...")
    
    runner = MethodBenchmarkRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "test_baseline", 0.0)
    
    # Test only baseline
    method = "baseline"
    config_data = runner.methods[method]
    
    # Create config with minimal contexts - match your working example
    contexts = [2048, 4096]  # Use same as original working baseline
    config_file, save_dir = runner.create_config(
        method, 
        contexts, 
        config_data.get("rope_config"), 
        config_data.get("extra_params", {})
    )
    
    print(f"Config file: {config_file}")
    print(f"Save dir: {save_dir}")
    
    # Run evaluation
    result = runner.run_evaluation(method, config_file, save_dir, timeout=120)
    print(f"Result: {result}")
    
    # Cleanup
    if config_file.exists():
        config_file.unlink()

if __name__ == "__main__":
    test_baseline_only()