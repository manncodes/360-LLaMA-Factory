#!/usr/bin/env python3

# Test RoPE configuration generation in benchmark runner

import sys
sys.path.append('scripts/methods')
from benchmark_runner import MethodBenchmarkRunner
import yaml
import tempfile
from pathlib import Path

def test_rope_configurations():
    print("Testing RoPE configurations...")
    
    runner = MethodBenchmarkRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "test_results", 0.0)
    
    # Test each method configuration
    methods_to_test = ["baseline", "linear", "dynamic", "yarn", "longrope", "nope"]
    
    for method in methods_to_test:
        print(f"\n=== Testing {method.upper()} method ===")
        
        if method not in runner.methods:
            print(f"✗ Method {method} not found in methods")
            continue
            
        config_data = runner.methods[method]
        try:
            config_file = runner.create_config(
                method, 
                config_data["contexts"], 
                config_data.get("rope_config"), 
                config_data.get("extra_params", {})
            )
            
            # Read and validate the generated config
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            print(f"✓ YAML generation successful")
            print(f"  Model: {config['model_name_or_path']}")
            print(f"  Max context: {config['cutoff_len']}")
            print(f"  Do sample: {config['do_sample']}")
            
            if 'temperature' in config:
                print(f"  Temperature: {config['temperature']}")
            else:
                print(f"  Temperature: NOT_SET (correct for deterministic)")
            
            if 'rope_scaling' in config:
                print(f"  RoPE scaling: {config['rope_scaling']}")
            else:
                print(f"  RoPE scaling: NOT_SET")
            
            # Check for any extra parameters that might cause issues
            problematic_keys = ['rope_factor', 'gradient_checkpointing', 'torch_dtype']
            for key in problematic_keys:
                if key in config:
                    print(f"  WARNING: Found potentially problematic key: {key} = {config[key]}")
            
            # Clean up
            config_file.unlink()
            
        except Exception as e:
            print(f"✗ Error with {method}: {e}")
    
    print("\n✅ RoPE configuration tests complete!")

if __name__ == "__main__":
    test_rope_configurations()