#!/usr/bin/env python3

# Test temperature configuration in benchmark runner

import sys
sys.path.append('scripts/methods')
from benchmark_runner import MethodBenchmarkRunner
import yaml

def test_temperature_configs():
    print("Testing temperature configurations...")
    
    # Test different temperature settings
    temperatures = [0.0, 0.3, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n=== Testing temperature {temp} ===")
        runner = MethodBenchmarkRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "test_results", temp)
        
        # Test baseline config
        config_data = runner.methods["baseline"]
        config_file = runner.create_config("baseline", config_data["contexts"], 
                                          config_data.get("rope_config"), 
                                          config_data.get("extra_params", {}))
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Temperature: {config['temperature']}")
        print(f"✓ Do sample: {config['do_sample']}")
        
        expected_do_sample = temp > 0.0
        if config['do_sample'] == expected_do_sample:
            print("✓ Sampling correctly configured")
        else:
            print("✗ Sampling incorrectly configured")
        
        # Clean up
        config_file.unlink()
    
    print("\n✅ Temperature configuration tests complete!")

if __name__ == "__main__":
    test_temperature_configs()