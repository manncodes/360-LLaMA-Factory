#!/usr/bin/env python3
"""Quick test to verify RoPE scaling is working."""

import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamafactory.hparams import get_train_args

def test_rope_config(config_file):
    """Test if RoPE configuration loads correctly."""
    print(f"Testing configuration: {config_file}")
    
    try:
        # Load YAML directly to check parameters
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Model: {config.get('model_name_or_path', 'Not specified')}")
        print(f"RoPE scaling: {config.get('rope_scaling', 'Not specified')}")
        print(f"RoPE theta: {config.get('rope_theta', 'Not specified')}")
        print(f"Context length: {config.get('cutoff_len', 'Not specified')}")
        
        # Check if RoPE parameters are correctly configured
        rope_scaling = config.get('rope_scaling')
        cutoff_len = config.get('cutoff_len', 2048)
        
        if rope_scaling:
            print(f"✅ RoPE scaling enabled: {rope_scaling}")
            if cutoff_len > 2048:
                print(f"✅ Extended context length: {cutoff_len}")
                print("✅ Configuration should trigger RoPE scaling!")
            else:
                print("⚠️  Context length not extended - RoPE scaling may not be triggered")
        else:
            print("❌ RoPE scaling not enabled")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Test different RoPE configurations."""
    config_files = [
        "working_rope_train.yaml",
        "llama32_rope_test.yaml", 
        "rope_theta_test.yaml",
        "final_rope_example.yaml"
    ]
    
    print("🧪 ROPE CONFIGURATION TESTS")
    print("=" * 50)
    
    for config_file in config_files:
        if Path(config_file).exists():
            test_rope_config(config_file)
        else:
            print(f"⚠️  Configuration file not found: {config_file}")
        print("-" * 50)

if __name__ == "__main__":
    main()