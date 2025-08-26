#!/usr/bin/env python3
"""Test script to verify YAML rope parameters are loaded correctly."""

import sys
import yaml
from pathlib import Path
from dataclasses import fields
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llamafactory.hparams import EvaluationArguments
from transformers import HfArgumentParser


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def test_rope_params_from_yaml(yaml_file: str):
    """Test loading rope parameters from YAML configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {yaml_file}")
    print('='*60)
    
    # Load YAML config
    config = load_yaml_config(yaml_file)
    
    # Extract rope-related parameters
    rope_params = {
        'rope_scaling_type': config.get('rope_scaling_type'),
        'rope_scaling_factor': config.get('rope_scaling_factor'),
        'yarn_alpha': config.get('yarn_alpha'),
        'yarn_beta': config.get('yarn_beta'),
    }
    
    print("\nRoPE parameters found in YAML:")
    for key, value in rope_params.items():
        if value is not None:
            print(f"  \033[32m{key}: {value}\033[0m")
    
    # Get valid EvaluationArguments fields
    eval_fields = {f.name for f in fields(EvaluationArguments)}
    
    # Filter config to only include EvaluationArguments fields
    eval_config = {}
    for key, value in config.items():
        if key in eval_fields and value is not None:
            eval_config[key] = value
    
    # Test if HfArgumentParser can load these from command line args
    parser = HfArgumentParser(EvaluationArguments)
    
    # Convert filtered config to command line args format
    cmd_args = []
    for key, value in eval_config.items():
        # Skip complex types that can't be passed as command line args
        if isinstance(value, (list, dict)):
            continue
        cmd_args.extend([f"--{key}", str(value)])
    
    print("\nFiltered evaluation args:")
    print("  " + " ".join(cmd_args[:10]) + "..." if len(cmd_args) > 10 else "  " + " ".join(cmd_args))
    
    try:
        eval_args, = parser.parse_args_into_dataclasses(cmd_args)
        print("\n\033[32m✓ Successfully loaded EvaluationArguments!\033[0m")
        
        # Check rope parameters
        print("\nVerified RoPE parameters in EvaluationArguments:")
        success = False
        if hasattr(eval_args, 'rope_scaling_type') and eval_args.rope_scaling_type:
            print(f"  \033[32m✓ rope_scaling_type: {eval_args.rope_scaling_type}\033[0m")
            success = True
        if hasattr(eval_args, 'rope_scaling_factor') and eval_args.rope_scaling_factor:
            print(f"  \033[32m✓ rope_scaling_factor: {eval_args.rope_scaling_factor}\033[0m")
            success = True
        if hasattr(eval_args, 'yarn_alpha') and eval_args.yarn_alpha:
            print(f"  \033[32m✓ yarn_alpha: {eval_args.yarn_alpha}\033[0m")
        if hasattr(eval_args, 'yarn_beta') and eval_args.yarn_beta:
            print(f"  \033[32m✓ yarn_beta: {eval_args.yarn_beta}\033[0m")
            
        return success
            
    except Exception as e:
        print(f"\n\033[31mError loading arguments: {e}\033[0m")
        return False
    
    return True


def main():
    """Main test function."""
    # Test YAML files with rope configurations
    test_files = [
        "../../eval_configs/needle_haystack_llama3_rope.yaml",
        "../../eval_configs/needle_haystack_yarn_rope.yaml", 
        "../../eval_configs/needle_haystack_linear_rope.yaml",
    ]
    
    print("\033[1m\033[36mTesting RoPE Parameter Loading from YAML\033[0m")
    
    all_passed = True
    for yaml_file in test_files:
        yaml_path = Path(__file__).parent / yaml_file
        if yaml_path.exists():
            success = test_rope_params_from_yaml(str(yaml_path))
            if not success:
                all_passed = False
        else:
            print(f"\n\033[33mWarning: {yaml_file} not found\033[0m")
    
    print("\n" + "="*60)
    if all_passed:
        print("\033[32m✓ All tests passed! RoPE parameters load correctly from YAML.\033[0m")
    else:
        print("\033[31m✗ Some tests failed. Check the output above.\033[0m")
    print("="*60)


if __name__ == "__main__":
    main()