#!/usr/bin/env python3
"""Validate advanced RoPE scaling configurations for continual pretraining."""

import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamafactory.hparams.parser import _TRAIN_ARGS
from transformers import HfArgumentParser

def test_advanced_rope_config(config_params, name):
    """Test if an advanced RoPE configuration is valid."""
    
    base_config = {
        "model_name_or_path": "unsloth/Llama-3.2-1B-Instruct",
        "dataset": "alpaca_en_demo",
        "template": "llama3",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "output_dir": f"test_output_{name}",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "bf16": True,
        "max_samples": 1
    }
    
    # Merge with rope params
    config = {**base_config, **config_params}
    
    # Save to temp file
    temp_file = f"temp_advanced_{name}.yaml"
    with open(temp_file, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Try to parse arguments with new advanced RoPE parameters
        parser = HfArgumentParser(_TRAIN_ARGS)
        model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_yaml_file(temp_file)
        
        # Check what was parsed
        result = {
            "status": "✅ SUPPORTED",
            "rope_scaling": getattr(model_args, 'rope_scaling', None),
            "rope_scaling_type": getattr(model_args, 'rope_scaling_type', None),
            "rope_scaling_factor": getattr(model_args, 'rope_scaling_factor', None),
            "rope_theta": getattr(model_args, 'rope_theta', None),
            "yarn_alpha": getattr(model_args, 'yarn_alpha', None),
            "yarn_beta": getattr(model_args, 'yarn_beta', None),
            "longrope_short_factor": getattr(model_args, 'longrope_short_factor', None),
            "longrope_long_factor": getattr(model_args, 'longrope_long_factor', None),
            "original_max_position": getattr(model_args, 'original_max_position', None),
            "cutoff_len": getattr(data_args, 'cutoff_len', None)
        }
        
        return result
        
    except ValueError as e:
        if "Some keys are not used" in str(e):
            unused_keys = str(e).split(': ')[-1]
            return {
                "status": "❌ UNSUPPORTED",
                "error": f"Invalid parameters: {unused_keys}"
            }
        else:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    except Exception as e:
        return {
            "status": "❌ ERROR", 
            "error": str(e)
        }
    finally:
        # Clean up temp file
        Path(temp_file).unlink(missing_ok=True)

def main():
    """Test all advanced RoPE configurations."""
    
    print("🧪 ADVANCED ROPE CONFIGURATION VALIDATION")
    print("=" * 70)
    
    # Test cases for advanced RoPE scaling
    test_cases = [
        {
            "name": "yarn_basic",
            "description": "YaRN scaling with basic parameters",
            "params": {
                "rope_scaling_type": "yarn",
                "rope_scaling_factor": 2.0,
                "yarn_alpha": 1.0,
                "yarn_beta": 32.0,
                "rope_theta": 500000.0,
                "cutoff_len": 8192
            }
        },
        {
            "name": "yarn_advanced",
            "description": "YaRN with original max position",
            "params": {
                "rope_scaling_type": "yarn",
                "rope_scaling_factor": 4.0,
                "yarn_alpha": 1.0,
                "yarn_beta": 32.0,
                "rope_theta": 1000000.0,
                "original_max_position": 4096,
                "cutoff_len": 16384
            }
        },
        {
            "name": "longrope_basic",
            "description": "LongRoPE scaling with default factors",
            "params": {
                "rope_scaling_type": "longrope",
                "rope_scaling_factor": 8.0,
                "rope_theta": 2000000.0,
                "cutoff_len": 32768
            }
        },
        {
            "name": "longrope_custom",
            "description": "LongRoPE with custom short/long factors",
            "params": {
                "rope_scaling_type": "longrope",
                "rope_scaling_factor": 8.0,
                "longrope_short_factor": [1.0, 1.5, 2.0],
                "longrope_long_factor": [1.0, 2.0, 4.0],
                "rope_theta": 2000000.0,
                "original_max_position": 4096,
                "cutoff_len": 32768
            }
        },
        {
            "name": "legacy_linear",
            "description": "Legacy linear scaling (backward compatibility)",
            "params": {
                "rope_scaling": "linear",
                "rope_theta": 1000000.0,
                "cutoff_len": 8192
            }
        },
        {
            "name": "mixed_params",
            "description": "Should fail - both rope_scaling and rope_scaling_type",
            "params": {
                "rope_scaling": "linear",
                "rope_scaling_type": "yarn",
                "cutoff_len": 4096
            }
        },
        {
            "name": "yarn_missing_factor",
            "description": "Should fail - YARN without scaling factor",
            "params": {
                "rope_scaling_type": "yarn",
                "yarn_alpha": 1.0,
                "yarn_beta": 32.0,
                "cutoff_len": 8192
            }
        }
    ]
    
    # Test each configuration
    results = {"supported": [], "unsupported": [], "errors": []}
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"  Description: {test_case['description']}")
        print(f"  Parameters: {test_case['params']}")
        
        result = test_advanced_rope_config(test_case['params'], test_case['name'])
        
        print(f"  Result: {result['status']}")
        if result['status'] == "✅ SUPPORTED":
            print(f"    rope_scaling_type: {result.get('rope_scaling_type')}")
            print(f"    rope_scaling_factor: {result.get('rope_scaling_factor')}")
            print(f"    rope_theta: {result.get('rope_theta')}")
            if result.get('yarn_alpha') is not None:
                print(f"    yarn_alpha: {result.get('yarn_alpha')}")
                print(f"    yarn_beta: {result.get('yarn_beta')}")
            if result.get('longrope_short_factor') is not None:
                print(f"    longrope_short_factor: {result.get('longrope_short_factor')}")
                print(f"    longrope_long_factor: {result.get('longrope_long_factor')}")
            results["supported"].append(test_case['name'])
        elif "UNSUPPORTED" in result['status']:
            print(f"    Reason: {result.get('error', 'Unknown')}")
            results["unsupported"].append(test_case['name'])
        else:
            print(f"    Error: {result.get('error', 'Unknown')}")
            results["errors"].append(test_case['name'])
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ADVANCED ROPE VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ SUPPORTED CONFIGURATIONS ({len(results['supported'])} total):")
    for name in results['supported']:
        test = next(t for t in test_cases if t['name'] == name)
        print(f"  • {name}: {test['description']}")
    
    print(f"\n❌ UNSUPPORTED CONFIGURATIONS ({len(results['unsupported'])} total):")
    for name in results['unsupported']:
        test = next(t for t in test_cases if t['name'] == name)
        print(f"  • {name}: {test['description']}")
    
    if results['errors']:
        print(f"\n⚠️  ERROR CONFIGURATIONS ({len(results['errors'])} total):")
        for name in results['errors']:
            test = next(t for t in test_cases if t['name'] == name)
            print(f"  • {name}: {test['description']}")
    
    # Final recommendations
    print("\n" + "=" * 70)
    print("💡 CONTINUAL PRETRAINING RECOMMENDATIONS")
    print("=" * 70)
    
    if len(results['supported']) >= 4:  # yarn_basic, yarn_advanced, longrope_basic, longrope_custom
        print("\n🎉 Advanced RoPE scaling is READY for continual pretraining!")
        print("\nRecommended workflow:")
        print("  1. Stage 1: Use yarn_basic config for 4K→8K extension")
        print("  2. Stage 2: Use yarn_advanced config for 8K→16K extension") 
        print("  3. Stage 3: Use longrope_custom config for 16K→32K extension")
        print("\nNext steps:")
        print("  • cd continual_pretraining_configs")
        print("  • python3 run_progressive_training.py --validate-only")
        print("  • python3 run_progressive_training.py --dry-run")
    else:
        print("\n⚠️  Advanced RoPE scaling implementation incomplete")
        print("Check ModelArguments and RoPE configuration implementation")
    
    return len(results['errors']) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)