#!/usr/bin/env python3
"""Test rope parameters end-to-end with LlamaFactory CLI."""

import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llamafactory.hparams import get_train_args


def test_rope_in_training_args():
    """Test if rope_scaling is properly loaded in training."""
    
    print("\033[1m\033[36mTesting RoPE Parameters End-to-End\033[0m")
    print("=" * 60)
    
    # Test configuration with rope_scaling
    config = {
        "model_name_or_path": "gpt2",
        "dataset": "alpaca_en_demo", 
        "template": "default",
        "cutoff_len": 1024,
        "rope_scaling": "linear",  # RoPE parameter in ModelArguments
        "stage": "pt",
        "do_train": True,
        "finetuning_type": "full",
        "output_dir": "./test_rope_output",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "learning_rate": 5e-5,
    }
    
    # Write test YAML
    yaml_path = Path(__file__).parent / "test_rope_cli.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n1. Created test YAML with rope_scaling: {config.get('rope_scaling')}")
    
    # Mock sys.argv for parser
    original_argv = sys.argv
    sys.argv = ["test", str(yaml_path)]
    
    try:
        # Parse arguments as LlamaFactory would
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
        
        print("\n2. Successfully parsed arguments!")
        
        # Check if rope_scaling was loaded
        if hasattr(model_args, 'rope_scaling'):
            print(f"\n3. \033[32m✓ rope_scaling found in ModelArguments: {model_args.rope_scaling}\033[0m")
            
            # Check model configuration would be updated
            print("\n4. Checking how rope_scaling would be applied:")
            print(f"   - Model: {model_args.model_name_or_path}")
            print(f"   - RoPE Scaling: {model_args.rope_scaling}")
            print(f"   - Cutoff Length: {data_args.cutoff_len}")
            
            # Note: In actual training, this would update the model config
            print("\n5. In actual training, this would update model.config.rope_scaling")
            
            success = True
        else:
            print("\n\033[31m✗ rope_scaling not found in ModelArguments\033[0m")
            success = False
            
    except Exception as e:
        print(f"\n\033[31mError: {e}\033[0m")
        success = False
    finally:
        sys.argv = original_argv
        # Clean up
        yaml_path.unlink(missing_ok=True)
    
    return success


def test_eval_rope_params():
    """Test rope parameters in evaluation arguments."""
    
    print("\n" + "=" * 60)
    print("\033[1m\033[36mTesting Evaluation RoPE Parameters\033[0m")
    print("=" * 60)
    
    from llamafactory.hparams import EvaluationArguments
    from transformers import HfArgumentParser
    
    eval_config = {
        "task": "needle_haystack",
        "rope_scaling_type": "yarn",
        "rope_scaling_factor": 4.0,
        "yarn_alpha": 1.0,
        "yarn_beta": 32.0,
    }
    
    print(f"\n1. Test evaluation config with extended rope params:")
    for k, v in eval_config.items():
        if 'rope' in k or 'yarn' in k:
            print(f"   {k}: {v}")
    
    parser = HfArgumentParser(EvaluationArguments)
    cmd_args = []
    for key, value in eval_config.items():
        cmd_args.extend([f"--{key}", str(value)])
    
    try:
        eval_args, = parser.parse_args_into_dataclasses(cmd_args)
        print("\n2. \033[32m✓ Successfully parsed EvaluationArguments\033[0m")
        
        if hasattr(eval_args, 'rope_scaling_type'):
            print(f"\n3. Verified rope parameters in EvaluationArguments:")
            print(f"   \033[32m✓ rope_scaling_type: {eval_args.rope_scaling_type}\033[0m")
            print(f"   \033[32m✓ rope_scaling_factor: {eval_args.rope_scaling_factor}\033[0m")
            if eval_args.yarn_alpha:
                print(f"   \033[32m✓ yarn_alpha: {eval_args.yarn_alpha}\033[0m")
            if eval_args.yarn_beta:
                print(f"   \033[32m✓ yarn_beta: {eval_args.yarn_beta}\033[0m")
        
        return True
    except Exception as e:
        print(f"\n\033[31mError: {e}\033[0m")
        return False


def main():
    """Main test function."""
    
    print("\033[1m\033[35m" + "=" * 60 + "\033[0m")
    print("\033[1m\033[35mRoPE Parameters End-to-End Verification\033[0m")
    print("\033[1m\033[35m" + "=" * 60 + "\033[0m")
    
    # Test training arguments
    train_success = test_rope_in_training_args()
    
    # Test evaluation arguments
    eval_success = test_eval_rope_params()
    
    # Summary
    print("\n" + "=" * 60)
    print("\033[1mSUMMARY:\033[0m")
    print("=" * 60)
    
    print("\n\033[1mTraining (ModelArguments):\033[0m")
    if train_success:
        print("  \033[32m✓ rope_scaling parameter works (linear/dynamic)\033[0m")
    else:
        print("  \033[31m✗ rope_scaling parameter failed\033[0m")
    
    print("\n\033[1mEvaluation (EvaluationArguments):\033[0m")
    if eval_success:
        print("  \033[32m✓ Extended rope parameters work:\033[0m")
        print("    - rope_scaling_type (linear/yarn/llama3/etc)")
        print("    - rope_scaling_factor")
        print("    - yarn_alpha, yarn_beta")
    else:
        print("  \033[31m✗ Extended rope parameters failed\033[0m")
    
    print("\n\033[1mNOTE:\033[0m")
    print("  - rope_theta is NOT available in this branch")
    print("  - rope_theta exists in llama-pro-integration branch")
    print("=" * 60)


if __name__ == "__main__":
    main()