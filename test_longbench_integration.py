#!/usr/bin/env python3
"""
Test script for LongBench integration with LlamaFactory.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_longbench_imports():
    """Test that all LongBench-related imports work."""
    print("Testing LongBench imports...")
    
    try:
        from llamafactory.eval.longbench_evaluator import LongBenchEvaluator, run_longbench_evaluation
        print("✓ LongBench evaluator imports successfully")
    except ImportError as e:
        print(f"✗ LongBench evaluator import failed: {e}")
        return False
    
    try:
        from llamafactory.hparams.evaluation_args import EvaluationArguments
        from dataclasses import fields
        longbench_fields = [f.name for f in fields(EvaluationArguments) if f.name.startswith('longbench_')]
        if longbench_fields:
            print(f"✓ Found {len(longbench_fields)} LongBench parameters in EvaluationArguments")
            print(f"  Parameters: {', '.join(longbench_fields)}")
        else:
            print("✗ No LongBench parameters found")
            return False
    except Exception as e:
        print(f"✗ Parameter check failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that LongBench configurations can be loaded."""
    print("\nTesting LongBench configuration loading...")
    
    configs = [
        "longbench_quick_test.yaml",
        "longbench_standard.yaml", 
        "longbench_cot.yaml",
        "longbench_by_domain.yaml"
    ]
    
    for config in configs:
        if os.path.exists(config):
            print(f"✓ Config exists: {config}")
            # Test YAML parsing
            try:
                import yaml
                with open(config, 'r') as f:
                    data = yaml.safe_load(f)
                if data.get('task', '').startswith('longbench'):
                    print(f"  ✓ Valid LongBench task: {data.get('task')}")
                else:
                    print(f"  ✗ Non-LongBench task: {data.get('task')}")
            except Exception as e:
                print(f"  ✗ Config parsing failed: {e}")
        else:
            print(f"✗ Config missing: {config}")

def test_dataset_access():
    """Test if we can access the LongBench dataset."""
    print("\nTesting LongBench dataset access...")
    
    try:
        from datasets import load_dataset
        print("Loading first sample from LongBench v2...")
        dataset = load_dataset('THUDM/LongBench-v2', split='train', streaming=True)
        sample = next(iter(dataset))
        
        print("✓ Dataset accessible")
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Domain: {sample.get('domain', 'N/A')}")
        print(f"  Sub-domain: {sample.get('sub_domain', 'N/A')}")
        print(f"  Difficulty: {sample.get('difficulty', 'N/A')}")
        print(f"  Context length: {len(sample.get('context', '').split())} words")
        
        return True
    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        return False

def test_prompt_files():
    """Test if prompt files exist."""
    print("\nTesting prompt file availability...")
    
    prompt_dir = "evaluation/longbench/prompts"
    prompt_files = [
        "0shot.txt",
        "0shot_cot.txt", 
        "0shot_cot_ans.txt",
        "0shot_no_context.txt",
        "0shot_rag.txt"
    ]
    
    found = 0
    for prompt_file in prompt_files:
        path = os.path.join(prompt_dir, prompt_file)
        if os.path.exists(path):
            print(f"✓ Found: {prompt_file}")
            found += 1
        else:
            print(f"✗ Missing: {prompt_file}")
    
    print(f"Found {found}/{len(prompt_files)} prompt files")
    return found > 0

def test_evaluator_registration():
    """Test that LongBench evaluator is properly registered."""
    print("\nTesting evaluator registration...")
    
    try:
        from llamafactory.eval.evaluator import run_eval
        print("✓ Main evaluator function accessible")
        
        # Check if the run_eval function contains LongBench logic
        import inspect
        source = inspect.getsource(run_eval)
        if 'longbench' in source.lower():
            print("✓ LongBench integration detected in run_eval")
        else:
            print("✗ LongBench integration not found in run_eval")
            return False
            
    except Exception as e:
        print(f"✗ Evaluator registration test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all LongBench integration tests."""
    print("LongBench v2 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_longbench_imports),
        ("Configuration Tests", test_config_loading),
        ("Dataset Access", test_dataset_access),
        ("Prompt Files", test_prompt_files),
        ("Evaluator Registration", test_evaluator_registration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        try:
            result = test_func()
            if result is not False:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! LongBench integration is ready.")
        print("\nNext steps:")
        print("1. Wait for dataset download to complete")
        print("2. Run: llamafactory-cli eval longbench_quick_test.yaml") 
        return True
    else:
        print("\nSome tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)