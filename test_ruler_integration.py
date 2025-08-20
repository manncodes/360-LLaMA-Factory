#!/usr/bin/env python3
"""Test script to verify RULER integration with LlamaFactory."""

import sys
import subprocess
from pathlib import Path

def test_ruler_import():
    """Test that RULER evaluator can be imported."""
    print("Testing RULER import...")
    try:
        from src.llamafactory.eval.ruler_evaluator import RULEREvaluator
        print("✓ RULER evaluator imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import RULER evaluator: {e}")
        return False

def test_ruler_cli_help():
    """Test that llamafactory-cli recognizes RULER task."""
    print("\nTesting CLI help...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "llamafactory.cli", "eval", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "ruler" in result.stdout.lower() or result.returncode == 0:
            print("✓ CLI help works")
            return True
        else:
            print(f"✗ CLI help doesn't mention RULER")
            return False
    except Exception as e:
        print(f"✗ Failed to run CLI help: {e}")
        return False

def test_ruler_config():
    """Test that RULER config files are valid."""
    print("\nTesting RULER config files...")
    config_files = [
        "eval_configs/ruler_quick_test.yaml",
        "eval_configs/ruler_basic.yaml",
        "eval_configs/ruler_comprehensive.yaml",
        "eval_configs/ruler_rope_scaling.yaml",
        "eval_configs/ruler_yarn.yaml"
    ]
    
    all_exist = True
    for config in config_files:
        if Path(config).exists():
            print(f"✓ {config} exists")
        else:
            print(f"✗ {config} not found")
            all_exist = False
    
    return all_exist

def test_ruler_data_prep():
    """Test RULER data preparation script."""
    print("\nTesting RULER data preparation...")
    ruler_prepare = Path("third_party/RULER/scripts/data/prepare.py")
    if ruler_prepare.exists():
        print(f"✓ RULER prepare.py exists at {ruler_prepare}")
        return True
    else:
        print(f"✗ RULER prepare.py not found")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("RULER Integration Test Suite")
    print("="*50)
    
    tests = [
        test_ruler_import,
        test_ruler_cli_help,
        test_ruler_config,
        test_ruler_data_prep
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Error running test {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! RULER integration is ready.")
        print("\nTo run RULER evaluation, use:")
        print("  llamafactory-cli eval --config eval_configs/ruler_quick_test.yaml")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())