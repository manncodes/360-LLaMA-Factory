#!/usr/bin/env python3
"""Demo script to test the RoPE hyperparameter sweep system."""

import os
import sys
import time
import shutil
from pathlib import Path


def run_demo():
    """Run a minimal demo of the sweep system."""
    print("🚀 RoPE Hyperparameter Sweep Demo")
    print("=" * 50)
    
    # Clean up previous demo results
    demo_dirs = ["demo_configs", "demo_results", "demo_analysis"]
    for dir_name in demo_dirs:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
    
    print("\n1. Generating demo configurations...")
    
    # Generate a small set of configurations for demo
    from generate_configs import generate_all_configs
    
    configs = generate_all_configs(
        model_key="fast",  # Use GPT-2 for speed
        rope_types=["linear", "yarn"],  # Just two methods
        context_keys=["short"],  # Just 4K context 
        output_dir="demo_configs"
    )
    
    print(f"   ✅ Generated {len(configs)} demo configurations")
    
    print("\n2. Running evaluations...")
    
    # Evaluate configurations
    import subprocess
    
    cmd = [
        sys.executable,
        "hyperparam_sweep/run_sweep.py",
        "--evaluate",
        "--config-dir", "demo_configs",
        "--num-gpus", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("   ✅ Evaluations completed successfully")
        else:
            print(f"   ❌ Evaluation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ⚠️ Evaluation timed out (this is expected for demo)")
        return False
    except Exception as e:
        print(f"   ❌ Error running evaluation: {e}")
        return False
    
    print("\n3. Analyzing results...")
    
    # Analyze results
    cmd = [
        sys.executable,
        "hyperparam_sweep/run_sweep.py",
        "--analyze"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   ✅ Analysis completed")
        else:
            print(f"   ❌ Analysis failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error in analysis: {e}")
    
    print("\n4. Creating visualizations...")
    
    # Create visualizations (without showing them)
    cmd = [
        sys.executable,
        "hyperparam_sweep/visualize_results.py",
        "--results-dir", "demo_results",
        "--output-dir", "demo_analysis"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   ✅ Visualizations created")
        else:
            print(f"   ⚠️ Visualization note: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️ Visualization note: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\nGenerated files:")
    
    # Show what was created
    for demo_dir in demo_dirs:
        if Path(demo_dir).exists():
            files = list(Path(demo_dir).glob("*"))
            if files:
                print(f"\n📁 {demo_dir}/")
                for file in files[:5]:  # Show first 5 files
                    print(f"   • {file.name}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more files")
    
    print("\n💡 Next steps:")
    print("   • Examine configurations: ls demo_configs/")
    print("   • Check results: ls demo_results/ (if any)")
    print("   • View analysis: ls demo_analysis/ (if any)")
    print("   • Run real sweep: python hyperparam_sweep/run_sweep.py --help")
    
    return True


def test_individual_components():
    """Test individual components of the sweep system."""
    print("\n🧪 Testing Individual Components")
    print("-" * 40)
    
    # Test 1: Configuration generation
    print("\n1. Testing configuration generation...")
    try:
        from generate_configs import generate_all_configs, estimate_compute_time
        
        configs = generate_all_configs(
            model_key="fast",
            rope_types=["linear"],
            context_keys=["short"],
            output_dir="test_configs"
        )
        
        if configs:
            print(f"   ✅ Generated {len(configs)} test configurations")
            
            # Test compute estimation
            estimates = estimate_compute_time(len(configs))
            print(f"   📊 Estimated compute: {estimates['hours']:.1f} hours")
        else:
            print("   ❌ No configurations generated")
            
    except Exception as e:
        print(f"   ❌ Configuration generation failed: {e}")
    
    # Test 2: Single evaluation (mock)
    print("\n2. Testing evaluation system...")
    try:
        # Just test imports and basic functionality
        from evaluate_perplexity import load_model_and_tokenizer
        print("   ✅ Evaluation system imports successful")
        
        # Could test with a tiny config here, but skip for speed
        print("   ⚠️ Skipping actual model loading (too slow for demo)")
        
    except Exception as e:
        print(f"   ❌ Evaluation system test failed: {e}")
    
    # Test 3: Results analysis (mock)
    print("\n3. Testing analysis system...")
    try:
        from visualize_results import load_results
        print("   ✅ Analysis system imports successful")
        print("   ⚠️ Skipping visualization (no results to analyze)")
        
    except Exception as e:
        print(f"   ❌ Analysis system test failed: {e}")
    
    # Clean up test files
    if Path("test_configs").exists():
        shutil.rmtree("test_configs")
    
    print("\n✅ Component testing completed!")


def main():
    """Main demo function."""
    print("RoPE Hyperparameter Sweep - Demo & Test")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-demo", action="store_true",
                       help="Run full demo with actual evaluations")
    parser.add_argument("--test-components", action="store_true", 
                       help="Test individual system components")
    parser.add_argument("--quick", action="store_true",
                       help="Quick component test only")
    
    args = parser.parse_args()
    
    if args.test_components or args.quick:
        test_individual_components()
    
    if args.full_demo and not args.quick:
        print("\n" + "=" * 50)
        run_demo()
    
    if not any([args.full_demo, args.test_components, args.quick]):
        print("\nSelect an option:")
        print("  --test-components  : Test system components")
        print("  --full-demo       : Run complete demo (slow)")
        print("  --quick           : Quick component test")
        print("\nExample: python hyperparam_sweep/demo.py --quick")


if __name__ == "__main__":
    main()