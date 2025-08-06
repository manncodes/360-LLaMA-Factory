#!/usr/bin/env python3
"""
LongBench-v2 Evaluation Runner
Integrated with 360-LLaMA-Factory evaluation framework
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_longbench_v2_evaluation(config_file: str = None):
    """Run LongBench-v2 evaluation with specified config"""
    
    if config_file is None:
        config_file = "evaluation/longbench_v2/longbench_v2_config.yaml"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found")
        return
    
    # Set sys.argv for YAML parsing (360-LLaMA-Factory style)
    sys.argv = ['eval', config_file]
    
    print(f"🚀 Starting LongBench-v2 evaluation with config: {config_file}")
    
    try:
        # Import and run LongBench-v2 evaluation
        from llamafactory.eval.longbench_v2_evaluator import run_longbench_v2_eval
        
        # Run evaluation
        results = run_longbench_v2_eval()
        
        if results:
            print("✅ LongBench-v2 evaluation completed successfully!")
            print(f"📊 Overall accuracy: {results['overall_accuracy']:.1%}")
        else:
            print("❌ Evaluation failed or returned no results")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_longbench_v2_evaluation(config_file)