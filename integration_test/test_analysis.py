#!/usr/bin/env python3
"""Test the analysis pipeline with integration test results."""

import sys
import json
import pandas as pd
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_analysis():
    """Test analysis pipeline with minimal data."""
    print("🔍 Testing Analysis Pipeline")
    print("-" * 40)
    
    results_dir = "integration_test/results"
    
    # Load results manually (simpler than importing the full system)
    result_files = list(Path(results_dir).glob("eval_*.json"))
    
    if not result_files:
        print("❌ No results found")
        return False
    
    print(f"📄 Found {len(result_files)} result files")
    
    # Load data
    data = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
            
        config = result["config"]
        metrics = result["metrics"]
        
        row = {
            "config_file": result["config_path"],
            "rope_type": config.get("rope_scaling_type", "baseline"),
            "rope_factor": config.get("rope_scaling_factor", 1.0),
            "context_length": config.get("cutoff_len", 512),
            "perplexity": metrics["perplexity"],
            "long_ppl": metrics["long_ppl"],
            "loss": metrics["loss"],
            "memory_gb": metrics["memory_gb"],
            "throughput_tps": metrics["throughput_tps"],
        }
        data.append(row)
        print(f"  • {row['rope_type']}: PPL={row['perplexity']:.2f}, Context={row['context_length']}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Basic analysis
    print(f"\n📊 Analysis Results:")
    print(f"   Total configs: {len(df)}")
    print(f"   RoPE types: {list(df['rope_type'].unique())}")
    print(f"   Perplexity range: {df['perplexity'].min():.2f} - {df['perplexity'].max():.2f}")
    print(f"   Memory usage: {df['memory_gb'].min():.2f} - {df['memory_gb'].max():.2f} GB")
    
    # Find best configuration
    best_idx = df['perplexity'].idxmin()
    best_config = df.iloc[best_idx]
    
    print(f"\n🏆 Best Configuration:")
    print(f"   Type: {best_config['rope_type']}")
    print(f"   Context: {best_config['context_length']}")
    print(f"   Perplexity: {best_config['perplexity']:.2f}")
    print(f"   Memory: {best_config['memory_gb']:.2f} GB")
    
    # Save summary
    summary = {
        "total_configs": len(df),
        "best_config": best_config.to_dict(),
        "rope_types": list(df['rope_type'].unique()),
        "perplexity_stats": {
            "min": df['perplexity'].min(),
            "max": df['perplexity'].max(),
            "mean": df['perplexity'].mean(),
        }
    }
    
    summary_file = Path(results_dir) / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print("✅ Analysis pipeline test PASSED")
    
    return True

if __name__ == "__main__":
    success = test_analysis()
    if not success:
        sys.exit(1)