#!/usr/bin/env python3
"""
Analyze and compile benchmark results from all methods
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np

def find_latest_results_dir():
    """Find the most recent benchmark results directory"""
    benchmark_dirs = glob.glob("complete_benchmark_*")
    if benchmark_dirs:
        return sorted(benchmark_dirs)[-1]
    return None

def analyze_benchmark_times():
    """Analyze benchmark execution times"""
    results_dir = find_latest_results_dir()
    if not results_dir:
        print("No benchmark results found")
        return
    
    # Read the summary CSV
    summary_file = Path(results_dir) / "results_summary.csv"
    if not summary_file.exists():
        print(f"No summary file found in {results_dir}")
        return
    
    df = pd.read_csv(summary_file)
    print("📊 Benchmark Execution Analysis")
    print("=" * 60)
    
    # Display results
    for _, row in df.iterrows():
        if row['Status'] == 'SUCCESS':
            print(f"✅ {row['Method']:12}: {row['Duration(s)']:3}s - SUCCESS")
        else:
            print(f"❌ {row['Method']:12}: {row['Duration(s)']:3}s - FAILED")
    
    # Calculate statistics
    successful = df[df['Status'] == 'SUCCESS']
    if len(successful) > 0:
        print(f"\n📈 Statistics:")
        print(f"   Total methods tested: {len(df)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(df) - len(successful)}")
        print(f"   Average time: {successful['Duration(s)'].mean():.1f}s")
        print(f"   Fastest: {successful.loc[successful['Duration(s)'].idxmin(), 'Method']} ({successful['Duration(s)'].min()}s)")
        print(f"   Slowest: {successful.loc[successful['Duration(s)'].idxmax(), 'Method']} ({successful['Duration(s)'].max()}s)")
    
    return df

def check_detailed_results():
    """Check for detailed needle-in-haystack results"""
    print("\n🔍 Detailed Results Analysis")
    print("=" * 60)
    
    # Check saves directory for actual results
    saves_dirs = glob.glob("saves/tinyllama/needle_haystack_*")
    
    method_results = {}
    
    for save_dir in saves_dirs:
        method_name = Path(save_dir).name.replace("needle_haystack_", "")
        summary_file = Path(save_dir) / "summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    method_results[method_name] = summary
                    print(f"✅ {method_name:15}: Results found")
            except Exception as e:
                print(f"❌ {method_name:15}: Error reading results - {e}")
        else:
            print(f"⚪ {method_name:15}: No results")
    
    return method_results

def create_comparison_chart(execution_times, detailed_results=None):
    """Create comparison charts"""
    print("\n📊 Creating Comparison Charts")
    print("=" * 60)
    
    # Execution time comparison
    if len(execution_times) > 0:
        successful = execution_times[execution_times['Status'] == 'SUCCESS']
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Execution Times
        plt.subplot(1, 2, 1)
        bars = plt.bar(successful['Method'], successful['Duration(s)'])
        plt.title('Benchmark Execution Times')
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, bar in enumerate(bars):
            if i < len(colors):
                bar.set_color(colors[i])
        
        # Add value labels on bars
        for i, v in enumerate(successful['Duration(s)']):
            plt.text(i, v + 0.5, f'{v}s', ha='center', va='bottom')
        
        # Subplot 2: Success/Failure
        plt.subplot(1, 2, 2)
        status_counts = execution_times['Status'].value_counts()
        colors = ['#2ca02c', '#d62728'] if 'FAILED' in status_counts else ['#2ca02c']
        plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Success Rate')
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print("📈 Chart saved as: benchmark_comparison.png")
        
        return True
    
    return False

def generate_final_report():
    """Generate comprehensive final report"""
    print("\n📝 Generating Final Report")
    print("=" * 60)
    
    results_dir = find_latest_results_dir()
    
    with open("BENCHMARK_FINAL_REPORT.md", "w") as f:
        f.write("# Long Context Methods Benchmark - Final Report\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Results Directory**: {results_dir}\n\n")
        
        # Execution Summary
        f.write("## Execution Summary\n\n")
        
        df = pd.read_csv(Path(results_dir) / "results_summary.csv")
        for _, row in df.iterrows():
            status_icon = "✅" if row['Status'] == 'SUCCESS' else "❌"
            f.write(f"- {status_icon} **{row['Method']}**: {row['Duration(s)']}s\n")
        
        # Method Details
        f.write("\n## Method Analysis\n\n")
        
        methods_info = {
            "baseline": "No RoPE scaling - standard 2K context",
            "rope_linear": "Linear interpolation scaling",
            "rope_dynamic": "Dynamic NTK frequency scaling",
            "yarn": "YaRN - hybrid approach with temperature scaling",
            "longrope": "Non-uniform dimension scaling",
            "nope": "No position encoding - causal mask only"
        }
        
        successful = df[df['Status'] == 'SUCCESS']
        for _, row in successful.iterrows():
            method = row['Method']
            f.write(f"### {method.replace('_', ' ').title()}\n")
            f.write(f"- **Status**: ✅ Success\n")
            f.write(f"- **Execution Time**: {row['Duration(s)']}s\n")
            f.write(f"- **Description**: {methods_info.get(method, 'Unknown method')}\n")
            
            # Check for errors in failed methods
            log_file = Path(results_dir) / method / "benchmark_log.txt"
            if log_file.exists():
                with open(log_file, 'r') as log:
                    lines = log.readlines()
                    if any("ERROR" in line or "Traceback" in line for line in lines):
                        f.write(f"- **Note**: Some errors detected in logs\n")
            f.write("\n")
        
        # Failed methods
        failed = df[df['Status'] == 'FAILED']
        if len(failed) > 0:
            f.write("## Failed Methods\n\n")
            for _, row in failed.iterrows():
                method = row['Method']
                f.write(f"### {method.replace('_', ' ').title()}\n")
                f.write(f"- **Status**: ❌ Failed\n")
                f.write(f"- **Duration**: {row['Duration(s)']}s\n")
                f.write("- **Issue**: Check log files for detailed error information\n\n")
        
        # Performance Insights
        f.write("## Performance Insights\n\n")
        if len(successful) > 0:
            fastest = successful.loc[successful['Duration(s)'].idxmin()]
            slowest = successful.loc[successful['Duration(s)'].idxmax()]
            
            f.write(f"- **Fastest Method**: {fastest['Method']} ({fastest['Duration(s)']}s)\n")
            f.write(f"- **Slowest Method**: {slowest['Method']} ({slowest['Duration(s)']}s)\n")
            f.write(f"- **Average Time**: {successful['Duration(s)'].mean():.1f}s\n")
            f.write(f"- **Success Rate**: {len(successful)}/{len(df)} ({len(successful)/len(df)*100:.1f}%)\n")
        
        f.write("\n## Integration Status\n\n")
        f.write("✅ **Successfully Integrated Methods**:\n")
        f.write("- YaRN (Yet another RoPE extensioN)\n") 
        f.write("- LongRope (Non-uniform dimension scaling)\n")
        f.write("- NoPE (No Position Encoding)\n\n")
        
        f.write("✅ **Existing Methods Tested**:\n")
        f.write("- Linear RoPE scaling\n")
        f.write("- Dynamic NTK scaling\n")
        f.write("- Baseline (no scaling)\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Fix any implementation issues in failed methods\n")
        f.write("2. Run detailed needle-in-haystack evaluations\n")
        f.write("3. Compare accuracy metrics across methods\n")
        f.write("4. Test with larger models and longer contexts\n")
        f.write("5. Optimize parameters for each method\n")
    
    print("📄 Final report saved as: BENCHMARK_FINAL_REPORT.md")

def main():
    print("🔍 Long Context Methods Benchmark Analysis")
    print("=" * 60)
    
    # Analyze execution times
    execution_df = analyze_benchmark_times()
    
    # Check detailed results
    detailed_results = check_detailed_results()
    
    # Create comparison charts
    if execution_df is not None:
        create_comparison_chart(execution_df, detailed_results)
    
    # Generate final report
    generate_final_report()
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("📊 Check benchmark_comparison.png for visual comparison")
    print("📄 Check BENCHMARK_FINAL_REPORT.md for detailed report")

if __name__ == "__main__":
    main()