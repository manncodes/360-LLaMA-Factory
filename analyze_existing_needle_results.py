#!/usr/bin/env python3
"""
Analyze existing needle-in-haystack results to understand actual performance
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_results():
    """Load all existing needle-in-haystack results"""
    results = {}
    saves_dir = Path("saves/tinyllama")
    
    for result_dir in saves_dir.glob("needle_haystack_*"):
        if result_dir.is_dir():
            summary_file = result_dir / "summary.json"
            detailed_file = result_dir / "detailed_results.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    detailed = None
                    if detailed_file.exists():
                        with open(detailed_file, 'r') as f:
                            detailed = json.load(f)
                    
                    method_name = result_dir.name.replace("needle_haystack_", "")
                    results[method_name] = {
                        'summary': summary,
                        'detailed': detailed,
                        'dir': result_dir
                    }
                    print(f"✅ Loaded results for: {method_name}")
                except Exception as e:
                    print(f"❌ Error loading {result_dir.name}: {e}")
    
    return results

def analyze_performance(results):
    """Analyze performance across different methods"""
    print("\n📊 ACTUAL Needle-in-Haystack Performance Analysis")
    print("=" * 60)
    
    performance_data = []
    
    for method_name, data in results.items():
        summary = data['summary']
        overall = summary.get('overall', {})
        
        accuracy = overall.get('average_score', 0)
        exact_match = overall.get('exact_match_rate', 0)
        total_examples = overall.get('total_examples', 0)
        
        print(f"\n🔍 {method_name.upper()}")
        print(f"   Overall Accuracy: {accuracy:.1%}")
        print(f"   Exact Match Rate: {exact_match:.1%}")
        print(f"   Total Examples: {total_examples}")
        
        # Analyze by context length
        by_length = summary.get('by_context_length', {})
        if by_length:
            print("   Performance by Context Length:")
            for length in sorted(by_length.keys(), key=int):
                stats = by_length[length]
                acc = stats.get('average_score', 0)
                count = stats.get('count', 0)
                print(f"     {length:>4} tokens: {acc:.1%} ({count} examples)")
        
        # Analyze by needle position
        by_depth = summary.get('by_depth_percent', {})
        if by_depth:
            print("   Performance by Needle Position:")
            for depth in sorted(by_depth.keys(), key=float):
                stats = by_depth[depth]
                acc = stats.get('average_score', 0)
                count = stats.get('count', 0)
                print(f"     {float(depth):>5.1f}% depth: {acc:.1%} ({count} examples)")
        
        # Store for comparison
        performance_data.append({
            'method': method_name,
            'accuracy': accuracy,
            'exact_match': exact_match,
            'total_examples': total_examples
        })
    
    return performance_data

def create_comparison_charts(performance_data, results):
    """Create visual comparisons"""
    print("\n📊 Creating Performance Charts")
    print("=" * 60)
    
    if not performance_data:
        print("No data to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall Accuracy Comparison
    methods = [d['method'] for d in performance_data]
    accuracies = [d['accuracy'] for d in performance_data]
    
    ax1.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(methods)])
    ax1.set_title('Overall Needle-in-Haystack Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.1)
    
    # Add percentage labels on bars
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Exact Match Rate
    exact_matches = [d['exact_match'] for d in performance_data]
    ax2.bar(methods, exact_matches, color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd'][:len(methods)])
    ax2.set_title('Exact Match Rate')
    ax2.set_ylabel('Exact Match Rate')
    ax2.set_ylim(0, 1.1)
    
    for i, v in enumerate(exact_matches):
        ax2.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Performance by Context Length (if we have detailed data)
    context_length_data = []
    for method_name, data in results.items():
        by_length = data['summary'].get('by_context_length', {})
        for length, stats in by_length.items():
            context_length_data.append({
                'method': method_name,
                'context_length': int(length),
                'accuracy': stats.get('average_score', 0)
            })
    
    if context_length_data:
        df = pd.DataFrame(context_length_data)
        
        # Pivot for heatmap
        pivot_df = df.pivot(index='method', columns='context_length', values='accuracy')
        sns.heatmap(pivot_df, annot=True, fmt='.2%', cmap='RdYlGn', 
                   ax=ax3, cbar_kws={'label': 'Accuracy'})
        ax3.set_title('Accuracy by Context Length')
        ax3.set_xlabel('Context Length (tokens)')
        ax3.set_ylabel('Method')
    else:
        ax3.text(0.5, 0.5, 'No context length data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Accuracy by Context Length')
    
    # 4. Performance by Needle Position
    position_data = []
    for method_name, data in results.items():
        by_depth = data['summary'].get('by_depth_percent', {})
        for depth, stats in by_depth.items():
            position_data.append({
                'method': method_name,
                'position': float(depth),
                'accuracy': stats.get('average_score', 0)
            })
    
    if position_data:
        df = pd.DataFrame(position_data)
        
        # Line plot for position performance
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax4.plot(method_data['position'], method_data['accuracy'], 
                    marker='o', label=method, linewidth=2)
        
        ax4.set_title('Accuracy by Needle Position')
        ax4.set_xlabel('Position in Context (%)')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
    else:
        ax4.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Accuracy by Needle Position')
    
    plt.tight_layout()
    plt.savefig('needle_haystack_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Analysis chart saved as: needle_haystack_analysis.png")

def generate_performance_report(performance_data, results):
    """Generate detailed performance report"""
    print("\n📝 Generating Performance Report")
    print("=" * 60)
    
    with open("NEEDLE_HAYSTACK_PERFORMANCE_REPORT.md", "w") as f:
        f.write("# Needle-in-Haystack Performance Report\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Methods Analyzed**: {len(performance_data)}\n\n")
        
        # Overall performance table
        f.write("## Overall Performance Summary\n\n")
        f.write("| Method | Overall Accuracy | Exact Match Rate | Total Examples |\n")
        f.write("|--------|------------------|------------------|----------------|\n")
        
        # Sort by accuracy descending
        sorted_data = sorted(performance_data, key=lambda x: x['accuracy'], reverse=True)
        
        for i, data in enumerate(sorted_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
            f.write(f"| {rank} {data['method']} | {data['accuracy']:.1%} | "
                   f"{data['exact_match']:.1%} | {data['total_examples']} |\n")
        
        # Detailed analysis for each method
        f.write("\n## Detailed Method Analysis\n\n")
        
        for method_name, data in results.items():
            f.write(f"### {method_name.replace('_', ' ').title()}\n\n")
            
            summary = data['summary']
            overall = summary.get('overall', {})
            
            f.write(f"- **Overall Accuracy**: {overall.get('average_score', 0):.1%}\n")
            f.write(f"- **Exact Match Rate**: {overall.get('exact_match_rate', 0):.1%}\n")
            f.write(f"- **Total Examples**: {overall.get('total_examples', 0)}\n\n")
            
            # Context length performance
            by_length = summary.get('by_context_length', {})
            if by_length:
                f.write("**Performance by Context Length:**\n")
                for length in sorted(by_length.keys(), key=int):
                    stats = by_length[length]
                    acc = stats.get('average_score', 0)
                    count = stats.get('count', 0)
                    f.write(f"- {length} tokens: {acc:.1%} ({count} examples)\n")
                f.write("\n")
            
            # Position performance
            by_depth = summary.get('by_depth_percent', {})
            if by_depth:
                f.write("**Performance by Needle Position:**\n")
                for depth in sorted(by_depth.keys(), key=float):
                    stats = by_depth[depth]
                    acc = stats.get('average_score', 0)
                    count = stats.get('count', 0)
                    f.write(f"- {float(depth):.1f}% through context: {acc:.1%} ({count} examples)\n")
                f.write("\n")
        
        # Key insights
        f.write("## Key Insights\n\n")
        
        if len(sorted_data) > 0:
            best = sorted_data[0]
            f.write(f"- **Best Performing Method**: {best['method']} ({best['accuracy']:.1%} accuracy)\n")
            
            if len(sorted_data) > 1:
                worst = sorted_data[-1]
                f.write(f"- **Lowest Performing Method**: {worst['method']} ({worst['accuracy']:.1%} accuracy)\n")
                
                if best['accuracy'] > 0:
                    improvement = (best['accuracy'] - worst['accuracy']) / worst['accuracy'] * 100 if worst['accuracy'] > 0 else 0
                    f.write(f"- **Performance Gap**: {improvement:.1f}% relative improvement from worst to best\n")
        
        # Context length insights
        all_context_lengths = set()
        for data in results.values():
            by_length = data['summary'].get('by_context_length', {})
            all_context_lengths.update(by_length.keys())
        
        if all_context_lengths:
            max_length = max(int(l) for l in all_context_lengths)
            f.write(f"- **Maximum Context Length Tested**: {max_length} tokens\n")
        
        f.write("\n## Methodology\n\n")
        f.write("- **Task**: Needle-in-haystack retrieval\n")
        f.write("- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n")
        f.write("- **Background Text**: Paul Graham essays\n")
        f.write("- **Evaluation**: Exact string matching for needle retrieval\n")
    
    print("📄 Performance report saved as: NEEDLE_HAYSTACK_PERFORMANCE_REPORT.md")

def main():
    print("🔍 Analyzing Existing Needle-in-Haystack Results")
    print("=" * 60)
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No needle-in-haystack results found!")
        print("   Looking for results in saves/tinyllama/needle_haystack_*/")
        return
    
    # Analyze performance
    performance_data = analyze_performance(results)
    
    # Create visualizations
    create_comparison_charts(performance_data, results)
    
    # Generate report
    generate_performance_report(performance_data, results)
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("📊 Check needle_haystack_analysis.png for visual analysis")
    print("📄 Check NEEDLE_HAYSTACK_PERFORMANCE_REPORT.md for detailed report")

if __name__ == "__main__":
    main()