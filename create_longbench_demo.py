#!/usr/bin/env python3
"""
Create mock LongBench evaluation results and visualizations.
This demonstrates the full evaluation pipeline and visualization capabilities.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

def create_mock_longbench_results():
    """Create realistic mock LongBench evaluation results."""
    
    # Create save directory
    save_dir = "saves/longbench_demo_mock"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating mock LongBench v2 evaluation results...")
    
    # Define domains and their characteristics
    domains = {
        "Single-Document QA": {"samples": 15, "difficulty": 0.65},
        "Multi-Document QA": {"samples": 12, "difficulty": 0.45},
        "Long In-context Learning": {"samples": 10, "difficulty": 0.35},
        "Long-dialogue History Understanding": {"samples": 8, "difficulty": 0.55},
        "Code Repo Understanding": {"samples": 6, "difficulty": 0.40},
        "Long Structured Data Understanding": {"samples": 4, "difficulty": 0.50},
    }
    
    difficulties = ["easy", "hard"]
    lengths = ["8k-32k", "32k-128k", "128k+"]
    
    # Generate detailed results
    detailed_results = []
    sample_id = 1
    
    for domain, info in domains.items():
        for i in range(info["samples"]):
            # Simulate realistic performance with some randomness
            base_accuracy = info["difficulty"]
            noise = np.random.normal(0, 0.15)
            is_correct = np.random.random() < max(0.1, min(0.9, base_accuracy + noise))
            
            # Select random characteristics
            difficulty = np.random.choice(difficulties)
            length = np.random.choice(lengths)
            
            # Difficulty adjustment
            if difficulty == "hard":
                is_correct = is_correct and (np.random.random() < 0.7)
            
            # Generate mock data
            choices = ["A", "B", "C", "D"]
            answer = np.random.choice(choices)
            pred = answer if is_correct else np.random.choice([c for c in choices if c != answer])
            
            result = {
                "_id": f"mock_{sample_id:03d}",
                "domain": domain,
                "sub_domain": f"{domain.split()[0]} Sub-task",
                "difficulty": difficulty,
                "length": length,
                "question": f"Mock question {sample_id} for {domain}",
                "choice_A": "Mock choice A",
                "choice_B": "Mock choice B", 
                "choice_C": "Mock choice C",
                "choice_D": "Mock choice D",
                "answer": answer,
                "context_preview": f"Mock context for {domain} (truncated)...",
                "response": f"The correct answer is ({pred})",
                "pred": pred,
                "judge": is_correct
            }
            
            detailed_results.append(result)
            sample_id += 1
    
    # Save detailed results
    detail_file = os.path.join(save_dir, "longbench_standard_details.jsonl")
    with open(detail_file, 'w', encoding='utf-8') as f:
        for result in detailed_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Calculate metrics
    metrics = calculate_metrics(detailed_results)
    
    # Create summary
    summary = {
        "config": {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "mode": "standard",
            "max_length": 8192,
            "temperature": 0.1,
            "batch_size": 1,
            "timestamp": datetime.now().isoformat()
        },
        "metrics": metrics,
        "total_samples": len(detailed_results),
        "note": "Mock evaluation results for demonstration"
    }
    
    # Save summary
    summary_file = os.path.join(save_dir, "longbench_standard_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created {len(detailed_results)} mock evaluation results")
    print(f"✓ Saved detailed results: {detail_file}")
    print(f"✓ Saved summary: {summary_file}")
    
    return summary, detailed_results

def calculate_metrics(results):
    """Calculate evaluation metrics from results."""
    metrics = {
        'overall': {'correct': 0, 'total': 0, 'accuracy': 0.0},
        'by_domain': {},
        'by_sub_domain': {},
        'by_difficulty': {},
        'by_length': {},
    }
    
    for result in results:
        # Overall
        metrics['overall']['total'] += 1
        if result['judge']:
            metrics['overall']['correct'] += 1
        
        # By domain
        domain = result['domain']
        if domain not in metrics['by_domain']:
            metrics['by_domain'][domain] = {'correct': 0, 'total': 0}
        metrics['by_domain'][domain]['total'] += 1
        if result['judge']:
            metrics['by_domain'][domain]['correct'] += 1
        
        # By sub-domain
        sub_domain = result['sub_domain']
        if sub_domain not in metrics['by_sub_domain']:
            metrics['by_sub_domain'][sub_domain] = {'correct': 0, 'total': 0}
        metrics['by_sub_domain'][sub_domain]['total'] += 1
        if result['judge']:
            metrics['by_sub_domain'][sub_domain]['correct'] += 1
        
        # By difficulty
        difficulty = result['difficulty']
        if difficulty not in metrics['by_difficulty']:
            metrics['by_difficulty'][difficulty] = {'correct': 0, 'total': 0}
        metrics['by_difficulty'][difficulty]['total'] += 1
        if result['judge']:
            metrics['by_difficulty'][difficulty]['correct'] += 1
        
        # By length
        length = result['length']
        if length not in metrics['by_length']:
            metrics['by_length'][length] = {'correct': 0, 'total': 0}
        metrics['by_length'][length]['total'] += 1
        if result['judge']:
            metrics['by_length'][length]['correct'] += 1
    
    # Calculate accuracies
    if metrics['overall']['total'] > 0:
        metrics['overall']['accuracy'] = metrics['overall']['correct'] / metrics['overall']['total']
    
    for category_metrics in [metrics['by_domain'], metrics['by_sub_domain'], 
                            metrics['by_difficulty'], metrics['by_length']]:
        for key, values in category_metrics.items():
            if values['total'] > 0:
                values['accuracy'] = values['correct'] / values['total']
            else:
                values['accuracy'] = 0.0
    
    return metrics

def create_visualizations(summary, results):
    """Create comprehensive visualizations of LongBench results."""
    
    print("\nCreating LongBench visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots directory
    plots_dir = "saves/longbench_demo_mock/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    metrics = summary['metrics']
    
    # 1. Overall Performance Summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LongBench v2 Evaluation Results - TinyLlama Demo', fontsize=16, fontweight='bold')
    
    # Overall accuracy
    overall_acc = metrics['overall']['accuracy']
    ax1.bar(['Overall Accuracy'], [overall_acc], color='skyblue', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Performance')
    ax1.set_ylim(0, 1)
    ax1.text(0, overall_acc + 0.02, f'{overall_acc:.1%}', ha='center', fontweight='bold')
    
    # Domain performance
    domains = list(metrics['by_domain'].keys())
    domain_accs = [metrics['by_domain'][d]['accuracy'] for d in domains]
    domain_totals = [metrics['by_domain'][d]['total'] for d in domains]
    
    bars = ax2.bar(range(len(domains)), domain_accs, color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Domain')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Performance by Domain')
    ax2.set_xticks(range(len(domains)))
    ax2.set_xticklabels([d.replace(' ', '\\n') for d in domains], fontsize=8)
    ax2.set_ylim(0, 1)
    
    # Add sample counts on bars
    for bar, total in zip(bars, domain_totals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}\\n(n={total})', ha='center', va='bottom', fontsize=8)
    
    # Difficulty comparison
    difficulties = list(metrics['by_difficulty'].keys())
    diff_accs = [metrics['by_difficulty'][d]['accuracy'] for d in difficulties]
    diff_totals = [metrics['by_difficulty'][d]['total'] for d in difficulties]
    
    bars = ax3.bar(difficulties, diff_accs, color='lightgreen', alpha=0.8)
    ax3.set_xlabel('Difficulty')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Performance by Difficulty')
    ax3.set_ylim(0, 1)
    
    for bar, total in zip(bars, diff_totals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}\\n(n={total})', ha='center', va='bottom')
    
    # Length analysis
    lengths = list(metrics['by_length'].keys())
    length_accs = [metrics['by_length'][l]['accuracy'] for l in lengths]
    length_totals = [metrics['by_length'][l]['total'] for l in lengths]
    
    bars = ax4.bar(lengths, length_accs, color='gold', alpha=0.8)
    ax4.set_xlabel('Context Length')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Performance by Context Length')
    ax4.set_ylim(0, 1)
    
    for bar, total in zip(bars, length_totals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}\\n(n={total})', ha='center', va='bottom')
    
    plt.tight_layout()
    summary_plot = os.path.join(plots_dir, "longbench_summary.png")
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Domain Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create detailed domain heatmap
    domain_data = []
    for domain in domains:
        domain_results = [r for r in results if r['domain'] == domain]
        
        # Calculate accuracy by difficulty within domain
        easy_results = [r for r in domain_results if r['difficulty'] == 'easy']
        hard_results = [r for r in domain_results if r['difficulty'] == 'hard']
        
        easy_acc = sum(r['judge'] for r in easy_results) / len(easy_results) if easy_results else 0
        hard_acc = sum(r['judge'] for r in hard_results) / len(hard_results) if hard_results else 0
        
        domain_data.append([easy_acc, hard_acc])
    
    # Create heatmap
    domain_matrix = np.array(domain_data)
    im = ax.imshow(domain_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Easy', 'Hard'])
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d.replace(' ', '\\n') for d in domains])
    ax.set_title('Accuracy Heatmap: Domain vs Difficulty', fontweight='bold')
    
    # Add text annotations
    for i in range(len(domains)):
        for j in range(2):
            text = ax.text(j, i, f'{domain_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    plt.tight_layout()
    heatmap_plot = os.path.join(plots_dir, "longbench_heatmap.png")
    plt.savefig(heatmap_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Domain accuracy distribution
    ax1.pie(domain_totals, labels=[d.split()[0] for d in domains], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Sample Distribution by Domain')
    
    # Performance comparison with benchmarks
    benchmark_data = {
        'TinyLlama (Demo)': overall_acc,
        'Expected 7B Model': 0.42,
        'Expected 13B Model': 0.48,
        'Human Expert': 0.537
    }
    
    models = list(benchmark_data.keys())
    accuracies = list(benchmark_data.values())
    colors = ['lightblue', 'orange', 'green', 'red']
    
    bars = ax2.bar(models, accuracies, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Performance Comparison')
    ax2.set_ylim(0, 0.6)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    comparison_plot = os.path.join(plots_dir, "longbench_comparison.png")
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created summary plot: {summary_plot}")
    print(f"✓ Created heatmap: {heatmap_plot}")
    print(f"✓ Created comparison plot: {comparison_plot}")
    
    return plots_dir

def print_results_summary(summary):
    """Print a formatted summary of results."""
    print("\n" + "="*60)
    print("LONGBENCH v2 EVALUATION RESULTS")
    print("="*60)
    
    metrics = summary['metrics']
    config = summary['config']
    
    print(f"Model: {config['model']}")
    print(f"Mode: {config['mode']}")
    print(f"Max Length: {config['max_length']} tokens")
    print(f"Total Samples: {summary['total_samples']}")
    print()
    
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"Correct: {metrics['overall']['correct']}/{metrics['overall']['total']}")
    print()
    
    print("Performance by Domain:")
    for domain, domain_metrics in metrics['by_domain'].items():
        print(f"  {domain}: {domain_metrics['accuracy']:.2%} ({domain_metrics['total']} samples)")
    print()
    
    print("Performance by Difficulty:")
    for difficulty, diff_metrics in metrics['by_difficulty'].items():
        print(f"  {difficulty.title()}: {diff_metrics['accuracy']:.2%} ({diff_metrics['total']} samples)")
    print()
    
    print("Performance by Context Length:")
    for length, length_metrics in metrics['by_length'].items():
        print(f"  {length}: {length_metrics['accuracy']:.2%} ({length_metrics['total']} samples)")

def main():
    """Main execution function."""
    print("LongBench v2 Demo Results Generator")
    print("="*40)
    
    # Create mock results
    summary, results = create_mock_longbench_results()
    
    # Create visualizations
    plots_dir = create_visualizations(summary, results)
    
    # Print summary
    print_results_summary(summary)
    
    print(f"\n📊 Visualizations saved in: {plots_dir}/")
    print("\nThis demonstrates the complete LongBench v2 evaluation workflow:")
    print("  ✓ Result generation and storage")
    print("  ✓ Comprehensive metrics calculation")
    print("  ✓ Multi-dimensional analysis (domain, difficulty, length)")
    print("  ✓ Professional visualization creation")
    print("  ✓ Performance comparison with benchmarks")
    
    print(f"\n📁 All files saved in: saves/longbench_demo_mock/")

if __name__ == "__main__":
    main()