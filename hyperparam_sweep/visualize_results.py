#!/usr/bin/env python3
"""Visualize hyperparameter sweep results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import argparse


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all results into a pandas DataFrame."""
    results_path = Path(results_dir)
    result_files = list(results_path.glob("eval_*.json"))
    
    data = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
            
        # Extract relevant data
        config = result["config"]
        metrics = result["metrics"]
        
        row = {
            "config_file": result["config_path"],
            "rope_type": config.get("rope_scaling_type", "baseline"),
            "rope_factor": config.get("rope_scaling_factor", 1.0),
            "context_length": config.get("cutoff_len", 2048),
            "yarn_alpha": config.get("yarn_alpha", None),
            "yarn_beta": config.get("yarn_beta", None),
            "perplexity": metrics["perplexity"],
            "long_ppl": metrics["long_ppl"],
            "loss": metrics["loss"],
            "memory_gb": metrics["memory_gb"],
            "throughput_tps": metrics["throughput_tps"],
            "tokens_processed": metrics["tokens_processed"],
        }
        data.append(row)
    
    return pd.DataFrame(data)


def plot_ppl_vs_context(df: pd.DataFrame, save_path: str = None):
    """Plot perplexity vs context length for different RoPE types."""
    plt.figure(figsize=(12, 8))
    
    # Create separate plots for PPL and LongPPL
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Standard perplexity
    for rope_type in df['rope_type'].unique():
        type_df = df[df['rope_type'] == rope_type]
        
        # Group by context length and get mean/std
        grouped = type_df.groupby('context_length').agg({
            'perplexity': ['mean', 'std'],
            'long_ppl': ['mean', 'std']
        }).reset_index()
        
        ax1.errorbar(
            grouped['context_length'],
            grouped[('perplexity', 'mean')],
            yerr=grouped[('perplexity', 'std')],
            marker='o', label=rope_type, capsize=5
        )
        
        ax2.errorbar(
            grouped['context_length'],
            grouped[('long_ppl', 'mean')],
            yerr=grouped[('long_ppl', 'std')],
            marker='s', label=rope_type, capsize=5
        )
    
    # Format plots
    ax1.set_xlabel('Context Length (tokens)')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Standard Perplexity vs Context Length')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Context Length (tokens)')
    ax2.set_ylabel('Long-Context Perplexity')
    ax2.set_title('Long-Context Perplexity vs Context Length')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_scaling_factor_analysis(df: pd.DataFrame, save_path: str = None):
    """Plot performance vs scaling factor for different RoPE types."""
    # Filter out baseline and focus on scaling methods
    df_scaled = df[df['rope_type'] != 'baseline'].copy()
    
    if df_scaled.empty:
        print("No scaling configurations found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: PPL vs Scaling Factor
    for rope_type in df_scaled['rope_type'].unique():
        type_df = df_scaled[df_scaled['rope_type'] == rope_type]
        axes[0,0].scatter(type_df['rope_factor'], type_df['perplexity'], 
                         label=rope_type, alpha=0.7)
    
    axes[0,0].set_xlabel('RoPE Scaling Factor')
    axes[0,0].set_ylabel('Perplexity')
    axes[0,0].set_title('Perplexity vs Scaling Factor')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Long PPL vs Scaling Factor
    for rope_type in df_scaled['rope_type'].unique():
        type_df = df_scaled[df_scaled['rope_type'] == rope_type]
        axes[0,1].scatter(type_df['rope_factor'], type_df['long_ppl'], 
                         label=rope_type, alpha=0.7)
    
    axes[0,1].set_xlabel('RoPE Scaling Factor')
    axes[0,1].set_ylabel('Long-Context Perplexity')
    axes[0,1].set_title('Long-Context Perplexity vs Scaling Factor')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Memory vs Context Length
    for rope_type in df_scaled['rope_type'].unique():
        type_df = df_scaled[df_scaled['rope_type'] == rope_type]
        axes[1,0].scatter(type_df['context_length'], type_df['memory_gb'], 
                         label=rope_type, alpha=0.7)
    
    axes[1,0].set_xlabel('Context Length')
    axes[1,0].set_ylabel('Memory Usage (GB)')
    axes[1,0].set_title('Memory Usage vs Context Length')
    axes[1,0].set_xscale('log', base=2)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Throughput vs Context Length
    for rope_type in df_scaled['rope_type'].unique():
        type_df = df_scaled[df_scaled['rope_type'] == rope_type]
        axes[1,1].scatter(type_df['context_length'], type_df['throughput_tps'], 
                         label=rope_type, alpha=0.7)
    
    axes[1,1].set_xlabel('Context Length')
    axes[1,1].set_ylabel('Throughput (tokens/sec)')
    axes[1,1].set_title('Throughput vs Context Length')
    axes[1,1].set_xscale('log', base=2)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_yarn_hyperparams(df: pd.DataFrame, save_path: str = None):
    """Plot YARN-specific hyperparameter analysis."""
    yarn_df = df[df['rope_type'] == 'yarn'].copy()
    
    if yarn_df.empty:
        print("No YARN configurations found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PPL vs Alpha (grouped by Beta)
    yarn_df['yarn_beta_str'] = yarn_df['yarn_beta'].astype(str)
    for beta in yarn_df['yarn_beta'].unique():
        beta_df = yarn_df[yarn_df['yarn_beta'] == beta]
        axes[0].plot(beta_df['yarn_alpha'], beta_df['perplexity'], 
                    'o-', label=f'β={beta}', markersize=6)
    
    axes[0].set_xlabel('YARN Alpha')
    axes[0].set_ylabel('Perplexity')
    axes[0].set_title('YARN: Perplexity vs Alpha (by Beta)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PPL vs Beta (grouped by Alpha)
    for alpha in yarn_df['yarn_alpha'].unique():
        alpha_df = yarn_df[yarn_df['yarn_alpha'] == alpha]
        axes[1].plot(alpha_df['yarn_beta'], alpha_df['perplexity'], 
                    's-', label=f'α={alpha}', markersize=6)
    
    axes[1].set_xlabel('YARN Beta')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('YARN: Perplexity vs Beta (by Alpha)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_pareto_plot(df: pd.DataFrame, save_path: str = None):
    """Create Pareto frontier plot of performance vs compute cost."""
    plt.figure(figsize=(12, 8))
    
    # Use memory as proxy for compute cost
    for rope_type in df['rope_type'].unique():
        type_df = df[df['rope_type'] == rope_type]
        plt.scatter(type_df['memory_gb'], type_df['perplexity'], 
                   label=rope_type, alpha=0.7, s=60)
    
    plt.xlabel('Memory Usage (GB) - Compute Cost Proxy')
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Pareto Frontier: Model Quality vs Compute Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate best configurations
    for rope_type in df['rope_type'].unique():
        type_df = df[df['rope_type'] == rope_type]
        if not type_df.empty:
            best_ppl_idx = type_df['perplexity'].idxmin()
            best_config = type_df.loc[best_ppl_idx]
            plt.annotate(
                f"{rope_type}\nPPL={best_config['perplexity']:.1f}",
                (best_config['memory_gb'], best_config['perplexity']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, alpha=0.7
            )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(df: pd.DataFrame, output_dir: str = "analysis"):
    """Generate comprehensive analysis report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    summary_stats = []
    for rope_type in df['rope_type'].unique():
        type_df = df[df['rope_type'] == rope_type]
        stats = {
            "RoPE Type": rope_type,
            "Count": len(type_df),
            "PPL Mean": type_df['perplexity'].mean(),
            "PPL Std": type_df['perplexity'].std(),
            "PPL Min": type_df['perplexity'].min(),
            "PPL Max": type_df['perplexity'].max(),
            "LongPPL Mean": type_df['long_ppl'].mean(),
            "LongPPL Min": type_df['long_ppl'].min(),
            "Memory Mean (GB)": type_df['memory_gb'].mean(),
            "Throughput Mean (TPS)": type_df['throughput_tps'].mean(),
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_path / "summary_statistics.csv", index=False)
    
    # Generate visualizations
    plot_ppl_vs_context(df, save_path=str(output_path / "ppl_vs_context.png"))
    plot_scaling_factor_analysis(df, save_path=str(output_path / "scaling_analysis.png"))
    plot_yarn_hyperparams(df, save_path=str(output_path / "yarn_analysis.png"))
    create_pareto_plot(df, save_path=str(output_path / "pareto_frontier.png"))
    
    print(f"\nAnalysis complete! Results saved to: {output_path}")
    print("\nGenerated files:")
    print("  - summary_statistics.csv")
    print("  - ppl_vs_context.png")
    print("  - scaling_analysis.png") 
    print("  - yarn_analysis.png")
    print("  - pareto_frontier.png")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize RoPE sweep results")
    parser.add_argument("--results-dir", default="results",
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="analysis",
                       help="Directory to save analysis outputs")
    parser.add_argument("--show-plots", action="store_true",
                       help="Display plots interactively")
    
    args = parser.parse_args()
    
    # Load results
    try:
        df = load_results(args.results_dir)
        print(f"Loaded {len(df)} evaluation results")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Generate analysis
    if not args.show_plots:
        # Suppress matplotlib GUI
        import matplotlib
        matplotlib.use('Agg')
    
    generate_report(df, args.output_dir)


if __name__ == "__main__":
    main()