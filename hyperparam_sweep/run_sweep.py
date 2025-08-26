#!/usr/bin/env python3
"""Run hyperparameter sweep with parallel execution."""

import os
import sys
import yaml
import json
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def run_single_evaluation(config_file: str, gpu_id: int = 0) -> Dict:
    """Run evaluation for a single configuration."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        sys.executable,
        "hyperparam_sweep/evaluate_perplexity.py",
        config_file,
        "--dataset", "wikitext"
    ]
    
    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Parse output to get result file path
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "Saved to:" in line:
                    result_file = line.split("Saved to:")[-1].strip()
                    with open(result_file, 'r') as f:
                        return json.load(f)
            
            return {
                "config_file": config_file,
                "status": "success",
                "stdout": result.stdout
            }
        else:
            return {
                "config_file": config_file,
                "status": "failed",
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            "config_file": config_file,
            "status": "timeout"
        }
    except Exception as e:
        return {
            "config_file": config_file,
            "status": "error",
            "error": str(e)
        }


def distribute_configs_to_gpus(config_files: List[str], num_gpus: int) -> List[List[str]]:
    """Distribute configuration files across GPUs."""
    gpu_configs = [[] for _ in range(num_gpus)]
    
    for i, config in enumerate(config_files):
        gpu_idx = i % num_gpus
        gpu_configs[gpu_idx].append(config)
    
    return gpu_configs


def run_parallel_sweep(
    config_dir: str,
    num_gpus: int = 1,
    max_workers: int = None,
    resume: bool = False
) -> List[Dict]:
    """Run sweep in parallel across multiple GPUs."""
    
    config_path = Path(config_dir)
    config_files = list(config_path.glob("*.yaml"))
    
    # Filter out manifest and completed configs
    config_files = [f for f in config_files if f.name != "manifest.yaml"]
    
    if resume:
        # Check which configs have already been evaluated
        results_dir = Path("results")
        completed = set()
        if results_dir.exists():
            for result_file in results_dir.glob("eval_*.json"):
                config_name = result_file.stem.replace("eval_", "")
                completed.add(config_name)
        
        config_files = [f for f in config_files if f.stem not in completed]
        print(f"Resuming: {len(config_files)} configs remaining")
    
    if not config_files:
        print("No configurations to evaluate")
        return []
    
    print(f"Running sweep with {len(config_files)} configurations on {num_gpus} GPU(s)")
    
    # Distribute configs across GPUs
    gpu_configs = distribute_configs_to_gpus([str(f) for f in config_files], num_gpus)
    
    # Prepare jobs
    jobs = []
    for gpu_id, configs in enumerate(gpu_configs):
        for config in configs:
            jobs.append((config, gpu_id))
    
    # Run evaluations in parallel
    results = []
    
    if max_workers is None:
        max_workers = num_gpus
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_single_evaluation, config, gpu_id): (config, gpu_id)
            for config, gpu_id in jobs
        }
        
        # Process results as they complete
        with tqdm(total=len(jobs), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                config, gpu_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress
                    if "metrics" in result:
                        ppl = result["metrics"]["perplexity"]
                        pbar.set_postfix({"last_ppl": f"{ppl:.2f}"})
                        
                except Exception as e:
                    print(f"Error processing {config}: {e}")
                    results.append({
                        "config_file": config,
                        "status": "error",
                        "error": str(e)
                    })
                
                pbar.update(1)
    
    return results


def analyze_results(results_dir: str = "results") -> Dict:
    """Analyze sweep results and find best configurations."""
    
    results_path = Path(results_dir)
    result_files = list(results_path.glob("eval_*.json"))
    
    if not result_files:
        print("No results found")
        return {}
    
    # Load all results
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            all_results.append(json.load(f))
    
    # Group by rope type and context length
    grouped = {}
    for result in all_results:
        config = result["config"]
        rope_type = config.get("rope_scaling_type", "baseline")
        context_len = config.get("cutoff_len", 2048)
        
        key = f"{rope_type}_{context_len}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    # Find best configuration for each group
    best_configs = {}
    for key, group_results in grouped.items():
        # Sort by perplexity (lower is better)
        sorted_results = sorted(
            group_results,
            key=lambda x: x["metrics"]["perplexity"]
        )
        best_configs[key] = sorted_results[0]
    
    # Create summary report
    summary = {
        "total_evaluated": len(all_results),
        "best_configs": best_configs,
        "statistics": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Calculate statistics per rope type
    for rope_type in set(r["config"].get("rope_scaling_type", "baseline") for r in all_results):
        rope_results = [r for r in all_results 
                       if r["config"].get("rope_scaling_type", "baseline") == rope_type]
        
        if rope_results:
            ppls = [r["metrics"]["perplexity"] for r in rope_results]
            long_ppls = [r["metrics"]["long_ppl"] for r in rope_results]
            
            summary["statistics"][rope_type] = {
                "count": len(rope_results),
                "ppl_mean": np.mean(ppls),
                "ppl_std": np.std(ppls),
                "ppl_min": min(ppls),
                "ppl_max": max(ppls),
                "long_ppl_mean": np.mean(long_ppls),
                "long_ppl_std": np.std(long_ppls),
                "long_ppl_min": min(long_ppls),
                "long_ppl_max": max(long_ppls),
            }
    
    # Save summary
    summary_file = results_path / "sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSweep Summary:")
    print(f"Total configurations evaluated: {summary['total_evaluated']}")
    print(f"\nBest configurations per setting:")
    
    for key, config in summary["best_configs"].items():
        rope_type, context = key.rsplit('_', 1)
        ppl = config["metrics"]["perplexity"]
        long_ppl = config["metrics"]["long_ppl"]
        print(f"  {rope_type} @ {context} tokens: PPL={ppl:.2f}, LongPPL={long_ppl:.2f}")
    
    print(f"\nFull summary saved to: {summary_file}")
    
    return summary


def main():
    """Main function for running sweep."""
    parser = argparse.ArgumentParser(description="Run RoPE hyperparameter sweep")
    parser.add_argument("--generate", action="store_true",
                       help="Generate configurations")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze results")
    parser.add_argument("--config-dir", default="configs",
                       help="Directory with config files")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Max parallel workers")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    parser.add_argument("--model", default="fast",
                       help="Model to use for generation")
    parser.add_argument("--rope-types", nargs="+", default=None,
                       help="RoPE types to test")
    parser.add_argument("--contexts", nargs="+", 
                       default=["short", "medium"],
                       help="Context lengths to test")
    
    args = parser.parse_args()
    
    # Generate configurations
    if args.generate:
        print("Generating configurations...")
        from generate_configs import generate_all_configs
        
        configs = generate_all_configs(
            model_key=args.model,
            rope_types=args.rope_types,
            context_keys=args.contexts,
            output_dir=args.config_dir
        )
        print(f"Generated {len(configs)} configurations")
    
    # Run evaluation
    if args.evaluate:
        print("Running evaluation sweep...")
        results = run_parallel_sweep(
            args.config_dir,
            num_gpus=args.num_gpus,
            max_workers=args.max_workers,
            resume=args.resume
        )
        print(f"Completed {len(results)} evaluations")
    
    # Analyze results
    if args.analyze:
        print("Analyzing results...")
        import numpy as np  # Import here to avoid dependency if not analyzing
        summary = analyze_results()
    
    if not any([args.generate, args.evaluate, args.analyze]):
        print("Please specify at least one action: --generate, --evaluate, or --analyze")


if __name__ == "__main__":
    main()