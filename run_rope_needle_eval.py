#!/usr/bin/env python3
"""
Minimal needle-in-haystack evaluation with RoPE extension support.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datetime import datetime

from src.llamafactory.rope_eval import (
    NeedleInHaystackEvaluator,
    NeedleConfig,
    RoPEConfig,
    RoPEManager,
    ModelLoader
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_evaluation(
    model_name: str,
    rope_technique: str,
    context_length: int,
    needle_config: NeedleConfig,
    rope_manager: RoPEManager,
    device: str = "cuda"
) -> Dict:
    """Run a single evaluation with specific RoPE configuration."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {rope_technique} @ {context_length} tokens")
    print(f"{'='*60}")
    
    # Create RoPE configuration
    rope_config = None
    if rope_technique != "baseline":
        rope_config = rope_manager.create_config(rope_technique, context_length)
        print(f"RoPE Config: {rope_config.to_dict() if rope_config else 'None'}")
    
    # Load model with RoPE configuration
    print(f"Loading model: {model_name}")
    model, tokenizer = ModelLoader.load_model_and_tokenizer(
        model_name,
        rope_config=rope_config,
        device=device,
        dtype=torch.float16
    )
    
    # Update needle config for this specific context length
    needle_config.context_lengths = [context_length]
    
    # Create evaluator and run
    evaluator = NeedleInHaystackEvaluator(needle_config)
    results = evaluator.run_evaluation(model, tokenizer)
    
    # Add metadata
    results["rope_technique"] = rope_technique
    results["context_length"] = context_length
    results["model"] = model_name
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    return results


def run_rope_analysis(config: Dict):
    """Run complete RoPE analysis based on configuration."""
    
    # Extract configuration
    model_name = config["model_name"]
    rope_techniques = config.get("rope_techniques", ["baseline", "linear", "dynamic"])
    context_lengths = config.get("context_lengths", [2048, 4096, 8192])
    base_context = config.get("base_context_length", 4096)
    
    # Create needle configuration
    needle_config = NeedleConfig(
        needle_text=config.get("needle_text", "The secret key is: BENCHMARK_SUCCESS_42"),
        retrieval_question=config.get("retrieval_question", "What is the secret key?"),
        depth_percents=config.get("depth_percents", [0.1, 0.3, 0.5, 0.7, 0.9]),
        samples_per_needle=config.get("samples_per_needle", 1),
        haystack_source=config.get("haystack_source", "paulgraham"),
        save_results=False  # We'll save aggregated results
    )
    
    # Create RoPE manager
    rope_manager = RoPEManager(base_context_length=base_context)
    
    # Get model info
    print(f"Analyzing model: {model_name}")
    model_info = ModelLoader.get_model_info(model_name)
    print(f"Model info: {json.dumps(model_info, indent=2)}")
    
    # Run evaluations
    all_results = []
    for technique in rope_techniques:
        for context_length in context_lengths:
            # Skip if context length exceeds technique limits
            if technique == "baseline" and context_length > base_context:
                print(f"Skipping {technique} @ {context_length} (exceeds base context)")
                continue
                
            try:
                results = run_single_evaluation(
                    model_name,
                    technique,
                    context_length,
                    needle_config,
                    rope_manager,
                    device=config.get("device", "cuda")
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"Error evaluating {technique} @ {context_length}: {e}")
                continue
    
    # Save aggregated results
    save_results(all_results, config.get("output_dir", "./rope_results"))
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def save_results(results: List[Dict], output_dir: str):
    """Save evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    with open(output_path / f"results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    summary = create_summary(results)
    with open(output_path / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def create_summary(results: List[Dict]) -> Dict:
    """Create summary of results."""
    summary = {
        "total_evaluations": len(results),
        "by_technique": {},
        "by_context_length": {},
        "best_configurations": []
    }
    
    # Aggregate by technique
    techniques = set(r["rope_technique"] for r in results)
    for technique in techniques:
        technique_results = [r for r in results if r["rope_technique"] == technique]
        summary["by_technique"][technique] = {
            "avg_accuracy": sum(r["overall_accuracy"] for r in technique_results) / len(technique_results),
            "evaluations": len(technique_results)
        }
    
    # Aggregate by context length
    context_lengths = set(r["context_length"] for r in results)
    for length in context_lengths:
        length_results = [r for r in results if r["context_length"] == length]
        
        # Find best technique for this context length
        best_result = max(length_results, key=lambda x: x["overall_accuracy"])
        
        summary["by_context_length"][length] = {
            "best_technique": best_result["rope_technique"],
            "best_accuracy": best_result["overall_accuracy"],
            "all_techniques": {
                r["rope_technique"]: r["overall_accuracy"] 
                for r in length_results
            }
        }
        
        summary["best_configurations"].append({
            "context_length": length,
            "technique": best_result["rope_technique"],
            "accuracy": best_result["overall_accuracy"]
        })
    
    return summary


def print_summary(results: List[Dict]):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    summary = create_summary(results)
    
    print("\nBy RoPE Technique:")
    for technique, stats in summary["by_technique"].items():
        print(f"  {technique}: {stats['avg_accuracy']:.2%} avg accuracy")
    
    print("\nBest Configuration per Context Length:")
    for config in summary["best_configurations"]:
        print(f"  {config['context_length']} tokens: {config['technique']} ({config['accuracy']:.2%})")


def main():
    parser = argparse.ArgumentParser(description="Run needle-in-haystack evaluation with RoPE extensions")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rope_eval_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (overrides config)"
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        choices=["baseline", "linear", "dynamic", "yarn", "longrope", "llama3"],
        help="RoPE techniques to evaluate"
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        help="Context lengths to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rope_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "rope_techniques": ["baseline", "linear", "dynamic"],
            "context_lengths": [2048, 4096],
            "base_context_length": 2048,
            "depth_percents": [0.1, 0.5, 0.9],
            "samples_per_needle": 1,
            "haystack_source": "paulgraham"
        }
    
    # Override with command line arguments
    if args.model:
        config["model_name"] = args.model
    if args.techniques:
        config["rope_techniques"] = args.techniques
    if args.contexts:
        config["context_lengths"] = args.contexts
    config["device"] = args.device
    config["output_dir"] = args.output_dir
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Run analysis
    run_rope_analysis(config)


if __name__ == "__main__":
    main()