#!/usr/bin/env python3
"""Generate configuration files for hyperparameter sweep of RoPE scaling methods."""

import os
import yaml
import itertools
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configuration space
ROPE_CONFIGS = {
    "linear": {
        "rope_scaling_type": ["linear"],
        "rope_scaling_factor": [2, 4, 8, 16, 32],
    },
    "dynamic": {
        "rope_scaling_type": ["dynamic"],
        "rope_scaling_factor": [2, 4, 8, 16, 32],
    },
    "yarn": {
        "rope_scaling_type": ["yarn"],
        "rope_scaling_factor": [2, 4, 8, 16],
        "yarn_alpha": [0.5, 1.0, 2.0, 4.0],
        "yarn_beta": [8.0, 16.0, 32.0, 64.0],
    },
    "longrope": {
        "rope_scaling_type": ["longrope"],
        "rope_scaling_factor": [2, 4, 8, 16],
        "longrope_short_factor": [1.0, 1.5, 2.0],
        "longrope_long_factor": [1.0, 2.0, 4.0],
    },
    "llama3": {
        "rope_scaling_type": ["llama3"],
        "rope_scaling_factor": [2, 4, 8, 16],
    },
}

# Context lengths to test (in tokens)
CONTEXT_LENGTHS = {
    "baseline": 2048,
    "short": 4096,
    "medium": 8192,
    "long": 16384,
    "very_long": 32768,
    "extreme": 65536,
    "ultra": 131072,
}

# Base models for experiments
BASE_MODELS = {
    "fast": "gpt2",  # For quick iteration
    "small": "pythia-1.4b",  # Small but capable
    "medium": "meta-llama/Llama-2-7b-hf",  # Standard size
    "large": "meta-llama/Meta-Llama-3-8B",  # Latest architecture
}

# Evaluation datasets
EVAL_DATASETS = {
    "perplexity": ["wikitext", "c4", "pg19"],
    "long_context": ["scrolls", "longbench", "needle_haystack"],
}


def generate_config_id(config: Dict[str, Any]) -> str:
    """Generate unique ID for configuration."""
    config_str = yaml.dump(config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def create_base_config(model_name: str, context_length: int) -> Dict[str, Any]:
    """Create base configuration for evaluation."""
    return {
        # Model configuration
        "model_name_or_path": model_name,
        "trust_remote_code": True,
        
        # Data configuration
        "cutoff_len": context_length,
        "max_samples": 1000,  # For faster evaluation
        
        # Inference configuration
        "infer_dtype": "bfloat16",
        "use_cache": True,
        "low_cpu_mem_usage": True,
        
        # Evaluation settings
        "batch_size": 1,  # For accurate perplexity
        "seed": 42,
        
        # Task configuration
        "task": "perplexity",
        "save_dir": "results/",
    }


def generate_rope_configs(rope_type: str, params: Dict) -> List[Dict[str, Any]]:
    """Generate all combinations of RoPE parameters."""
    configs = []
    
    # Get all parameter names and values
    param_names = list(params.keys())
    param_values = list(params.values())
    
    # Generate all combinations
    for combination in itertools.product(*param_values):
        config = {}
        for name, value in zip(param_names, combination):
            config[name] = value
        configs.append(config)
    
    return configs


def create_sweep_config(
    base_config: Dict[str, Any],
    rope_config: Dict[str, Any],
    context_length: int,
    config_id: str
) -> Dict[str, Any]:
    """Combine base and rope configs into complete sweep configuration."""
    config = base_config.copy()
    config.update(rope_config)
    config["cutoff_len"] = context_length
    config["experiment_id"] = config_id
    config["timestamp"] = datetime.now().isoformat()
    
    # Adjust save directory
    rope_type = rope_config.get("rope_scaling_type", "baseline")
    model_name = base_config["model_name_or_path"].split("/")[-1]
    config["save_dir"] = f"results/{model_name}/{rope_type}/{context_length}/{config_id}"
    
    return config


def generate_all_configs(
    model_key: str = "fast",
    rope_types: List[str] = None,
    context_keys: List[str] = None,
    output_dir: str = "configs"
) -> List[str]:
    """Generate all configuration files for the sweep."""
    
    if rope_types is None:
        rope_types = list(ROPE_CONFIGS.keys())
    
    if context_keys is None:
        context_keys = ["short", "medium", "long"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model and context lengths
    model_name = BASE_MODELS[model_key]
    context_lengths = {k: CONTEXT_LENGTHS[k] for k in context_keys}
    
    generated_configs = []
    config_manifest = []
    
    # Generate baseline configuration (no rope scaling)
    for ctx_key, ctx_len in context_lengths.items():
        base_config = create_base_config(model_name, ctx_len)
        config_id = generate_config_id(base_config)
        
        # Save baseline config
        filename = f"baseline_{ctx_key}_{config_id}.yaml"
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        generated_configs.append(str(filepath))
        config_manifest.append({
            "file": filename,
            "type": "baseline",
            "context": ctx_len,
            "model": model_name,
            "id": config_id
        })
    
    # Generate RoPE scaling configurations
    for rope_type in rope_types:
        rope_param_configs = generate_rope_configs(rope_type, ROPE_CONFIGS[rope_type])
        
        for rope_config in rope_param_configs:
            for ctx_key, ctx_len in context_lengths.items():
                # Skip invalid combinations
                scaling_factor = rope_config.get("rope_scaling_factor", 1)
                if ctx_len > 2048 * scaling_factor * 2:
                    # Skip if context is too long for scaling factor
                    continue
                
                base_config = create_base_config(model_name, ctx_len)
                config_id = generate_config_id({**base_config, **rope_config})
                
                full_config = create_sweep_config(
                    base_config, rope_config, ctx_len, config_id
                )
                
                # Generate filename
                filename = f"{rope_type}_{ctx_key}_sf{scaling_factor}_{config_id}.yaml"
                filepath = output_path / filename
                
                # Save configuration
                with open(filepath, 'w') as f:
                    yaml.dump(full_config, f, default_flow_style=False)
                
                generated_configs.append(str(filepath))
                config_manifest.append({
                    "file": filename,
                    "type": rope_type,
                    "context": ctx_len,
                    "model": model_name,
                    "rope_config": rope_config,
                    "id": config_id
                })
    
    # Save manifest
    manifest_path = output_path / "manifest.yaml"
    with open(manifest_path, 'w') as f:
        yaml.dump({
            "total_configs": len(config_manifest),
            "model": model_name,
            "rope_types": rope_types,
            "context_lengths": context_lengths,
            "timestamp": datetime.now().isoformat(),
            "configs": config_manifest
        }, f, default_flow_style=False)
    
    print(f"Generated {len(generated_configs)} configurations")
    print(f"Saved to {output_dir}/")
    print(f"Manifest: {manifest_path}")
    
    return generated_configs


def estimate_compute_time(num_configs: int, avg_time_per_config: float = 30) -> Dict[str, float]:
    """Estimate total compute time for sweep."""
    total_minutes = num_configs * avg_time_per_config
    return {
        "configs": num_configs,
        "minutes": total_minutes,
        "hours": total_minutes / 60,
        "gpu_hours": total_minutes / 60,
        "days_single_gpu": total_minutes / (60 * 24),
        "days_8_gpus": total_minutes / (60 * 24 * 8),
    }


def main():
    """Main function to generate sweep configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RoPE hyperparameter sweep configs")
    parser.add_argument("--model", default="fast", choices=list(BASE_MODELS.keys()),
                       help="Base model to use")
    parser.add_argument("--rope-types", nargs="+", default=None,
                       help="RoPE types to test (default: all)")
    parser.add_argument("--contexts", nargs="+", default=["short", "medium", "long"],
                       choices=list(CONTEXT_LENGTHS.keys()),
                       help="Context lengths to test")
    parser.add_argument("--output", default="configs", help="Output directory")
    parser.add_argument("--estimate", action="store_true", help="Estimate compute time")
    
    args = parser.parse_args()
    
    # Generate configurations
    configs = generate_all_configs(
        model_key=args.model,
        rope_types=args.rope_types,
        context_keys=args.contexts,
        output_dir=args.output
    )
    
    # Estimate compute time
    if args.estimate:
        estimates = estimate_compute_time(len(configs))
        print("\nCompute Time Estimates:")
        print(f"  Total configs: {estimates['configs']}")
        print(f"  Single GPU: {estimates['days_single_gpu']:.1f} days")
        print(f"  8 GPUs parallel: {estimates['days_8_gpus']:.1f} days")
        print(f"  GPU hours: {estimates['gpu_hours']:.1f}")


if __name__ == "__main__":
    main()