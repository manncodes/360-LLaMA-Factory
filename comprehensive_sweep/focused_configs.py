#!/usr/bin/env python3
"""Generate focused comprehensive sweep with all RoPE types but fewer combinations."""

import sys
import yaml
import itertools
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from hyperparam_sweep.generate_configs import create_base_config, generate_config_id

# Focused parameter ranges for comprehensive testing
FOCUSED_ROPE_CONFIGS = {
    "linear": {
        "rope_scaling_type": ["linear"],
        "rope_scaling_factor": [2, 4, 8],  # Reduced from [2,4,8,16,32]
    },
    "dynamic": {
        "rope_scaling_type": ["dynamic"], 
        "rope_scaling_factor": [2, 4, 8],  # Reduced
    },
    "yarn": {
        "rope_scaling_type": ["yarn"],
        "rope_scaling_factor": [2, 4, 8],  # Reduced
        "yarn_alpha": [1.0, 2.0],  # Reduced from [0.5,1.0,2.0,4.0]
        "yarn_beta": [16.0, 32.0],  # Reduced from [8,16,32,64]
    },
    "llama3": {
        "rope_scaling_type": ["llama3"],
        "rope_scaling_factor": [2, 4, 8],  # Reduced
    },
}

CONTEXT_LENGTHS = {
    "short": 4096,    # 4K tokens
    "medium": 8192,   # 8K tokens  
    "long": 16384,    # 16K tokens
}

def generate_focused_configs():
    """Generate focused comprehensive sweep."""
    print("🎯 Generating Focused Comprehensive Sweep")
    print("=" * 50)
    
    output_dir = Path("comprehensive_sweep/focused")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "gpt2"
    generated_configs = []
    config_manifest = []
    
    # Generate baseline configs
    for ctx_key, ctx_len in CONTEXT_LENGTHS.items():
        base_config = create_base_config(model_name, ctx_len)
        config_id = generate_config_id(base_config)
        
        filename = f"baseline_{ctx_key}_{config_id}.yaml"
        filepath = output_dir / filename
        
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
    
    # Generate RoPE configs
    total_rope_configs = 0
    
    for rope_type, params in FOCUSED_ROPE_CONFIGS.items():
        print(f"\n📋 Generating {rope_type.upper()} configurations...")
        
        # Get parameter combinations
        param_names = list(params.keys())
        param_values = list(params.values())
        
        combinations = list(itertools.product(*param_values))
        print(f"   Parameter combinations: {len(combinations)}")
        
        for combination in combinations:
            rope_config = {}
            for name, value in zip(param_names, combination):
                rope_config[name] = value
            
            for ctx_key, ctx_len in CONTEXT_LENGTHS.items():
                # Skip extreme combinations that might not work
                scaling_factor = rope_config.get("rope_scaling_factor", 1)
                if ctx_len > 4096 and scaling_factor > 8:
                    continue  # Skip very large context + high scaling
                
                base_config = create_base_config(model_name, ctx_len)
                full_config = {**base_config, **rope_config}
                
                config_id = generate_config_id(full_config)
                
                filename = f"{rope_type}_{ctx_key}_sf{scaling_factor}_{config_id}.yaml"
                filepath = output_dir / filename
                
                with open(filepath, 'w') as f:
                    yaml.dump(full_config, f, default_flow_style=False)
                
                generated_configs.append(str(filepath))
                config_manifest.append({
                    "file": filename,
                    "type": rope_type,
                    "context": ctx_len,
                    "rope_config": rope_config,
                    "model": model_name,
                    "id": config_id
                })
                
                total_rope_configs += 1
        
        print(f"   ✅ Generated {len([c for c in config_manifest if c['type'] == rope_type])} {rope_type} configs")
    
    # Save manifest
    manifest = {
        "total_configs": len(config_manifest),
        "baseline_configs": len([c for c in config_manifest if c['type'] == 'baseline']),
        "rope_configs": total_rope_configs,
        "model": model_name,
        "context_lengths": CONTEXT_LENGTHS,
        "rope_types": list(FOCUSED_ROPE_CONFIGS.keys()),
        "configs": config_manifest
    }
    
    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"\n📊 Summary:")
    print(f"   Total configurations: {len(config_manifest)}")
    print(f"   Baseline configs: {len([c for c in config_manifest if c['type'] == 'baseline'])}")
    print(f"   RoPE configs: {total_rope_configs}")
    print(f"   RoPE types: {list(FOCUSED_ROPE_CONFIGS.keys())}")
    print(f"   Context lengths: {list(CONTEXT_LENGTHS.keys())}")
    
    # Estimate compute time
    estimate_minutes = len(config_manifest) * 5  # 5 min per config (faster with optimized eval)
    print(f"\n⏱️ Estimated time:")
    print(f"   Single GPU: {estimate_minutes/60:.1f} hours")  
    print(f"   4 GPUs: {estimate_minutes/60/4:.1f} hours")
    print(f"   8 GPUs: {estimate_minutes/60/8:.1f} hours")
    
    print(f"\n💾 Saved to: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    
    return generated_configs

if __name__ == "__main__":
    configs = generate_focused_configs()