#!/usr/bin/env python3
"""Progressive continual pretraining with advanced RoPE scaling."""

import subprocess
import argparse
import time
from pathlib import Path
import yaml
import json

def run_stage(config_file, dry_run=False):
    """Run a single training stage."""
    print(f"Running stage: {config_file}")
    
    if dry_run:
        print("DRY RUN - would execute:", ["llamafactory-cli", "train", config_file])
        return True
    
    start_time = time.time()
    result = subprocess.run(
        ["llamafactory-cli", "train", config_file],
        capture_output=True,
        text=True
    )
    
    duration = time.time() - start_time
    print(f"Stage completed in {duration:.1f} seconds")
    
    if result.returncode == 0:
        print(f"SUCCESS: {config_file}")
        return True
    else:
        print(f"FAILED: {config_file}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

def validate_config(config_file):
    """Validate configuration before training."""
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        required_keys = ["model_name_or_path", "rope_scaling_type", "cutoff_len"]
        missing = [key for key in required_keys if key not in config]
        
        if missing:
            print(f"WARNING: Missing required keys in {config_file}: {missing}")
            return False
        
        # Validate RoPE parameters
        rope_type = config.get("rope_scaling_type")
        if rope_type == "yarn":
            if not config.get("yarn_alpha") or not config.get("yarn_beta"):
                print(f"WARNING: YARN parameters incomplete in {config_file}")
        elif rope_type == "longrope":
            if not config.get("longrope_short_factor") or not config.get("longrope_long_factor"):
                print(f"WARNING: LongRoPE parameters incomplete in {config_file}")
        
        print(f"✅ {config_file} validation passed")
        return True
        
    except Exception as e:
        print(f"ERROR validating {config_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run progressive RoPE continual pretraining")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs without training")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1,2,3], help="Starting stage")
    parser.add_argument("--end-stage", type=int, default=3, choices=[1,2,3], help="Ending stage")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configurations")
    
    args = parser.parse_args()
    
    # Define stage configurations
    stages = [
        "stage1_yarn_8k.yaml",
        "stage2_yarn_16k.yaml", 
        "stage3_longrope_32k.yaml"
    ]
    
    print("Progressive RoPE Continual Pretraining")
    print("=" * 50)
    
    # Validate all configs first
    print("Validating configurations...")
    config_dir = Path(".")
    
    for i, stage_file in enumerate(stages, 1):
        config_path = config_dir / stage_file
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            return False
        
        if not validate_config(config_path):
            return False
    
    if args.validate_only:
        print("All configurations validated successfully!")
        return True
    
    # Run progressive training
    print("\nStarting progressive training...")
    
    for i in range(args.start_stage - 1, args.end_stage):
        stage_file = config_dir / stages[i]
        stage_num = i + 1
        
        print(f"\nStage {stage_num}: {stage_file.name}")
        print("-" * 30)
        
        if not run_stage(stage_file, args.dry_run):
            print(f"Training failed at Stage {stage_num}")
            return False
    
    print("\n" + "=" * 50)
    print("Progressive continual pretraining completed successfully!")
    print(f"Final model: continual_pt_stage{args.end_stage}_32k" if args.end_stage == 3 else f"continual_pt_stage{args.end_stage}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)