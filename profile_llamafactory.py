#!/usr/bin/env python3
"""
PyTorch Profiler for LlamaFactory Training
Profiles training iterations without checkpoint overhead.
Generates Chrome traces for performance analysis.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Optional

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.model import load_model, load_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from transformers import AdamW
from torch.utils.data import DataLoader


def profile_training(
    config_file: str,
    output_dir: str = "profiling_traces",
    profile_steps: int = 10,
    wait_steps: int = 2,
    warmup_steps: int = 1,
    active_steps: int = 3
) -> None:
    """
    Profile LlamaFactory training iterations.
    
    Args:
        config_file: Path to YAML configuration file
        output_dir: Directory to save Chrome traces
        profile_steps: Total steps to profile
        wait_steps: Steps to skip before profiling
        warmup_steps: Warmup steps for profiler
        active_steps: Steps to actively profile
    """
    print("=" * 60)
    print("LlamaFactory Training Profiler")
    print("=" * 60)
    print(f"Config: {config_file}")
    print(f"Output: {output_dir}")
    print(f"Profile schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}")
    print(f"Total steps: {profile_steps}")
    print("-" * 60)
    
    # Parse training arguments
    original_argv = sys.argv
    sys.argv = ["profile_llamafactory.py", config_file]
    
    try:
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(None)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv
    
    # Override settings for profiling
    training_args.max_steps = profile_steps
    training_args.save_steps = profile_steps + 100  # Never save
    training_args.logging_steps = profile_steps + 100  # Never log
    training_args.eval_steps = profile_steps + 100  # Never eval
    training_args.output_dir = "temp_profiling"
    training_args.overwrite_output_dir = True
    
    print("Loading model components...")
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Load dataset
    print("Loading dataset...")
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]
    
    # Load model
    print("Loading model...")
    model = load_model(tokenizer, model_args, finetuning_args, True, full_determinism=training_args.full_determinism)
    model.train()
    
    # Data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        require_position_ids=model_args.sequence_parallel_size > 1,
        **tokenizer_module,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    # Simple optimizer
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    
    # Setup profiler
    os.makedirs(output_dir, exist_ok=True)
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False  # Reduce overhead
    )
    
    print("-" * 60)
    print("Starting profiled training...")
    print("-" * 60)
    
    # Training loop with profiling
    step = 0
    total_loss = 0.0
    
    with profiler:
        for batch in dataloader:
            if step >= profile_steps:
                break
            
            # Move batch to device
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Profiler step
            profiler.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Progress update
            if (step + 1) % 5 == 0 or step == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step [{step+1}/{profile_steps}] - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}")
            
            step += 1
    
    print("-" * 60)
    print("Profiling completed!")
    print("-" * 60)
    print(f"Total steps profiled: {step}")
    print(f"Average loss: {total_loss/step:.4f}")
    print(f"Traces saved to: {output_dir}")
    print("\nTo view traces:")
    print("1. Open Chrome and navigate to: chrome://tracing")
    print("2. Click 'Load' and select a .json file from the traces directory")
    print("3. Use WASD keys to navigate (W/S: zoom, A/D: scroll)")
    print("=" * 60)
    
    # Cleanup
    import shutil
    if os.path.exists("temp_profiling"):
        shutil.rmtree("temp_profiling")


def main():
    parser = argparse.ArgumentParser(
        description="Profile LlamaFactory training for performance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Profile with default settings:
    python profile_llamafactory.py config.yaml
    
  Profile 20 steps with custom output:
    python profile_llamafactory.py config.yaml --steps 20 --output my_traces
    
  Profile with custom schedule:
    python profile_llamafactory.py config.yaml --wait 5 --warmup 2 --active 5
        """
    )
    
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("-o", "--output", default="profiling_traces", help="Output directory for traces")
    parser.add_argument("-s", "--steps", type=int, default=10, help="Total steps to profile")
    parser.add_argument("--wait", type=int, default=2, help="Steps to wait before profiling")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup steps for profiler")
    parser.add_argument("--active", type=int, default=3, help="Steps to actively profile")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    # Validate profiler schedule
    if args.wait < 0 or args.warmup < 0 or args.active <= 0:
        print("Error: Invalid profiler schedule (wait>=0, warmup>=0, active>0)")
        sys.exit(1)
    
    if args.wait + args.warmup + args.active > args.steps:
        print(f"Warning: Profiler schedule ({args.wait}+{args.warmup}+{args.active}) exceeds total steps ({args.steps})")
        print("Adjusting total steps to match schedule...")
        args.steps = args.wait + args.warmup + args.active
    
    profile_training(
        config_file=args.config,
        output_dir=args.output,
        profile_steps=args.steps,
        wait_steps=args.wait,
        warmup_steps=args.warmup,
        active_steps=args.active
    )


if __name__ == "__main__":
    main()