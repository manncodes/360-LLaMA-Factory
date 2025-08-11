#!/usr/bin/env python3
"""
Ultra-simple profiling script that profiles only core training iterations.
No trainer overhead, no checkpointing, just pure forward/backward passes.
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.model import load_model, load_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


def simple_profile_training(config_file: str, output_dir: str = "simple_traces", 
                           wait_steps: int = 2, warmup_steps: int = 1, active_steps: int = 5, 
                           total_steps: int = 15):
    """
    Profile only the core training iteration loop.
    
    Args:
        config_file: YAML config file path
        output_dir: Trace output directory
        wait_steps: Profiler wait steps
        warmup_steps: Profiler warmup steps  
        active_steps: Profiler active steps
        total_steps: Total training steps to run
    """
    print(f"Simple profiling with config: {config_file}")
    print(f"Schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}, total={total_steps}")
    
    # Load config - save original argv and restore after
    original_argv = sys.argv
    sys.argv = ["simple_profile.py", config_file]  # Set argv for parser
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(None)
    sys.argv = original_argv  # Restore original argv
    
    # Load components
    print("Loading model components...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Minimal training args to avoid overhead
    training_args.max_steps = total_steps
    training_args.save_steps = total_steps + 100  # Never save
    training_args.logging_steps = total_steps + 100  # Never log
    training_args.eval_steps = total_steps + 100  # Never eval
    training_args.output_dir = "temp_simple_profile"
    training_args.overwrite_output_dir = True
    
    # Load dataset
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]
    
    # Load model
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
    
    # Simple dataloader
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
        with_stack=False
    )
    
    print("Starting simple profiled training...")
    
    # Pure training loop with profiling
    step = 0
    with profiler:
        for batch in dataloader:
            if step >= total_steps:
                break
                
            # Move to device if needed
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
            
            # Profile step
            profiler.step()
            
            if step % 5 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
            
            step += 1
    
    # Export trace
    chrome_trace_path = os.path.join(output_dir, "simple_training_trace.json")
    try:
        profiler.export_chrome_trace(chrome_trace_path)
        print(f"\nSimple profiling completed!")
        print(f"Chrome trace: {chrome_trace_path}")
        print(f"Steps profiled: {step}")
        print(f"Open chrome://tracing and load the JSON file")
    except Exception as e:
        print(f"Failed to export trace: {e}")
    
    # Cleanup
    import shutil
    if os.path.exists("temp_simple_profile"):
        shutil.rmtree("temp_simple_profile")


def main():
    parser = argparse.ArgumentParser(description="Simple training profiler")
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--output-dir", default="simple_traces", help="Trace output directory")
    parser.add_argument("--wait", type=int, default=2, help="Wait steps")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup steps")
    parser.add_argument("--active", type=int, default=5, help="Active profile steps")
    parser.add_argument("--total", type=int, default=15, help="Total training steps")
    
    args = parser.parse_args()
    
    simple_profile_training(
        config_file=args.config,
        output_dir=args.output_dir,
        wait_steps=args.wait,
        warmup_steps=args.warmup,
        active_steps=args.active,
        total_steps=args.total
    )


if __name__ == "__main__":
    main()