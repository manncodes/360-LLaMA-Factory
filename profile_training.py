#!/usr/bin/env python3
"""
Standalone profiling script for LlamaFactory training.
Profiles only training iteration steps without checkpoint saving overhead.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.model import load_model, load_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


def profile_training(config_file: str, output_dir: str = "traces", wait_steps: int = 5, 
                    warmup_steps: int = 2, active_steps: int = 10, max_profile_steps: int = 50):
    """
    Profile training iterations without checkpoint overhead.
    
    Args:
        config_file: Path to YAML configuration file
        output_dir: Directory to save profiling traces
        wait_steps: Steps to wait before profiling starts
        warmup_steps: Warmup steps for profiler
        active_steps: Steps to actively profile
        max_profile_steps: Maximum steps to run (keeps it short)
    """
    print(f"Starting profiling with config: {config_file}")
    print(f"Profiler schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}")
    
    # Parse training arguments from YAML
    try:
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config_file)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        return
    
    # Override training args to prevent checkpointing during profiling
    training_args.save_steps = max_profile_steps + 100  # Don't save during profiling
    training_args.logging_steps = 1  # Log every step for monitoring
    training_args.max_steps = max_profile_steps  # Keep training short
    training_args.output_dir = "temp_profile_output"  # Temporary output
    training_args.overwrite_output_dir = True
    
    # Create output directory for traces
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    print("Loading tokenizer and model...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    print("Loading dataset...")
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    
    print("Loading model...")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, full_determinism=training_args.full_determinism)
    
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
    
    # Setup profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    print("Setting up PyTorch profiler...")
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
        with_stack=False  # Reduces overhead
    )
    
    # Create trainer (without callbacks to avoid checkpoint overhead)
    print("Creating trainer...")
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module.get("eval_dataset"),
        tokenizer=tokenizer,
        callbacks=[]  # No callbacks to avoid overhead
    )
    
    # Start profiling and training
    print("Starting profiled training...")
    with profiler:
        try:
            # Custom training loop for precise profiling control
            trainer.model.train()
            train_dataloader = trainer.get_train_dataloader()
            
            step = 0
            for batch in train_dataloader:
                if step >= max_profile_steps:
                    break
                
                # Move batch to device
                batch = trainer._prepare_inputs(batch)
                
                # Forward pass
                with trainer.compute_loss_context_manager():
                    loss = trainer.compute_loss(trainer.model, batch)
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                trainer.optimizer.step()
                trainer.lr_scheduler.step()
                trainer.optimizer.zero_grad()
                
                # Profile step
                profiler.step()
                
                step += 1
                
                if step % 5 == 0:
                    print(f"Step {step}/{max_profile_steps}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
    
    # Export Chrome trace
    chrome_trace_path = os.path.join(output_dir, "training_profile.json")
    try:
        profiler.export_chrome_trace(chrome_trace_path)
        print(f"\nProfiling completed!")
        print(f"Chrome trace saved to: {chrome_trace_path}")
        print(f"View in Chrome: chrome://tracing")
        print(f"Total steps profiled: {step}")
    except Exception as e:
        print(f"Failed to export Chrome trace: {e}")
    
    # Cleanup temporary output
    import shutil
    if os.path.exists("temp_profile_output"):
        shutil.rmtree("temp_profile_output")
    
    print("Profiling script completed.")


def main():
    parser = argparse.ArgumentParser(description="Profile LlamaFactory training iterations")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", default="traces", help="Output directory for traces")
    parser.add_argument("--wait-steps", type=int, default=5, help="Steps to wait before profiling")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps for profiler")
    parser.add_argument("--active-steps", type=int, default=10, help="Steps to actively profile")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum training steps")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found")
        sys.exit(1)
    
    profile_training(
        config_file=args.config,
        output_dir=args.output_dir,
        wait_steps=args.wait_steps,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
        max_profile_steps=args.max_steps
    )


if __name__ == "__main__":
    main()