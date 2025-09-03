#!/usr/bin/env python3
"""
Profile preprocessing bottlenecks for long-context datasets.
This script measures time taken for different preprocessing stages.
"""

import time
import json
import psutil
import os
import sys
from datetime import datetime
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def profile_stage(stage_name):
    """Decorator to profile individual stages"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"Starting: {stage_name}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Initial Memory: {get_memory_usage():.2f} GB")
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            elapsed = end_time - start_time
            print(f"Completed: {stage_name}")
            print(f"Duration: {elapsed:.2f} seconds")
            print(f"Final Memory: {get_memory_usage():.2f} GB")
            print(f"{'='*60}\n")
            
            return result, {
                'stage': stage_name,
                'duration': elapsed,
                'memory_gb': get_memory_usage()
            }
        return wrapper
    return decorator

@profile_stage("Environment Setup")
def setup_environment():
    """Setup and verify environment"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    # Import after environment setup
    import torch
    from llamafactory.train import run_exp
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    return True

@profile_stage("Dataset Loading")
def test_dataset_loading():
    """Test loading the dataset"""
    from datasets import load_dataset
    
    print("Loading NExtLong-512K dataset (first 10 samples)...")
    dataset = load_dataset(
        "caskcsg/NExtLong-512K-dataset",
        split="train",
        streaming=False,
        trust_remote_code=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"First sample length: {len(dataset[0]['text'])} characters")
    
    return dataset

@profile_stage("Tokenization Test")
def test_tokenization(num_samples=5):
    """Test tokenization speed"""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    print(f"Testing tokenization on {num_samples} samples...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_fast=True,
        trust_remote_code=True
    )
    
    # Load dataset samples
    dataset = load_dataset(
        "caskcsg/NExtLong-512K-dataset",
        split=f"train[:{num_samples}]",
        trust_remote_code=True
    )
    
    # Test different cutoff lengths
    cutoff_lengths = [8192, 32768, 131072, 524288]
    results = []
    
    for cutoff in cutoff_lengths:
        print(f"\nTesting cutoff_len={cutoff}...")
        start = time.time()
        
        for sample in dataset:
            tokens = tokenizer(
                sample['text'],
                truncation=True,
                max_length=cutoff,
                return_tensors=None
            )
            
        elapsed = time.time() - start
        tokens_per_sec = (num_samples * cutoff) / elapsed if elapsed > 0 else 0
        
        result = {
            'cutoff': cutoff,
            'time': elapsed,
            'samples_per_sec': num_samples / elapsed if elapsed > 0 else 0,
            'tokens_per_sec': tokens_per_sec
        }
        results.append(result)
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Samples/sec: {result['samples_per_sec']:.2f}")
        print(f"  Est. tokens/sec: {tokens_per_sec:.0f}")
    
    return results

@profile_stage("Full Training Launch")
def test_training_launch(config_path):
    """Launch actual training with profiling config"""
    from llamafactory.train import run_exp
    
    print(f"Launching training with config: {config_path}")
    
    # Set limited steps for profiling
    os.environ['MAX_STEPS'] = '5'
    
    try:
        run_exp(config_path)
    except Exception as e:
        print(f"Training error (expected for profiling): {e}")
    
    return True

def main():
    """Main profiling function"""
    print("\n" + "="*80)
    print("LONG-CONTEXT PREPROCESSING PROFILER")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'stages': []
    }
    
    # Profile each stage
    try:
        # Setup
        _, setup_stats = setup_environment()
        results['stages'].append(setup_stats)
        
        # Dataset loading
        _, load_stats = test_dataset_loading()
        results['stages'].append(load_stats)
        
        # Tokenization
        token_results, token_stats = test_tokenization(num_samples=3)
        results['stages'].append(token_stats)
        results['tokenization_details'] = token_results
        
        # Full training launch (optional)
        config_path = "profile_nextlong_512k.yaml"
        if Path(config_path).exists():
            print("\nLaunching full preprocessing test...")
            print("NOTE: This may take significant time for 512K context")
            _, train_stats = test_training_launch(config_path)
            results['stages'].append(train_stats)
        
    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user")
    except Exception as e:
        print(f"\n\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_file = f"profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PROFILING SUMMARY")
    print(f"{'='*80}")
    
    total_time = sum(s['duration'] for s in results['stages'])
    print(f"Total time: {total_time:.2f} seconds")
    
    print("\nStage breakdown:")
    for stage in results['stages']:
        pct = (stage['duration'] / total_time * 100) if total_time > 0 else 0
        print(f"  {stage['stage']:<30} {stage['duration']:>8.2f}s ({pct:>5.1f}%)")
    
    if 'tokenization_details' in results:
        print("\nTokenization performance by context length:")
        for detail in results['tokenization_details']:
            print(f"  {detail['cutoff']:>7} tokens: {detail['samples_per_sec']:.2f} samples/sec")
    
    print(f"\nResults saved to: {output_file}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if 'tokenization_details' in results:
        # Check if 512K is slow
        max_cutoff_result = max(results['tokenization_details'], key=lambda x: x['cutoff'])
        if max_cutoff_result['samples_per_sec'] < 0.1:
            print("\n⚠️  CRITICAL: 512K context tokenization is extremely slow!")
            print("   Recommendation: Use pretokenization for datasets with >128K context")
            print("   This will save significant preprocessing time during training")

if __name__ == "__main__":
    main()