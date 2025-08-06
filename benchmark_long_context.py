#!/usr/bin/env python3
"""
Benchmark script for testing inference-time long context methods in 360-LLaMA-Factory.
Tests various RoPE scaling methods and attention optimizations on long sequence tasks.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import subprocess
import sys

import torch
import numpy as np
from transformers import AutoTokenizer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    base_model_path: Optional[str] = None
    context_lengths: List[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384, 32768])
    rope_methods: List[str] = field(default_factory=lambda: ["none", "linear", "dynamic"])
    attention_types: List[str] = field(default_factory=lambda: ["auto", "sdpa", "fa2"])
    use_shift_attn: List[bool] = field(default_factory=lambda: [False, True])
    batch_size: int = 1
    num_samples: int = 10
    output_dir: str = "benchmark_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    template: str = "llama3"
    

class LongContextBenchmark:
    """Benchmark suite for long context methods."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_test_prompts(self, context_length: int, num_samples: int) -> List[str]:
        """Generate test prompts with specified context length."""
        prompts = []
        
        # Create a simple needle-in-haystack style prompt
        for i in range(num_samples):
            # Generate context
            context_words = []
            target_word = f"KEYWORD_{i}"
            target_position = np.random.randint(context_length // 4, 3 * context_length // 4)
            
            for j in range(context_length // 10):  # Approximate word count
                if j == target_position // 10:
                    context_words.append(target_word)
                else:
                    context_words.append(f"word_{j}")
            
            context = " ".join(context_words)
            prompt = f"In the following text, find the keyword that starts with 'KEYWORD_'. Text: {context}\n\nThe keyword is:"
            prompts.append((prompt, target_word))
            
        return prompts
    
    def run_inference(self, 
                     rope_scaling: str,
                     attention_type: str,
                     shift_attn: bool,
                     context_length: int,
                     prompt: str) -> Tuple[str, float, float]:
        """Run inference with specified configuration."""
        
        # Create config file for this run
        config_file = self.output_dir / f"config_temp.yaml"
        
        config_content = f"""
model_name_or_path: {self.config.model_name or self.config.base_model_path}
template: {self.config.template}
model_max_length: {context_length}
rope_scaling: {rope_scaling}
flash_attn: {attention_type}
shift_attn: {shift_attn}
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Create input file
        input_file = self.output_dir / "input_temp.txt"
        with open(input_file, 'w') as f:
            f.write(prompt)
        
        # Run inference using llamafactory CLI
        start_time = time.time()
        
        cmd = [
            "python", "-m", "llamafactory.cli",
            "chat",
            "--config", str(config_file),
            "--input", str(input_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            inference_time = time.time() - start_time
            
            if result.returncode == 0:
                output = result.stdout
                # Calculate memory usage if possible
                memory_usage = self._get_memory_usage()
                return output, inference_time, memory_usage
            else:
                print(f"Error in inference: {result.stderr}")
                return "", inference_time, 0.0
                
        except subprocess.TimeoutExpired:
            print("Inference timeout")
            return "", 120.0, 0.0
        except Exception as e:
            print(f"Inference failed: {e}")
            return "", 0.0, 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def evaluate_accuracy(self, output: str, expected: str) -> bool:
        """Check if the output contains the expected answer."""
        return expected.lower() in output.lower()
    
    def run_benchmark(self):
        """Run the complete benchmark suite."""
        
        print("Starting Long Context Benchmark")
        print("=" * 60)
        
        total_configs = (
            len(self.config.context_lengths) * 
            len(self.config.rope_methods) * 
            len(self.config.attention_types) * 
            len(self.config.use_shift_attn)
        )
        
        print(f"Total configurations to test: {total_configs}")
        
        config_idx = 0
        
        for context_length in self.config.context_lengths:
            print(f"\nTesting context length: {context_length}")
            
            # Generate test prompts for this context length
            test_prompts = self.generate_test_prompts(context_length, self.config.num_samples)
            
            for rope_method in self.config.rope_methods:
                for attention_type in self.config.attention_types:
                    for use_shift_attn in self.config.use_shift_attn:
                        config_idx += 1
                        
                        # Skip shift_attn if not using appropriate rope scaling
                        if use_shift_attn and rope_method == "none":
                            continue
                        
                        print(f"\nConfig {config_idx}/{total_configs}:")
                        print(f"  RoPE: {rope_method}")
                        print(f"  Attention: {attention_type}")
                        print(f"  Shift Attn: {use_shift_attn}")
                        print(f"  Context: {context_length}")
                        
                        config_results = {
                            "context_length": context_length,
                            "rope_scaling": rope_method,
                            "attention_type": attention_type,
                            "shift_attn": use_shift_attn,
                            "inference_times": [],
                            "memory_usage": [],
                            "accuracy": [],
                        }
                        
                        for prompt, expected in test_prompts[:self.config.num_samples]:
                            output, inf_time, memory = self.run_inference(
                                rope_method,
                                attention_type,
                                use_shift_attn,
                                context_length,
                                prompt
                            )
                            
                            accuracy = self.evaluate_accuracy(output, expected)
                            
                            config_results["inference_times"].append(inf_time)
                            config_results["memory_usage"].append(memory)
                            config_results["accuracy"].append(accuracy)
                        
                        # Calculate statistics
                        config_results["avg_inference_time"] = np.mean(config_results["inference_times"])
                        config_results["std_inference_time"] = np.std(config_results["inference_times"])
                        config_results["avg_memory_gb"] = np.mean(config_results["memory_usage"])
                        config_results["accuracy_rate"] = np.mean(config_results["accuracy"])
                        
                        self.results.append(config_results)
                        
                        # Save intermediate results
                        self.save_results()
                        
                        print(f"  Avg Time: {config_results['avg_inference_time']:.2f}s")
                        print(f"  Accuracy: {config_results['accuracy_rate']:.2%}")
                        print(f"  Memory: {config_results['avg_memory_gb']:.2f}GB")
    
    def save_results(self):
        """Save benchmark results to file."""
        output_file = self.output_dir / "benchmark_results.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "config": {
                    "model": self.config.model_name,
                    "context_lengths": self.config.context_lengths,
                    "rope_methods": self.config.rope_methods,
                    "attention_types": self.config.attention_types,
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def generate_report(self):
        """Generate a detailed report of benchmark results."""
        report_file = self.output_dir / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Long Context Methods Benchmark Report\n\n")
            f.write(f"## Model: {self.config.model_name}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Config | Context | RoPE | Attention | Shift | Avg Time (s) | Accuracy | Memory (GB) |\n")
            f.write("|--------|---------|------|-----------|-------|--------------|----------|-------------|\n")
            
            for result in self.results:
                f.write(f"| {self.results.index(result)+1} | "
                       f"{result['context_length']} | "
                       f"{result['rope_scaling']} | "
                       f"{result['attention_type']} | "
                       f"{'Yes' if result['shift_attn'] else 'No'} | "
                       f"{result['avg_inference_time']:.2f} | "
                       f"{result['accuracy_rate']:.2%} | "
                       f"{result['avg_memory_gb']:.2f} |\n")
            
            # Best configurations
            f.write("\n## Best Configurations\n\n")
            
            for context_length in self.config.context_lengths:
                context_results = [r for r in self.results if r['context_length'] == context_length]
                if context_results:
                    best_speed = min(context_results, key=lambda x: x['avg_inference_time'])
                    best_accuracy = max(context_results, key=lambda x: x['accuracy_rate'])
                    
                    f.write(f"\n### Context Length: {context_length}\n")
                    f.write(f"- **Fastest**: RoPE={best_speed['rope_scaling']}, "
                           f"Attention={best_speed['attention_type']}, "
                           f"Time={best_speed['avg_inference_time']:.2f}s\n")
                    f.write(f"- **Most Accurate**: RoPE={best_accuracy['rope_scaling']}, "
                           f"Attention={best_accuracy['attention_type']}, "
                           f"Accuracy={best_accuracy['accuracy_rate']:.2%}\n")
        
        print(f"Report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark long context methods")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model name or path")
    parser.add_argument("--contexts", nargs="+", type=int,
                       default=[2048, 4096, 8192],
                       help="Context lengths to test")
    parser.add_argument("--rope-methods", nargs="+",
                       default=["none", "linear", "dynamic"],
                       help="RoPE scaling methods to test")
    parser.add_argument("--attention-types", nargs="+",
                       default=["auto", "sdpa", "fa2"],
                       help="Attention implementations to test")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples per configuration")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer samples")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick benchmark settings
        config = BenchmarkConfig(
            model_name=args.model,
            context_lengths=[2048, 4096],
            rope_methods=["none", "linear"],
            attention_types=["auto"],
            use_shift_attn=[False],
            num_samples=2,
            output_dir=args.output_dir
        )
    else:
        config = BenchmarkConfig(
            model_name=args.model,
            context_lengths=args.contexts,
            rope_methods=args.rope_methods,
            attention_types=args.attention_types,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    
    benchmark = LongContextBenchmark(config)
    
    try:
        benchmark.run_benchmark()
        benchmark.generate_report()
        print("\nBenchmark completed successfully!")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        benchmark.save_results()
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        benchmark.save_results()
        raise


if __name__ == "__main__":
    main()