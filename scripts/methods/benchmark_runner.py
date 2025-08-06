#!/usr/bin/env python3

import os
import sys
import json
import time
import shutil
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class MethodBenchmarkRunner:
    def __init__(self, model_path: str, results_dir: Optional[str] = None):
        self.model_path = model_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(results_dir or f"saves/methodwise/results_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Method configurations
        self.methods = {
            "baseline": {
                "contexts": [2048, 4096], 
                "rope_config": None
            },
            "linear": {
                "contexts": [2048, 4096, 8192, 16384], 
                "rope_config": {"type": "linear", "factor": 2.0}
            },
            "dynamic": {
                "contexts": [2048, 4096, 8192, 16384], 
                "rope_config": {"type": "dynamic", "factor": 2.0}
            },
            "yarn": {
                "contexts": [2048, 4096, 8192, 16384, 32768], 
                "rope_config": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 2048,
                    "attention_factor": 1.0,
                    "beta_fast": 32,
                    "beta_slow": 1
                },
                "extra_params": {}
            },
            "longrope": {
                "contexts": [2048, 4096, 8192, 16384, 32768], 
                "rope_config": {
                    "rope_type": "longrope",
                    "factor": 4.0,
                    "original_max_position_embeddings": 2048,
                    "short_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "long_factor": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
                },
                "extra_params": {}
            },
            "nope": {
                "contexts": [2048, 4096, 8192, 16384, 32768], 
                "rope_config": "nope",
                "extra_params": {}
            }
        }
        
        # Initialize results tracking
        self.results = []
        self.detailed_log_file = self.results_dir / "detailed_results.jsonl"
        self.csv_file = self.results_dir / "results.csv"
        
    def create_config(self, method: str, contexts: List[int], rope_config = None, extra_params: Optional[Dict] = None) -> Path:
        """Create YAML configuration for a method"""
        max_context = max(contexts)
        config_file = Path(f"{method}_config.yaml")
        
        # Clean up existing save directory to avoid conflicts
        save_dir = Path(f"saves/methodwise/{method}")
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "model_name_or_path": self.model_path,
            "finetuning_type": "full",
            "cutoff_len": max_context,
            "task": "needle_haystack_proper",
            "task_dir": "evaluation",
            "template": "llama3",
            "lang": "en",
            "save_dir": str(save_dir),
            "batch_size": 1,
            "needle_context_lengths": contexts,
            "needle_depth_percents": [10, 50, 90],
            "needle_text": f"The benchmark secret is METHOD_{method.upper()}_SUCCESS.",
            "needle_question": "What is the benchmark secret mentioned in the document?",
            "needle_haystack_data_source": "paulgraham",
            "flash_attn": "fa2",
            "use_cache": True,
            "low_cpu_mem_usage": True
        }
        
        # Add rope scaling configuration
        if rope_config:
            if isinstance(rope_config, str):
                config["rope_scaling"] = rope_config
            elif isinstance(rope_config, dict):
                if "rope_type" in rope_config:
                    # Handle official rope types (YaRN, LongRope) with full dictionary
                    config["rope_scaling"] = rope_config.copy()
                elif "type" in rope_config:
                    # Handle simple types (linear, dynamic)
                    rope_type = rope_config["type"]
                    if rope_type in ["linear", "dynamic"]:
                        config["rope_scaling"] = rope_type
                        config["rope_factor"] = rope_config["factor"]
                    else:
                        config["rope_scaling"] = rope_config.copy()
        
        # Add extra parameters for specialized methods
        if extra_params:
            config.update(extra_params)
        
        # Write proper YAML file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return config_file
    
    def run_evaluation(self, method: str, config_file: Path, timeout: int = 1800) -> Dict:
        """Run evaluation for a single method"""
        print(f"\nRunning {method.upper()} evaluation...")
        
        start_time = time.time()
        result_dir = Path(f"saves/methodwise/{method}")
        
        # Run evaluation (tqdm progress is handled inside the evaluation script)
        cmd = ["python3", "run_needle_eval.py", str(config_file)]
        log_file = self.results_dir / f"{method}.log"
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Stream output in real-time
            for line in process.stdout:
                f.write(line)
                # Show tqdm progress lines
                if "Processing examples" in line or "%" in line:
                    print(line.strip())
        
        process.wait()
        duration = int(time.time() - start_time)
        
        # Check results
        summary_file = result_dir / "summary.json"
        if process.returncode == 0 and summary_file.exists():
            return self.process_success(method, result_dir, duration)
        else:
            return self.process_failure(method, log_file, duration, process.returncode)
    
    def process_success(self, method: str, result_dir: Path, duration: int) -> Dict:
        """Process successful evaluation results"""
        try:
            with open(result_dir / "summary.json") as f:
                data = json.load(f)
            
            overall = data.get('overall', {})
            accuracy = overall.get('average_score', 0)
            total_examples = overall.get('total_examples', 0)
            
            by_length = data.get('by_context_length', {})
            max_context = max([int(k) for k in by_length.keys()]) if by_length else 0
            
            result = {
                "method": method,
                "status": "SUCCESS",
                "duration": duration,
                "accuracy": accuracy,
                "total_examples": total_examples,
                "max_context": max_context,
                "timestamp": datetime.now().isoformat(),
                "detailed_results": data
            }
            
            print(f"✓ {method.upper()}: SUCCESS ({duration}s)")
            print(f"  Overall accuracy: {accuracy:.1%} ({total_examples} examples)")
            
            # Show context length breakdown
            if by_length:
                print("  By context length:")
                for length in sorted(by_length.keys(), key=int):
                    stats = by_length[length]
                    acc = stats.get('average_score', 0)
                    count = stats.get('count', 0)
                    status = 'GOOD' if acc >= 0.8 else 'FAIR' if acc >= 0.5 else 'POOR'
                    print(f"    {status:4} {int(length):>6}T: {acc:.1%} ({count})")
            
            return result
            
        except Exception as e:
            print(f"✗ {method.upper()}: ERROR processing results - {e}")
            return {
                "method": method,
                "status": "ERROR",
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_failure(self, method: str, log_file: Path, duration: int, return_code: int) -> Dict:
        """Process failed evaluation"""
        try:
            with open(log_file) as f:
                log_content = f.read()
            error_lines = log_content.split('\n')[-10:]  # Last 10 lines
            error_summary = '\n'.join([line for line in error_lines if line.strip()])
        except:
            error_summary = "Could not read error log"
        
        status = "TIMEOUT" if return_code == 124 else "FAILED"
        print(f"✗ {method.upper()}: {status} ({duration}s)")
        print("  Error details:")
        for line in error_summary.split('\n')[-3:]:  # Show last 3 lines
            if line.strip():
                print(f"    {line.strip()}")
        
        return {
            "method": method,
            "status": status,
            "duration": duration,
            "error_log": error_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self):
        """Save results to CSV and JSON files"""
        # Save detailed JSON log
        with open(self.detailed_log_file, 'a') as f:
            for result in self.results:
                f.write(json.dumps(result) + '\n')
        
        # Save CSV summary
        with open(self.csv_file, 'w') as f:
            f.write("Method,Status,Duration,Accuracy,Examples,MaxContext\n")
            for result in self.results:
                accuracy = result.get('accuracy', 0)
                examples = result.get('total_examples', 0)
                max_context = result.get('max_context', 0)
                f.write(f"{result['method']},{result['status']},{result['duration']},{accuracy:.3f},{examples},{max_context}\n")
    
    def run_all_benchmarks(self):
        """Run benchmarks for all methods"""
        print("=" * 60)
        print("Method-wise Long Context Benchmark")
        print("Testing contexts: 2K -> 4K -> 8K -> 16K -> 32K")
        print("=" * 60)
        print(f"Results directory: {self.results_dir}")
        print(f"Model: {self.model_path}")
        
        # Initialize result files
        with open(self.detailed_log_file, 'w') as f:
            f.write(json.dumps({
                "benchmark_start": datetime.now().isoformat(),
                "model": self.model_path
            }) + '\n')
        
        total_methods = len(self.methods)
        
        for i, (method, config) in enumerate(self.methods.items(), 1):
            print(f"\n{'='*60}")
            print(f"Method {i}/{total_methods}: {method.upper()}")
            print(f"Contexts: {config['contexts']}")
            print(f"{'='*60}")
            
            # Create configuration
            extra_params = config.get("extra_params", {})
            config_file = self.create_config(method, config["contexts"], config.get("rope_config"), extra_params)
            
            # Run evaluation
            result = self.run_evaluation(method, config_file, timeout=1800)
            self.results.append(result)
            
            # Cleanup config file
            if config_file.exists():
                config_file.unlink()
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between methods
            if i < total_methods:
                time.sleep(2)
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final benchmark report"""
        print("\n" + "="*60)
        print("Benchmark Complete!")
        print("="*60)
        
        print(f"\nRESULTS SUMMARY:")
        print(f"{'Method':<12} | {'Status':<8} | {'Duration':<8} | {'Accuracy':<8} | {'MaxContext':<10}")
        print("-"*13 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*11)
        
        successful_methods = []
        
        for result in self.results:
            method = result["method"]
            status = result["status"]
            duration = result["duration"]
            
            if status == "SUCCESS":
                accuracy = result.get("accuracy", 0)
                max_context = result.get("max_context", 0)
                print(f"{method:<12} | {status:<8} | {duration:>6}s | {accuracy:>6.1%} | {max_context:<10}")
                successful_methods.append(method)
            else:
                print(f"{method:<12} | {status:<8} | {duration:>6}s | {'N/A':<8} | {'0':<10}")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"JSON logs: {self.detailed_log_file}")
        print(f"CSV summary: {self.csv_file}")
        
        if successful_methods:
            print(f"\nSuccessful methods: {', '.join(successful_methods)}")
        else:
            print(f"\nNo methods completed successfully. Check logs for issues.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_runner.py <model_path> [results_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    runner = MethodBenchmarkRunner(model_path, results_dir)
    runner.run_all_benchmarks()

if __name__ == "__main__":
    main()