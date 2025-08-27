#!/usr/bin/env python3
"""
Unified sweep runner for RoPE hyperparameter optimization.
Supports single-node and distributed execution.
"""

import argparse
import json
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RoPESweepRunner:
    """Main class for running RoPE hyperparameter sweeps."""
    
    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
        self.results_dir = Path(self.config.get('results_dir', 'sweep_results'))
        self.results_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load sweep configuration."""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'model_name_or_path': 'unsloth/Llama-3.2-1B-Instruct',
            'dataset': 'alpaca_en_demo',
            'template': 'llama3',
            'max_steps': 5,
            'bf16': True,
            'rope_configurations': [
                {'rope_scaling': 'linear', 'rope_theta': 500000.0, 'cutoff_len': 4096},
                {'rope_scaling': 'linear', 'rope_theta': 1000000.0, 'cutoff_len': 8192},
                {'rope_scaling': 'dynamic', 'rope_theta': 1000000.0, 'cutoff_len': 8192},
                {'rope_scaling': 'linear', 'rope_theta': 2000000.0, 'cutoff_len': 16384},
                {'rope_scaling': 'dynamic', 'rope_theta': 2000000.0, 'cutoff_len': 16384},
            ],
            'results_dir': 'sweep_results'
        }
    
    def generate_config(self, rope_config: Dict[str, Any], run_id: int) -> str:
        """Generate YAML configuration for a single run."""
        base_config = {
            'model_name_or_path': self.config['model_name_or_path'],
            'dataset': self.config['dataset'],
            'template': self.config['template'],
            'stage': 'sft',
            'do_train': True,
            'finetuning_type': 'full',
            'output_dir': f"{self.results_dir}/run_{run_id}",
            'overwrite_output_dir': True,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 1,
            'learning_rate': 5e-5,
            'max_steps': self.config.get('max_steps', 5),
            'logging_steps': 1,
            'save_steps': max(self.config.get('max_steps', 5), 1),
            'warmup_steps': 0,
            'optim': 'adamw_torch',
            'bf16': self.config.get('bf16', True),
            'seed': 42,
            **rope_config
        }
        
        config_file = self.results_dir / f"config_{run_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        return str(config_file)
    
    def run_single_config(self, config_file: str, run_id: int) -> Dict[str, Any]:
        """Run training with a single configuration."""
        logger.info(f"Running configuration {run_id}: {Path(config_file).name}")
        
        start_time = datetime.now()
        
        try:
            # Run training
            result = subprocess.run(
                ['llamafactory-cli', 'train', config_file],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per run
            )
            
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            
            # Parse results
            if result.returncode == 0:
                status = "success"
                # Try to extract metrics from output
                metrics = self.extract_metrics(result.stdout)
            else:
                status = "failed"
                metrics = {"error": result.stderr}
            
            return {
                "run_id": run_id,
                "config_file": config_file,
                "status": status,
                "metrics": metrics,
                "elapsed_seconds": elapsed,
                "timestamp": end_time.isoformat()
            }
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Configuration {run_id} timed out")
            return {
                "run_id": run_id,
                "config_file": config_file,
                "status": "timeout",
                "elapsed_seconds": 1800,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Configuration {run_id} failed with error: {e}")
            return {
                "run_id": run_id,
                "config_file": config_file,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_metrics(self, output: str) -> Dict[str, Any]:
        """Extract metrics from training output."""
        metrics = {}
        
        # Look for common patterns in output
        lines = output.split('\n')
        for line in lines:
            if 'train_loss' in line:
                try:
                    # Extract loss value
                    parts = line.split('train_loss')
                    if len(parts) > 1:
                        loss_part = parts[1].split(',')[0]
                        loss_value = float(loss_part.strip(': ='))
                        metrics['final_loss'] = loss_value
                except:
                    pass
            
            if 'trainable params' in line:
                try:
                    # Extract parameter count
                    parts = line.split('trainable params:')
                    if len(parts) > 1:
                        params = parts[1].split('||')[0].strip()
                        metrics['trainable_params'] = params
                except:
                    pass
        
        return metrics
    
    def run_sweep(self) -> None:
        """Run the complete hyperparameter sweep."""
        logger.info("Starting RoPE hyperparameter sweep")
        logger.info(f"Total configurations: {len(self.config['rope_configurations'])}")
        
        results = []
        
        for i, rope_config in enumerate(self.config['rope_configurations']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Configuration {i+1}/{len(self.config['rope_configurations'])}")
            logger.info(f"RoPE config: {rope_config}")
            
            # Generate configuration file
            config_file = self.generate_config(rope_config, i)
            
            # Run training
            result = self.run_single_config(config_file, i)
            results.append(result)
            
            # Log result
            if result['status'] == 'success':
                logger.info(f"✅ Configuration {i} completed successfully")
                if 'final_loss' in result['metrics']:
                    logger.info(f"   Final loss: {result['metrics']['final_loss']:.4f}")
            else:
                logger.warning(f"❌ Configuration {i} {result['status']}")
        
        # Save results summary
        summary = {
            'total_configs': len(self.config['rope_configurations']),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] in ['failed', 'error']]),
            'timeout': len([r for r in results if r['status'] == 'timeout']),
            'results': results,
            'sweep_config': self.config,
            'completed_at': datetime.now().isoformat()
        }
        
        summary_file = self.results_dir / 'sweep_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info("Sweep completed!")
        logger.info(f"Successful: {summary['successful']}/{summary['total_configs']}")
        logger.info(f"Results saved to: {summary_file}")
        
        # Show best configuration if any succeeded
        successful_results = [r for r in results if r['status'] == 'success' and 'final_loss' in r['metrics']]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['metrics']['final_loss'])
            logger.info(f"Best configuration: {best_result['config_file']}")
            logger.info(f"Best loss: {best_result['metrics']['final_loss']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run RoPE hyperparameter sweep')
    parser.add_argument('--config', type=str, help='Sweep configuration file')
    parser.add_argument('--preset', choices=['quick', 'standard', 'comprehensive'], 
                       default='standard', help='Use preset configuration')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.preset and not args.config:
        preset_configs = {
            'quick': {
                'max_steps': 2,
                'rope_configurations': [
                    {'rope_scaling': 'linear', 'rope_theta': 500000.0, 'cutoff_len': 2048},
                    {'rope_scaling': 'linear', 'rope_theta': 1000000.0, 'cutoff_len': 4096},
                ]
            },
            'standard': {
                'max_steps': 5,
                'rope_configurations': [
                    {'rope_scaling': 'linear', 'rope_theta': 500000.0, 'cutoff_len': 4096},
                    {'rope_scaling': 'linear', 'rope_theta': 1000000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'dynamic', 'rope_theta': 1000000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'linear', 'rope_theta': 2000000.0, 'cutoff_len': 16384},
                ]
            },
            'comprehensive': {
                'max_steps': 10,
                'rope_configurations': [
                    {'rope_scaling': 'linear', 'rope_theta': 500000.0, 'cutoff_len': 4096},
                    {'rope_scaling': 'linear', 'rope_theta': 1000000.0, 'cutoff_len': 4096},
                    {'rope_scaling': 'linear', 'rope_theta': 2000000.0, 'cutoff_len': 4096},
                    {'rope_scaling': 'linear', 'rope_theta': 500000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'linear', 'rope_theta': 1000000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'linear', 'rope_theta': 2000000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'dynamic', 'rope_theta': 500000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'dynamic', 'rope_theta': 1000000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'dynamic', 'rope_theta': 2000000.0, 'cutoff_len': 8192},
                    {'rope_scaling': 'linear', 'rope_theta': 1000000.0, 'cutoff_len': 16384},
                    {'rope_scaling': 'linear', 'rope_theta': 2000000.0, 'cutoff_len': 16384},
                    {'rope_scaling': 'dynamic', 'rope_theta': 2000000.0, 'cutoff_len': 16384},
                ]
            }
        }
        
        # Save preset config
        preset_file = f"sweep_preset_{args.preset}.yaml"
        with open(preset_file, 'w') as f:
            yaml.dump(preset_configs[args.preset], f)
        args.config = preset_file
    
    # Run sweep
    runner = RoPESweepRunner(args.config)
    runner.run_sweep()

if __name__ == '__main__':
    main()