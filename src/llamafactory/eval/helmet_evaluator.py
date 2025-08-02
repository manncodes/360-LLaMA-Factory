"""
HELMET Long Context Benchmark Integration

This module integrates the HELMET (How to Evaluate Long-context Models Effectively 
and Thoroughly) benchmark into LlamaFactory's evaluation system.

HELMET includes 7 diverse task categories:
- Recall (JSON KV, NIAH variants)
- RAG (Retrieval-Augmented Generation)
- Passage Re-ranking
- Citation (ALCE)
- Long QA
- Summarization  
- In-Context Learning

Reference: https://github.com/princeton-nlp/HELMET
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer

from ..hparams import EvaluationArguments, ModelArguments, DataArguments, FinetuningArguments
from ..data import get_template_and_fix_tokenizer


logger = logging.getLogger(__name__)


class HelmetEvaluator:
    """
    HELMET benchmark evaluator that integrates with LlamaFactory.
    """
    
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        eval_args: EvaluationArguments,
        finetuning_args: FinetuningArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.eval_args = eval_args
        self.finetuning_args = finetuning_args
        
        # Import HELMET modules dynamically
        self._import_helmet_modules()
        
        # Initialize HELMET arguments
        self.helmet_args = self._create_helmet_args()
        
        # Load model using HELMET's model loading system
        self.helmet_model = None
        
    def _import_helmet_modules(self):
        """Import HELMET modules dynamically from the HELMET directory."""
        # Get the path to HELMET directory (one level up from this project)
        helmet_path = Path(__file__).parent.parent.parent.parent.parent / "HELMET"
        
        if not helmet_path.exists():
            raise FileNotFoundError(
                f"HELMET directory not found at {helmet_path}. "
                "Please clone HELMET repository: git clone https://github.com/princeton-nlp/HELMET.git"
            )
        
        # Add HELMET to Python path
        sys.path.insert(0, str(helmet_path))
        
        try:
            # Import HELMET modules
            from arguments import parse_arguments
            from model_utils import load_LLM
            from data import load_data, TestItemDataset
            import eval as helmet_eval
            
            self.helmet_parse_arguments = parse_arguments
            self.helmet_load_LLM = load_LLM
            self.helmet_load_data = load_data
            self.helmet_TestItemDataset = TestItemDataset
            self.helmet_eval = helmet_eval
            
        except ImportError as e:
            raise ImportError(f"Failed to import HELMET modules: {e}")
    
    def _create_helmet_args(self):
        """Convert LlamaFactory arguments to HELMET format."""
        # Create a temporary config file for HELMET
        helmet_config = {
            'model_name_or_path': self.model_args.model_name_or_path,
            'use_chat_template': True,  # LlamaFactory uses chat templates by default
            'do_sample': False,  # Greedy decoding for consistent evaluation
            'temperature': 0.0,
            'top_p': 1.0,
            'generation_min_length': 0,
            'seed': 42,
            'shots': getattr(self.eval_args, 'helmet_shots', 2),
            'max_test_samples': getattr(self.eval_args, 'helmet_max_test_samples', None),
            'no_cuda': not torch.cuda.is_available(),
            'no_bf16': self.finetuning_args.fp16 if hasattr(self.finetuning_args, 'fp16') else False,
            'debug': False,
            'overwrite': True,
            'num_workers': 1,  # Single worker for compatibility
        }
        
        # Set HELMET-specific parameters from eval_args
        if hasattr(self.eval_args, 'helmet_tasks'):
            helmet_config['datasets'] = self.eval_args.helmet_tasks
        if hasattr(self.eval_args, 'helmet_test_files'):
            helmet_config['test_files'] = self.eval_args.helmet_test_files
        if hasattr(self.eval_args, 'helmet_demo_files'):
            helmet_config['demo_files'] = self.eval_args.helmet_demo_files
        if hasattr(self.eval_args, 'helmet_input_max_length'):
            helmet_config['input_max_length'] = str(self.eval_args.helmet_input_max_length)
        if hasattr(self.eval_args, 'helmet_generation_max_length'):
            helmet_config['generation_max_length'] = str(self.eval_args.helmet_generation_max_length)
        
        # Set output directory
        output_dir = getattr(self.eval_args, 'helmet_output_dir', None)
        if output_dir is None:
            output_dir = os.path.join(self.eval_args.save_dir, "helmet_results")
        helmet_config['output_dir'] = output_dir
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(helmet_config, f)
            config_path = f.name
        
        # Parse arguments using HELMET's argument parser
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['helmet_eval', '--config', config_path]
            args = self.helmet_parse_arguments()
            return args
        finally:
            sys.argv = original_argv
            os.unlink(config_path)  # Clean up temp file
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run HELMET evaluation and return results.
        """
        logger.info("Starting HELMET benchmark evaluation...")
        
        # Create output directory
        os.makedirs(self.helmet_args.output_dir, exist_ok=True)
        
        # Load model using HELMET's loading system
        logger.info(f"Loading model: {self.helmet_args.model_name_or_path}")
        self.helmet_model = self.helmet_load_LLM(self.helmet_args)
        
        # Run evaluation on specified tasks
        results = {}
        
        # Parse tasks, test files, and demo files
        datasets = self.helmet_args.datasets.split(",") if self.helmet_args.datasets else []
        test_files = self.helmet_args.test_files.split(",") if self.helmet_args.test_files else []
        demo_files = self.helmet_args.demo_files.split(",") if self.helmet_args.demo_files else []
        
        # Handle input/generation max lengths
        max_lengths = self._parse_lengths(self.helmet_args.input_max_length, len(datasets))
        gen_lengths = self._parse_lengths(self.helmet_args.generation_max_length, len(datasets))
        
        if not datasets:
            # Default to a subset of HELMET tasks
            datasets = ["json_kv", "ruler_niah_s_2"]
            test_files = ["data/json_kv/test_k1800_dep6.jsonl", "data/ruler/niah_single_2/validation_131072.jsonl"]
            demo_files = ["", ""]
            max_lengths = [131072, 131072]
            gen_lengths = [100, 50]
        
        # Ensure lists have same length
        if len(test_files) < len(datasets):
            test_files.extend([''] * (len(datasets) - len(test_files)))
        if len(demo_files) < len(datasets):
            demo_files.extend([''] * (len(datasets) - len(demo_files)))
        
        # Run evaluation for each task
        for dataset, test_file, demo_file, max_length, gen_length in zip(
            datasets, test_files, demo_files, max_lengths, gen_lengths
        ):
            logger.info(f"Evaluating task: {dataset}")
            
            # Update args for this specific task
            task_args = self._update_args_for_task(
                self.helmet_args, dataset, test_file, demo_file, max_length, gen_length
            )
            
            try:
                # Run HELMET evaluation for this task
                output_path = self.helmet_eval.run_test(
                    task_args, self.helmet_model, dataset, test_file, demo_file
                )
                
                # Load and process results
                task_results = self._process_task_results(output_path, dataset)
                results[dataset] = task_results
                
                logger.info(f"Completed task {dataset}: {task_results.get('averaged_metrics', {})}")
                
            except Exception as e:
                logger.error(f"Error evaluating task {dataset}: {e}")
                results[dataset] = {"error": str(e)}
        
        # Compile overall results
        overall_results = self._compile_overall_results(results)
        
        # Save compiled results
        self._save_results(overall_results)
        
        logger.info("HELMET evaluation completed successfully!")
        return overall_results
    
    def _parse_lengths(self, length_str: str, num_tasks: int) -> List[int]:
        """Parse comma-separated length string or repeat single value."""
        if ',' in length_str:
            return [int(l.strip()) for l in length_str.split(',')]
        else:
            return [int(length_str)] * num_tasks
    
    def _update_args_for_task(self, args, dataset, test_file, demo_file, max_length, gen_length):
        """Update arguments for a specific task."""
        task_args = type(args)()  # Create new args object
        for key, value in vars(args).items():
            setattr(task_args, key, value)
        
        # Update task-specific settings
        task_args.datasets = dataset
        task_args.test_files = test_file
        task_args.demo_files = demo_file
        task_args.input_max_length = max_length
        task_args.generation_max_length = gen_length
        
        # Update model settings
        self.helmet_model.max_length = max_length
        self.helmet_model.generation_max_length = gen_length
        
        return task_args
    
    def _process_task_results(self, output_path: str, dataset: str) -> Dict[str, Any]:
        """Process results from a single HELMET task."""
        try:
            # Load main results file
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            # Load score file if available
            score_path = output_path + ".score"
            if os.path.exists(score_path):
                with open(score_path, 'r') as f:
                    scores = json.load(f)
            else:
                scores = results.get('averaged_metrics', {})
            
            return {
                'dataset': dataset,
                'output_path': output_path,
                'averaged_metrics': scores,
                'num_samples': len(results.get('data', [])),
                'throughput': results.get('throughput', 0),
                'memory_usage': results.get('memory_usage', 0),
            }
            
        except Exception as e:
            logger.error(f"Error processing results for {dataset}: {e}")
            return {'dataset': dataset, 'error': str(e)}
    
    def _compile_overall_results(self, task_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compile results across all tasks."""
        overall_metrics = {}
        successful_tasks = []
        failed_tasks = []
        
        for task, results in task_results.items():
            if 'error' in results:
                failed_tasks.append(task)
            else:
                successful_tasks.append(task)
                
                # Aggregate metrics
                for metric, value in results.get('averaged_metrics', {}).items():
                    if metric not in overall_metrics:
                        overall_metrics[metric] = []
                    overall_metrics[metric].append(value)
        
        # Calculate overall averages
        averaged_overall = {}
        for metric, values in overall_metrics.items():
            if values:
                averaged_overall[f'avg_{metric}'] = np.mean(values)
                averaged_overall[f'std_{metric}'] = np.std(values)
        
        return {
            'task_results': task_results,
            'overall_metrics': averaged_overall,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'num_successful_tasks': len(successful_tasks),
            'num_failed_tasks': len(failed_tasks),
            'model_name': self.model_args.model_name_or_path,
            'evaluation_args': vars(self.eval_args),
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save compiled results to file."""
        output_path = os.path.join(self.helmet_args.output_dir, "helmet_evaluation_summary.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"HELMET evaluation summary saved to: {output_path}")
            
            # Also save a simple metrics file
            metrics_path = os.path.join(self.helmet_args.output_dir, "helmet_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(results['overall_metrics'], f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def run_helmet_evaluation(
    model_args: ModelArguments,
    data_args: DataArguments, 
    eval_args: EvaluationArguments,
    finetuning_args: FinetuningArguments,
) -> Dict[str, Any]:
    """
    Main entry point for HELMET evaluation.
    
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = HelmetEvaluator(model_args, data_args, eval_args, finetuning_args)
    return evaluator.evaluate()