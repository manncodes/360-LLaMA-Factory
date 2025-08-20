"""RULER evaluation for comprehensive long-context assessment."""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import yaml
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from ..hparams import ModelArguments, DataArguments, EvaluationArguments


class RULEREvaluator:
    """Evaluates language models using NVIDIA's RULER benchmark."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RULER evaluator with model and tokenizer."""
        self.model_args, self.data_args, self.eval_args, self.finetuning_args = get_eval_args(args)
        
        # Apply RoPE configuration if specified
        if hasattr(self.eval_args, 'rope_scaling_type') and self.eval_args.rope_scaling_type:
            self._apply_rope_config()
        
        # Load tokenizer and model
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"  # RULER typically uses left padding
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
        
        # Set RULER paths
        self.ruler_path = Path(__file__).parent.parent.parent.parent / "third_party" / "RULER"
        self.scripts_path = self.ruler_path / "scripts"
        
        # Ensure RULER dependencies are in path
        sys.path.insert(0, str(self.scripts_path))
        sys.path.insert(0, str(self.scripts_path / "data"))
        sys.path.insert(0, str(self.scripts_path / "eval"))
        
    def _apply_rope_config(self) -> None:
        """Apply RoPE configuration from eval_args to model_args."""
        rope_type = self.eval_args.rope_scaling_type.lower()
        scaling_factor = self.eval_args.rope_scaling_factor or 2.0
        
        print(f"[RULER] Applying RoPE configuration: {rope_type} (factor: {scaling_factor})")
        
        if rope_type in ["linear", "dynamic"]:
            self.model_args.rope_scaling = rope_type
        elif rope_type in ["yarn", "longrope", "llama3"]:
            # Store advanced RoPE config for later application
            self.model_args.rope_scaling = "linear"
            rope_config = {"type": rope_type, "factor": scaling_factor}
            
            if rope_type == "yarn" and hasattr(self.eval_args, 'yarn_alpha'):
                rope_config["alpha"] = self.eval_args.yarn_alpha
            elif rope_type == "longrope" and hasattr(self.eval_args, 'longrope_short_factor'):
                rope_config["short_factor"] = self.eval_args.longrope_short_factor
            
            self.advanced_rope_config = rope_config
    
    def prepare_data(
        self,
        task: str,
        max_seq_length: int,
        num_samples: int = 10,
        subset: str = "validation"
    ) -> Path:
        """Prepare RULER evaluation data for a specific task."""
        print(f"[RULER] Preparing data for task: {task}, max_length: {max_seq_length}")
        
        # Create temporary directory for data
        temp_dir = Path(tempfile.mkdtemp(prefix="ruler_eval_"))
        
        # Prepare tokenizer path
        tokenizer_path = self.model_args.model_name_or_path
        
        # Run RULER data preparation script
        prepare_cmd = [
            "python", str(self.scripts_path / "data" / "prepare.py"),
            "--save_dir", str(temp_dir),
            "--benchmark", "synthetic",
            "--task", task,
            "--subset", subset,
            "--tokenizer_path", tokenizer_path,
            "--tokenizer_type", "hf",
            "--max_seq_length", str(max_seq_length),
            "--num_samples", str(num_samples),
            "--model_template_type", "base",
            "--random_seed", str(self.eval_args.seed or 42)
        ]
        
        try:
            result = subprocess.run(
                prepare_cmd,
                capture_output=True,
                text=True,
                cwd=str(self.scripts_path),
                check=True
            )
            print(f"[RULER] Data prepared successfully in {temp_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[RULER] Error preparing data: {e.stderr}")
            raise
        
        # Find the generated jsonl file
        jsonl_files = list(temp_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {temp_dir}")
        
        return jsonl_files[0]
    
    def generate_predictions(
        self,
        data_path: Path,
        batch_size: int = 1,
        max_new_tokens: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate predictions for RULER evaluation data."""
        print(f"[RULER] Generating predictions from {data_path}")
        
        # Load evaluation data
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for sample in tqdm(data, desc="Generating predictions"):
                input_text = sample['input']
                
                # Tokenize input
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.eval_args.max_length or 32768,
                    padding=True
                ).to(self.model.device)
                
                # Generate prediction
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.eval_args.temperature or 0.0,
                    do_sample=False if (self.eval_args.temperature or 0.0) == 0 else True,
                    top_p=self.eval_args.top_p or 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode prediction
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Store prediction
                prediction = {
                    "index": sample.get("index", 0),
                    "input": input_text,
                    "pred": generated_text,
                    "outputs": sample.get("outputs", []),
                    "others": sample.get("others", {})
                }
                predictions.append(prediction)
        
        return predictions
    
    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        task: str
    ) -> Dict[str, float]:
        """Evaluate predictions using RULER metrics."""
        print(f"[RULER] Evaluating predictions for task: {task}")
        
        # Import RULER evaluation utilities
        try:
            from synthetic.constants import TASKS
            from eval.synthetic.constants import TASKS as EVAL_TASKS
        except ImportError:
            print("[RULER] Warning: Could not import RULER evaluation utilities")
            # Fallback to simple exact match evaluation
            return self._simple_evaluation(predictions)
        
        # Load task configuration
        with open(self.scripts_path / "synthetic.yaml", 'r') as f:
            task_configs = yaml.safe_load(f)
        
        if task not in task_configs:
            print(f"[RULER] Task {task} not found in configuration")
            return {"error": f"Task {task} not found"}
        
        task_config = task_configs[task]
        
        # Get task-specific metric function
        task_type = task_config['task']
        if task_type in EVAL_TASKS:
            metric_fn = EVAL_TASKS[task_type]['metric_fn']
        else:
            metric_fn = lambda preds, refs: sum(
                1 for p, r in zip(preds, refs) if any(ref in p for ref in r)
            ) / len(preds)
        
        # Extract predictions and references
        preds = [p['pred'] for p in predictions]
        refs = [p['outputs'] for p in predictions]
        
        # Calculate score
        score = metric_fn(preds, refs)
        
        # Calculate null predictions
        null_count = sum(1 for p in preds if not p.strip())
        
        results = {
            "task": task,
            "score": score,
            "null_predictions": f"{null_count}/{len(preds)}",
            "num_samples": len(predictions)
        }
        
        return results
    
    def _simple_evaluation(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Simple evaluation fallback using exact match."""
        correct = 0
        total = len(predictions)
        
        for pred in predictions:
            pred_text = pred['pred'].strip().lower()
            references = pred.get('outputs', [])
            
            if any(ref.lower() in pred_text for ref in references):
                correct += 1
        
        return {
            "score": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }
    
    def run_ruler_task(
        self,
        task: str,
        max_seq_length: int = 4096,
        num_samples: int = 10,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run a complete RULER evaluation task."""
        print(f"\n{'='*50}")
        print(f"[RULER] Running task: {task}")
        print(f"[RULER] Max sequence length: {max_seq_length}")
        print(f"[RULER] Number of samples: {num_samples}")
        print(f"{'='*50}\n")
        
        # Prepare data
        data_path = self.prepare_data(task, max_seq_length, num_samples)
        
        # Generate predictions
        predictions = self.generate_predictions(data_path)
        
        # Evaluate predictions
        results = self.evaluate_predictions(predictions, task)
        
        # Save results if requested
        if save_results:
            output_dir = Path(self.eval_args.output_dir or "ruler_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            pred_file = output_dir / f"{task}_{max_seq_length}_predictions.jsonl"
            with open(pred_file, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')
            
            # Save results
            result_file = output_dir / f"{task}_{max_seq_length}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"[RULER] Results saved to {output_dir}")
        
        # Clean up temporary data
        if data_path.parent.name.startswith("ruler_eval_"):
            import shutil
            shutil.rmtree(data_path.parent)
        
        return results


def run_ruler_eval() -> None:
    """Main entry point for RULER evaluation."""
    # Parse evaluation arguments
    model_args, data_args, eval_args, finetuning_args = get_eval_args()
    
    # Initialize evaluator
    evaluator = RULEREvaluator()
    
    # Get RULER tasks to run
    ruler_tasks = eval_args.ruler_tasks or ["niah_single_1"]
    context_lengths = eval_args.ruler_context_lengths or [4096]
    num_samples = eval_args.ruler_num_samples or 10
    
    # Run evaluations
    all_results = {}
    
    for context_length in context_lengths:
        for task in ruler_tasks:
            print(f"\n[RULER] Evaluating {task} at {context_length} tokens")
            
            results = evaluator.run_ruler_task(
                task=task,
                max_seq_length=context_length,
                num_samples=num_samples,
                save_results=True
            )
            
            all_results[f"{task}_{context_length}"] = results
            
            print(f"[RULER] Task: {task}, Length: {context_length}")
            print(f"[RULER] Score: {results.get('score', 'N/A'):.4f}")
            print(f"[RULER] Null predictions: {results.get('null_predictions', 'N/A')}")
    
    # Save summary
    output_dir = Path(eval_args.output_dir or "ruler_results")
    summary_file = output_dir / "ruler_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[RULER] Evaluation complete. Summary saved to {summary_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("RULER Evaluation Summary")
    print("="*70)
    print(f"{'Task':<30} {'Length':<10} {'Score':<10} {'Nulls':<10}")
    print("-"*70)
    
    for key, result in all_results.items():
        task, length = key.rsplit('_', 1)
        score = result.get('score', 0)
        nulls = result.get('null_predictions', 'N/A')
        print(f"{task:<30} {length:<10} {score:<10.4f} {nulls:<10}")
    
    print("="*70)