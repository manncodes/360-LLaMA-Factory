#!/usr/bin/env python3
"""
LongBench-v2 Evaluator integrated with 360-LLaMA-Factory
Provides comprehensive long-context evaluation using LongBench-v2 dataset
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template


class LongBenchV2Evaluator:
    """LongBench-v2 Evaluator integrated with 360-LLaMA-Factory"""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        """Initialize evaluator with model and tokenizer"""
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        
        # Results tracking
        self.results = []
        self.save_dir = Path(getattr(self.eval_args, 'save_dir', 'saves/longbench_v2_results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_longbench_v2_data(self) -> List[Dict]:
        """Load LongBench-v2 dataset from Hugging Face"""
        print("Loading LongBench-v2 dataset...")
        try:
            dataset = load_dataset('THUDM/LongBench-v2', split='train')
            data = list(dataset)
            print(f"Loaded {len(data)} samples from LongBench-v2")
            
            # Apply eval_max_samples limit if specified
            eval_max_samples = getattr(self.eval_args, 'eval_max_samples', None)
            if eval_max_samples and len(data) > eval_max_samples:
                data = data[:eval_max_samples]
                print(f"Limited to {eval_max_samples} samples for evaluation")
                
            return data
        except Exception as e:
            print(f"Error loading LongBench-v2 dataset: {e}")
            return []
    
    def format_question(self, sample: Dict) -> str:
        """Format a LongBench-v2 sample into a prompt"""
        context = sample["context"]
        question = sample["question"]
        choices = [
            f"A. {sample['choice_A']}",
            f"B. {sample['choice_B']}",
            f"C. {sample['choice_C']}",
            f"D. {sample['choice_D']}"
        ]
        
        prompt = f"""Based on the following context, answer the question by selecting the correct option.

Context:
{context}

Question: {question}

Options:
{chr(10).join(choices)}

Answer: """
        
        return prompt
    
    @torch.inference_mode()
    def generate_response(self, prompt: str) -> str:
        """Generate model response for given prompt"""
        try:
            # Use direct prompt without complex templating for now
            # This gives us working baseline results
            prompt_formatted = prompt
            
            # Tokenize
            inputs = self.tokenizer(
                prompt_formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.data_args.cutoff_len
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Only need A/B/C/D
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    use_cache=True,
                    temperature=0.0
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            if outputs.shape[1] > input_length:
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                return response.strip()
            
            return ""
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def extract_answer(self, response: str) -> str:
        """Extract answer choice (A/B/C/D) from model response"""
        # Clean response
        response = response.strip().upper()
        
        # Direct match
        if response in ["A", "B", "C", "D"]:
            return response
        
        # Pattern matching for common formats
        patterns = [
            r'^([ABCD])',  # Answer starts with A/B/C/D
            r'ANSWER:\s*([ABCD])',  # "Answer: A" format
            r'OPTION\s*([ABCD])',   # "Option A" format
            r'CHOICE\s*([ABCD])',   # "Choice A" format
            r'([ABCD])\.',          # "A." format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # If no clear answer found, return first character if it's A/B/C/D
        if response and response[0] in "ABCD":
            return response[0]
        
        return "UNKNOWN"
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample"""
        prompt = self.format_question(sample)
        
        # Check context length
        context_length = len(self.tokenizer.encode(prompt))
        
        response = self.generate_response(prompt)
        predicted_answer = self.extract_answer(response)
        correct_answer = sample["answer"]
        
        is_correct = predicted_answer == correct_answer
        
        result = {
            "id": sample["_id"],
            "domain": sample["domain"],
            "sub_domain": sample["sub_domain"],
            "difficulty": sample["difficulty"],
            "length": sample["length"],
            "context_length": context_length,
            "question": sample["question"][:100] + "..." if len(sample["question"]) > 100 else sample["question"],
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "raw_response": response[:200],  # Truncate for storage
            "is_correct": is_correct
        }
        
        return result
    
    def run_evaluation(self) -> Dict:
        """Run complete LongBench-v2 evaluation"""
        print("=" * 60)
        print("LongBench-v2 Evaluation")
        print("=" * 60)
        print(f"Model: {self.model_args.model_name_or_path}")
        print(f"Max context length: {self.data_args.cutoff_len}")
        print(f"Results will be saved to: {self.save_dir}")
        print("=" * 60)
        
        # Load dataset
        data = self.load_longbench_v2_data()
        if not data:
            print("Failed to load dataset. Exiting.")
            return {}
        
        # Track stats
        total_samples = len(data)
        correct_count = 0
        domain_stats = {}
        difficulty_stats = {"easy": {"correct": 0, "total": 0}, "hard": {"correct": 0, "total": 0}}
        length_stats = {"short": {"correct": 0, "total": 0}, "medium": {"correct": 0, "total": 0}, "long": {"correct": 0, "total": 0}}
        
        # Run evaluation
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(data, desc="Evaluating samples")):
            result = self.evaluate_sample(sample)
            self.results.append(result)
            
            # Update stats
            if result["is_correct"]:
                correct_count += 1
            
            # Domain stats
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}
            domain_stats[domain]["total"] += 1
            if result["is_correct"]:
                domain_stats[domain]["correct"] += 1
            
            # Difficulty stats
            difficulty = result["difficulty"]
            if difficulty in difficulty_stats:
                difficulty_stats[difficulty]["total"] += 1
                if result["is_correct"]:
                    difficulty_stats[difficulty]["correct"] += 1
            
            # Length stats
            length = result["length"]
            if length in length_stats:
                length_stats[length]["total"] += 1
                if result["is_correct"]:
                    length_stats[length]["correct"] += 1
            
            # Save intermediate results every 10 samples
            if (i + 1) % 10 == 0:
                self.save_results()
        
        duration = time.time() - start_time
        
        # Calculate final accuracy
        overall_accuracy = correct_count / total_samples if total_samples > 0 else 0
        
        # Compile summary
        summary = {
            "model": self.model_args.model_name_or_path,
            "total_samples": total_samples,
            "correct_count": correct_count,
            "overall_accuracy": overall_accuracy,
            "duration_seconds": duration,
            "domain_breakdown": {
                domain: {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                    "correct": stats["correct"],
                    "total": stats["total"]
                } for domain, stats in domain_stats.items()
            },
            "difficulty_breakdown": {
                difficulty: {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                    "correct": stats["correct"],
                    "total": stats["total"]
                } for difficulty, stats in difficulty_stats.items()
            },
            "length_breakdown": {
                length: {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                    "correct": stats["correct"],
                    "total": stats["total"]
                } for length, stats in length_stats.items()
            }
        }
        
        # Save final results
        self.save_results(summary)
        
        # Print results
        self.print_results(summary)
        
        return summary
    
    def save_results(self, summary: Optional[Dict] = None):
        """Save evaluation results"""
        # Save detailed results
        results_file = self.save_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary if provided
        if summary:
            summary_file = self.save_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
    
    def print_results(self, summary: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 60)
        print("LongBench-v2 Evaluation Results")
        print("=" * 60)
        print(f"Model: {summary['model']}")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Correct: {summary['correct_count']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f} ({summary['overall_accuracy']*100:.1f}%)")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        
        print("\n📊 Breakdown by Domain:")
        for domain, stats in summary['domain_breakdown'].items():
            print(f"  {domain}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        print("\n📊 Breakdown by Difficulty:")
        for difficulty, stats in summary['difficulty_breakdown'].items():
            print(f"  {difficulty}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        print("\n📊 Breakdown by Length:")
        for length, stats in summary['length_breakdown'].items():
            print(f"  {length}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        print(f"\n💾 Results saved to: {self.save_dir}")
        print("=" * 60)


def run_longbench_v2_eval():
    """Main entry point for LongBench-v2 evaluation"""
    evaluator = LongBenchV2Evaluator()
    return evaluator.run_evaluation()


if __name__ == "__main__":
    run_longbench_v2_eval()