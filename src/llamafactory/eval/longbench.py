"""
Minimal LongBench v2 evaluation for LlamaFactory.
Based on the official LongBench evaluation pipeline.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..model import load_model, load_tokenizer
from ..data import get_template_and_fix_tokenizer
from ..extras.logging import get_logger

logger = get_logger(__name__)


class LongBenchEvaluator:
    """Minimal LongBench v2 evaluator."""
    
    def __init__(self, model_args, data_args, eval_args, finetuning_args):
        self.model_args = model_args
        self.data_args = data_args
        self.eval_args = eval_args
        self.finetuning_args = finetuning_args
        
        # Load model and tokenizer
        self.tokenizer = load_tokenizer(model_args)["tokenizer"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.model = load_model(self.tokenizer, model_args, finetuning_args)
        
        # Set max length
        self.max_length = getattr(eval_args, 'longbench_max_length', 32768)
        model_max = getattr(self.model.config, 'max_position_embeddings', 32768)
        self.max_length = min(self.max_length, model_max)
        
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """Middle truncation strategy from official LongBench."""
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return context
            
        # Keep beginning and end
        half = max_tokens // 2
        truncated = tokens[:half] + tokens[-half:]
        return self.tokenizer.decode(truncated, skip_special_tokens=True)
    
    def prepare_prompt(self, item: Dict[str, Any]) -> str:
        """Prepare prompt using official template format."""
        # Simple template for now
        template = """Given the following document and question, choose the correct answer from the options provided.

Document:
$DOC$

Question: $Q$

Options:
A) $C_A$
B) $C_B$
C) $C_C$  
D) $C_D$

Answer (A/B/C/D):"""

        # Calculate available space for context
        overhead = len(self.tokenizer.encode(template, add_special_tokens=False))
        question_tokens = len(self.tokenizer.encode(item['question'], add_special_tokens=False))
        choice_tokens = sum(len(self.tokenizer.encode(item[f'choice_{c}'], add_special_tokens=False)) 
                           for c in ['A', 'B', 'C', 'D'])
        
        available = self.max_length - overhead - question_tokens - choice_tokens - 500  # safety margin
        
        # Truncate context if needed
        context = self.truncate_context(item['context'], max(available, 100))
        
        # Fill template
        prompt = template.replace('$DOC$', context)
        prompt = prompt.replace('$Q$', item['question'])
        prompt = prompt.replace('$C_A$', item['choice_A'])
        prompt = prompt.replace('$C_B$', item['choice_B'])
        prompt = prompt.replace('$C_C$', item['choice_C'])
        prompt = prompt.replace('$C_D$', item['choice_D'])
        
        return prompt
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from model response."""
        response = response.strip()
        
        # Try various patterns
        patterns = [
            r'\(([A-D])\)',
            r'Answer: ([A-D])',
            r'answer is ([A-D])',
            r'^([A-D])$',
            r'^([A-D])[).]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Check if starts with A/B/C/D
        if response and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
            
        return None
    
    @torch.inference_mode()
    def generate_answer(self, prompt: str) -> str:
        """Generate answer for a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        # Encode
        input_ids, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer,
            messages=messages
        )
        
        # Ensure not too long
        if len(input_ids) > self.max_length - 50:
            input_ids = input_ids[:self.max_length - 50]
        
        # Move to device
        input_ids = torch.tensor([input_ids], device=self.model.device)
        
        # Generate
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on LongBench v2."""
        logger.info("Starting LongBench v2 evaluation...")
        
        # Load dataset
        dataset = load_dataset('THUDM/LongBench-v2', split='train')
        
        # Limit samples if specified
        max_samples = getattr(self.eval_args, 'longbench_max_samples', None)
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
            
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        # Evaluate
        results = []
        correct = 0
        
        for item in tqdm(dataset, desc="Evaluating"):
            # Prepare prompt
            prompt = self.prepare_prompt(item)
            
            # Generate answer
            response = self.generate_answer(prompt)
            
            # Extract answer
            predicted = self.extract_answer(response)
            is_correct = predicted == item['answer'] if predicted else False
            
            if is_correct:
                correct += 1
            
            # Store result
            results.append({
                '_id': item['_id'],
                'predicted': predicted,
                'answer': item['answer'],
                'correct': is_correct,
                'response': response[:200],  # truncate for storage
            })
        
        # Calculate metrics
        accuracy = correct / len(results) if results else 0
        
        # Save results
        save_dir = self.eval_args.save_dir or "saves/longbench"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(save_dir, "predictions.jsonl"), 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Save summary
        summary = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'model': self.model_args.model_name_or_path,
            'max_length': self.max_length,
        }
        
        with open(os.path.join(save_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
        
        return summary


def run_longbench_eval(model_args, data_args, eval_args, finetuning_args):
    """Entry point for LongBench evaluation."""
    evaluator = LongBenchEvaluator(model_args, data_args, eval_args, finetuning_args)
    return evaluator.evaluate()