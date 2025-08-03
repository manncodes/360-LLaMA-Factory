"""
LongBench v2 Integration for LlamaFactory

LongBench v2 is a benchmark designed to assess the ability of LLMs to handle
long-context problems requiring deep understanding and reasoning across real-world multitasks.

Key features:
- 503 challenging multiple-choice questions
- Context lengths from 8k to 2M words
- Six task categories: single-doc QA, multi-doc QA, long ICL, dialogue, code, structured data
- Multiple evaluation modes: standard, CoT, no-context, RAG

Reference: https://github.com/THUDM/LongBench
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from ..hparams import EvaluationArguments, ModelArguments, DataArguments, FinetuningArguments
from ..model import load_model, load_tokenizer
from ..data import get_template_and_fix_tokenizer
from ..extras.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LongBenchConfig:
    """Configuration for LongBench evaluation."""
    mode: str = "standard"  # standard, cot, no_context, rag
    rag_top_k: int = 0  # Number of retrieved chunks for RAG mode
    temperature: float = 0.1
    max_new_tokens: int = 128
    cot_max_new_tokens: int = 1024
    truncation_strategy: str = "middle"  # middle truncation when exceeding max length
    batch_size: int = 1
    save_details: bool = True


class LongBenchEvaluator:
    """
    LongBench v2 evaluator integrated with LlamaFactory.
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
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_args.model_name_or_path}")
        self.tokenizer = load_tokenizer(model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"  # For batch generation
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.model = load_model(self.tokenizer, model_args, finetuning_args)
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Set max length - consider model's actual capacity
        self.max_length = getattr(self.eval_args, 'longbench_max_length', 131072)
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        if self.max_length > model_max_length:
            logger.warning(f"Reducing max_length from {self.max_length} to model capacity {model_max_length}")
            self.max_length = model_max_length
        
    def _load_config(self) -> LongBenchConfig:
        """Load LongBench configuration from evaluation arguments."""
        config = LongBenchConfig()
        
        # Update from eval_args if available
        if hasattr(self.eval_args, 'longbench_mode'):
            config.mode = self.eval_args.longbench_mode
        if hasattr(self.eval_args, 'longbench_rag_top_k'):
            config.rag_top_k = self.eval_args.longbench_rag_top_k
        if hasattr(self.eval_args, 'longbench_temperature'):
            config.temperature = self.eval_args.longbench_temperature
        if hasattr(self.eval_args, 'longbench_max_new_tokens'):
            config.max_new_tokens = self.eval_args.longbench_max_new_tokens
        if hasattr(self.eval_args, 'longbench_cot_max_new_tokens'):
            config.cot_max_new_tokens = self.eval_args.longbench_cot_max_new_tokens
        if hasattr(self.eval_args, 'batch_size'):
            config.batch_size = self.eval_args.batch_size
            
        return config
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load LongBench prompt templates."""
        prompts = {}
        
        # Use configurable repo path if provided
        if hasattr(self.eval_args, 'longbench_repo_path') and self.eval_args.longbench_repo_path:
            longbench_path = Path(self.eval_args.longbench_repo_path)
        else:
            # Default path
            longbench_path = Path(__file__).parent.parent.parent.parent / "evaluation" / "longbench" / "LongBench"
        
        prompt_dir = longbench_path / "prompts"
        
        # Also check in our local prompts directory as fallback
        local_prompt_dir = Path(__file__).parent.parent.parent.parent / "evaluation" / "longbench" / "prompts"
        
        prompt_files = {
            'standard': '0shot.txt',
            'cot': '0shot_cot.txt',
            'cot_ans': '0shot_cot_ans.txt',
            'no_context': '0shot_no_context.txt',
            'rag': '0shot_rag.txt',
        }
        
        for key, filename in prompt_files.items():
            # Try the configured repo path first
            prompt_path = prompt_dir / filename
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read()
            else:
                # Try local fallback
                local_path = local_prompt_dir / filename
                if local_path.exists():
                    with open(local_path, 'r', encoding='utf-8') as f:
                        prompts[key] = f.read()
                else:
                    logger.warning(f"Prompt file not found in {prompt_path} or {local_path}")
                
        # Fallback prompts if files not found
        if 'standard' not in prompts:
            prompts['standard'] = """Given the following document and question, choose the correct answer from the options provided.

Document:
$DOC$

Question: $Q$

Options:
A) $C_A$
B) $C_B$
C) $C_C$
D) $C_D$

The correct answer is ("""
        
        return prompts
    
    def _load_longbench_dataset(self):
        """Load LongBench v2 dataset with priority for local symlink."""
        # Priority order for finding the dataset:
        # 1. Symlinked dataset in evaluation/longbench/LongBench-v2
        # 2. Configured repo path
        # 3. HuggingFace datasets API
        
        dataset_paths = []
        
        # Check for symlinked dataset first
        symlink_path = Path(__file__).parent.parent.parent.parent / "evaluation" / "longbench" / "LongBench-v2"
        if symlink_path.exists():
            dataset_paths.append(symlink_path)
            logger.info(f"Found symlinked dataset at: {symlink_path}")
        
        # Check configured repo path
        if hasattr(self.eval_args, 'longbench_repo_path') and self.eval_args.longbench_repo_path:
            configured_path = Path(self.eval_args.longbench_repo_path)
            if configured_path.exists():
                dataset_paths.append(configured_path)
                logger.info(f"Found configured dataset at: {configured_path}")
        
        # Try to load from local paths first
        for dataset_path in dataset_paths:
            try:
                logger.info(f"Attempting to load dataset from: {dataset_path}")
                
                # Check if it's a HuggingFace dataset directory structure
                if (dataset_path / "data").exists() or any(dataset_path.glob("*.arrow")) or any(dataset_path.glob("*.parquet")):
                    # Load as HuggingFace dataset from local path
                    from datasets import load_from_disk, Dataset
                    
                    # Try different loading methods
                    try:
                        dataset = load_from_disk(str(dataset_path))
                        if hasattr(dataset, 'train'):
                            dataset = dataset['train']
                        logger.info(f"✓ Successfully loaded dataset from disk: {dataset_path}")
                        return dataset
                    except:
                        # Try loading as dataset from path
                        dataset = load_dataset(str(dataset_path), split='train')
                        logger.info(f"✓ Successfully loaded dataset from path: {dataset_path}")
                        return dataset
                
                # Check for specific LongBench-v2 structure
                elif any(dataset_path.glob("*.jsonl")) or any(dataset_path.glob("*.json")):
                    # Try to load JSON/JSONL files
                    json_files = list(dataset_path.glob("*.jsonl")) + list(dataset_path.glob("*.json"))
                    if json_files:
                        logger.info(f"Found data files: {[f.name for f in json_files]}")
                        dataset = load_dataset('json', data_files=str(json_files[0]), split='train')
                        logger.info(f"✓ Successfully loaded dataset from JSON: {dataset_path}")
                        return dataset
                        
            except Exception as e:
                logger.warning(f"Failed to load dataset from {dataset_path}: {e}")
                continue
        
        # Check if we should force local loading
        force_local = getattr(self.eval_args, 'longbench_force_local', True)
        
        if force_local and dataset_paths:
            # If force_local is True and we found local paths, don't fallback to HF hub
            raise RuntimeError(f"Could not load LongBench v2 dataset from local paths: {dataset_paths}. "
                             f"Set longbench_force_local: false to use HuggingFace hub as fallback.")
        
        # Fallback to HuggingFace hub only if force_local is False or no local paths found
        if not force_local:
            logger.info("Loading from HuggingFace hub as fallback...")
            try:
                dataset = load_dataset('THUDM/LongBench-v2', split='train')
                logger.info("✓ Successfully loaded dataset from HuggingFace hub")
                return dataset
            except Exception as e:
                logger.error(f"Failed to load dataset from HuggingFace hub: {e}")
                raise RuntimeError("Could not load LongBench v2 dataset from any source")
        else:
            raise RuntimeError("No local LongBench v2 dataset found and longbench_force_local is True. "
                             "Please run the setup script or set longbench_force_local: false")
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max tokens using middle truncation."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
            
        # Middle truncation strategy
        half_max = max_tokens // 2
        truncated_tokens = tokens[:half_max] + tokens[-half_max:]
        
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def _prepare_prompt(self, item: Dict[str, Any]) -> str:
        """Prepare prompt based on evaluation mode."""
        context = item['context']
        
        # Select template based on mode
        if self.config.mode == 'no_context':
            template = self.prompts.get('no_context', self.prompts['standard'])
        elif self.config.mode == 'cot':
            template = self.prompts.get('cot', self.prompts['standard'])
        elif self.config.mode == 'rag' and self.config.rag_top_k > 0:
            template = self.prompts.get('rag', self.prompts['standard'])
            # Note: RAG retrieval not implemented in this version
            logger.warning("RAG mode selected but retrieval not implemented. Using standard mode.")
            template = self.prompts['standard']
        else:
            template = self.prompts['standard']
        
        # Truncate context if needed
        if self.config.mode != 'no_context':
            # First, create prompt without context to measure overhead
            temp_prompt = template.replace('$DOC$', '')
            temp_prompt = temp_prompt.replace('$Q$', item['question'].strip())
            temp_prompt = temp_prompt.replace('$C_A$', item['choice_A'].strip())
            temp_prompt = temp_prompt.replace('$C_B$', item['choice_B'].strip())
            temp_prompt = temp_prompt.replace('$C_C$', item['choice_C'].strip())
            temp_prompt = temp_prompt.replace('$C_D$', item['choice_D'].strip())
            
            # Calculate overhead tokens (everything except context)
            overhead_tokens = len(self.tokenizer.encode(temp_prompt, add_special_tokens=False))
            
            # Reserve space for generation and safety margin
            reserved_for_generation = 200  # Space for answer generation
            safety_margin = 100  # Extra safety buffer
            
            # Calculate maximum tokens available for context
            max_context_tokens = self.max_length - overhead_tokens - reserved_for_generation - safety_margin
            
            # Ensure we have at least some context
            if max_context_tokens < 100:
                logger.warning(f"Very little space for context: {max_context_tokens} tokens")
                max_context_tokens = 100
            
            context = self._truncate_text(context, max_context_tokens)
        
        # Fill template
        prompt = template.replace('$DOC$', context.strip())
        prompt = prompt.replace('$Q$', item['question'].strip())
        prompt = prompt.replace('$C_A$', item['choice_A'].strip())
        prompt = prompt.replace('$C_B$', item['choice_B'].strip())
        prompt = prompt.replace('$C_C$', item['choice_C'].strip())
        prompt = prompt.replace('$C_D$', item['choice_D'].strip())
        
        # Final safety check - truncate entire prompt if still too long
        final_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        max_prompt_tokens = self.max_length - 150  # Reserve space for generation
        
        if final_tokens > max_prompt_tokens:
            logger.warning(f"Final prompt too long ({final_tokens} tokens), truncating to {max_prompt_tokens}")
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            truncated_tokens = prompt_tokens[:max_prompt_tokens]
            prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return prompt
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer choice from model response."""
        response = response.replace('*', '').strip()
        
        # Try different patterns
        patterns = [
            r'The correct answer is \(([A-D])\)',
            r'The correct answer is ([A-D])',
            r'Answer: \(([A-D])\)',
            r'Answer: ([A-D])',
            r'\(([A-D])\)',
            r'^([A-D])$',
            r'^([A-D])[).]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Check if response starts with a letter
        if response and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
            
        return None
    
    @torch.inference_mode()
    def _generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        # Encode prompts
        messages_batch = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            messages_batch.append(messages)
        
        # Batch encode with length validation
        encoded = []
        for i, messages in enumerate(messages_batch):
            try:
                input_ids, _ = self.template.encode_oneturn(
                    tokenizer=self.tokenizer,
                    messages=messages
                )
                
                # More aggressive truncation
                max_input_length = self.max_length - 200  # Reserve more space for generation
                
                if len(input_ids) > max_input_length:
                    logger.warning(f"Input {i} too long ({len(input_ids)} tokens), truncating to {max_input_length}")
                    input_ids = input_ids[:max_input_length]
                
                # Validate input_ids is not empty
                if len(input_ids) == 0:
                    logger.error(f"Empty input_ids for prompt {i}, using fallback")
                    # Create minimal fallback input
                    fallback_text = "Answer: A"
                    input_ids = self.tokenizer.encode(fallback_text, add_special_tokens=True)
                
                encoded.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                
            except Exception as e:
                logger.error(f"Error encoding prompt {i}: {e}")
                # Create fallback encoding
                fallback_text = "Answer: A"
                input_ids = self.tokenizer.encode(fallback_text, add_special_tokens=True)
                encoded.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        
        # Pad batch
        batch = self.tokenizer.pad(
            encoded,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        max_new_tokens = (
            self.config.cot_max_new_tokens if self.config.mode == 'cot'
            else self.config.max_new_tokens
        )
        
        generation_kwargs = {
            **batch,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        # Only add temperature/sampling if temperature > 0
        if self.config.temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": self.config.temperature,
            })
        else:
            generation_kwargs.update({
                "do_sample": False,
            })
        
        outputs = self.model.generate(**generation_kwargs)
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            output = output[len(encoded[i]["input_ids"]):]
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def _evaluate_batch(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of items."""
        results = []
        
        # Prepare prompts
        prompts = [self._prepare_prompt(item) for item in batch_items]
        
        # Generate responses
        responses = self._generate(prompts)
        
        # Process results
        for item, response in zip(batch_items, responses):
            result = item.copy()
            result['response'] = response
            
            # Handle CoT mode
            if self.config.mode == 'cot':
                # Extract reasoning
                result['response_cot'] = response
                
                # Generate final answer
                cot_template = self.prompts.get('cot_ans', '')
                if cot_template:
                    final_prompt = cot_template.replace('$COT$', response)
                    final_prompt = self._prepare_prompt(item).replace(
                        self.prompts['cot'], final_prompt
                    )
                    final_response = self._generate([final_prompt])[0]
                    result['response'] = final_response
                    response = final_response
            
            # Extract answer
            pred = self._extract_answer(response)
            result['pred'] = pred
            result['judge'] = pred == item['answer'] if pred else False
            
            # Truncate context in result to save space
            if self.config.save_details:
                result['context_preview'] = item['context'][:1000] + '...'
            
            results.append(result)
        
        return results
    
    def evaluate(self) -> Dict[str, Any]:
        """Run LongBench evaluation."""
        logger.info("Starting LongBench v2 evaluation...")
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Batch size: {self.config.batch_size}")
        
        # Load dataset
        logger.info("Loading LongBench v2 dataset...")
        dataset = self._load_longbench_dataset()
        
        # Filter by specific domains if requested
        if hasattr(self.eval_args, 'longbench_domains') and self.eval_args.longbench_domains:
            domains = self.eval_args.longbench_domains.split(',')
            dataset = dataset.filter(lambda x: x['domain'] in domains)
            logger.info(f"Filtered to domains: {domains}")
        
        # Limit samples for testing
        if hasattr(self.eval_args, 'longbench_max_samples') and self.eval_args.longbench_max_samples:
            dataset = dataset.select(range(min(len(dataset), self.eval_args.longbench_max_samples)))
            logger.info(f"Limited to {len(dataset)} samples")
        
        # Prepare data
        all_items = []
        for item in dataset:
            all_items.append({
                "_id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "question": item["question"],
                "choice_A": item["choice_A"],
                "choice_B": item["choice_B"],
                "choice_C": item["choice_C"],
                "choice_D": item["choice_D"],
                "answer": item["answer"],
                "context": item["context"]
            })
        
        # Evaluate in batches
        all_results = []
        for i in tqdm(range(0, len(all_items), self.config.batch_size), desc="Evaluating"):
            batch = all_items[i:i + self.config.batch_size]
            batch_results = self._evaluate_batch(batch)
            all_results.extend(batch_results)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_results)
        
        # Save results
        output_dir = self.eval_args.save_dir or "saves/longbench_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        if self.config.save_details:
            detail_file = os.path.join(output_dir, f"longbench_{self.config.mode}_details.jsonl")
            with open(detail_file, 'w', encoding='utf-8') as f:
                for result in all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"Detailed results saved to: {detail_file}")
        
        # Save summary
        summary = {
            "config": {
                "model": self.model_args.model_name_or_path,
                "mode": self.config.mode,
                "max_length": self.max_length,
                "temperature": self.config.temperature,
                "batch_size": self.config.batch_size,
            },
            "metrics": metrics,
            "total_samples": len(all_results),
        }
        
        summary_file = os.path.join(output_dir, f"longbench_{self.config.mode}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("LongBench v2 Evaluation Results")
        logger.info("="*50)
        logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
        logger.info(f"Total Examples: {metrics['overall']['total']}")
        logger.info("\nAccuracy by Domain:")
        for domain, domain_metrics in metrics['by_domain'].items():
            logger.info(f"  {domain}: {domain_metrics['accuracy']:.2%} ({domain_metrics['total']} examples)")
        logger.info("\nAccuracy by Difficulty:")
        for difficulty, diff_metrics in metrics['by_difficulty'].items():
            logger.info(f"  {difficulty}: {diff_metrics['accuracy']:.2%} ({diff_metrics['total']} examples)")
        
        return summary
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        metrics = {
            'overall': {'correct': 0, 'total': 0, 'accuracy': 0.0},
            'by_domain': {},
            'by_sub_domain': {},
            'by_difficulty': {},
            'by_length': {},
        }
        
        for result in results:
            # Overall
            metrics['overall']['total'] += 1
            if result['judge']:
                metrics['overall']['correct'] += 1
            
            # By domain
            domain = result['domain']
            if domain not in metrics['by_domain']:
                metrics['by_domain'][domain] = {'correct': 0, 'total': 0}
            metrics['by_domain'][domain]['total'] += 1
            if result['judge']:
                metrics['by_domain'][domain]['correct'] += 1
            
            # By sub-domain
            sub_domain = result['sub_domain']
            if sub_domain not in metrics['by_sub_domain']:
                metrics['by_sub_domain'][sub_domain] = {'correct': 0, 'total': 0}
            metrics['by_sub_domain'][sub_domain]['total'] += 1
            if result['judge']:
                metrics['by_sub_domain'][sub_domain]['correct'] += 1
            
            # By difficulty
            difficulty = result['difficulty']
            if difficulty not in metrics['by_difficulty']:
                metrics['by_difficulty'][difficulty] = {'correct': 0, 'total': 0}
            metrics['by_difficulty'][difficulty]['total'] += 1
            if result['judge']:
                metrics['by_difficulty'][difficulty]['correct'] += 1
            
            # By length
            length = result['length']
            if length not in metrics['by_length']:
                metrics['by_length'][length] = {'correct': 0, 'total': 0}
            metrics['by_length'][length]['total'] += 1
            if result['judge']:
                metrics['by_length'][length]['correct'] += 1
        
        # Calculate accuracies
        if metrics['overall']['total'] > 0:
            metrics['overall']['accuracy'] = metrics['overall']['correct'] / metrics['overall']['total']
        
        for category_metrics in [metrics['by_domain'], metrics['by_sub_domain'], 
                                metrics['by_difficulty'], metrics['by_length']]:
            for key, values in category_metrics.items():
                if values['total'] > 0:
                    values['accuracy'] = values['correct'] / values['total']
                else:
                    values['accuracy'] = 0.0
        
        return metrics


def run_longbench_evaluation(
    model_args: ModelArguments,
    data_args: DataArguments,
    eval_args: EvaluationArguments,
    finetuning_args: FinetuningArguments,
) -> Dict[str, Any]:
    """Main entry point for LongBench evaluation."""
    evaluator = LongBenchEvaluator(model_args, data_args, eval_args, finetuning_args)
    return evaluator.evaluate()