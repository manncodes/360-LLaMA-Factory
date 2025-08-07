"""Minimal needle-in-haystack evaluation implementation."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field


@dataclass
class NeedleConfig:
    """Configuration for needle-in-haystack evaluation."""
    
    # Needle configuration
    needle_text: str = "The secret key is: BENCHMARK_SUCCESS_42"
    retrieval_question: str = "What is the secret key mentioned in the document?"
    
    # Context configuration
    context_lengths: List[int] = field(default_factory=lambda: [2048, 4096, 8192])
    depth_percents: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Sampling
    samples_per_needle: int = 1
    
    # Dataset
    haystack_source: str = "paulgraham"  # or custom path
    
    # Output
    save_results: bool = True
    results_dir: str = "./results"


class NeedleInHaystackEvaluator:
    """Minimal needle-in-haystack evaluator."""
    
    def __init__(self, config: NeedleConfig):
        self.config = config
        self.haystack_texts = []
        self.results = []
        
    def load_haystack_data(self):
        """Load haystack data from PaulGraham essays or custom source."""
        if self.config.haystack_source == "paulgraham":
            # Use existing PaulGraham essays
            essays_dir = Path(__file__).parent.parent.parent / "LLMTest_NeedleInAHaystack" / "needlehaystack" / "PaulGrahamEssays"
            
            if essays_dir.exists():
                for essay_file in essays_dir.glob("*.txt"):
                    with open(essay_file, 'r', encoding='utf-8') as f:
                        self.haystack_texts.append(f.read())
                print(f"Loaded {len(self.haystack_texts)} PaulGraham essays")
            else:
                # Fallback to generic text
                print("PaulGraham essays not found, using generic haystack")
                self.haystack_texts = [self._generate_generic_haystack()]
        else:
            # Load from custom path
            custom_path = Path(self.config.haystack_source)
            if custom_path.is_file():
                with open(custom_path, 'r', encoding='utf-8') as f:
                    self.haystack_texts = [f.read()]
            elif custom_path.is_dir():
                for text_file in custom_path.glob("*.txt"):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        self.haystack_texts.append(f.read())
            else:
                raise ValueError(f"Invalid haystack source: {self.config.haystack_source}")
                
        if not self.haystack_texts:
            self.haystack_texts = [self._generate_generic_haystack()]
            
    def _generate_generic_haystack(self) -> str:
        """Generate generic haystack text."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "In the realm of artificial intelligence, language models have revolutionized text processing.",
            "Machine learning algorithms continue to advance at an unprecedented pace.",
            "Natural language processing enables computers to understand human communication.",
            "Deep learning neural networks form the backbone of modern AI systems.",
            "Transformer architectures have become the standard for language modeling.",
            "Attention mechanisms allow models to focus on relevant parts of input sequences.",
            "Pre-training on large corpora enables models to learn general language patterns.",
            "Fine-tuning adapts pre-trained models to specific downstream tasks.",
            "Tokenization breaks text into manageable units for processing.",
        ]
        # Repeat and shuffle to create a large haystack
        haystack = []
        for _ in range(1000):
            haystack.extend(sentences)
            random.shuffle(sentences)
        return " ".join(haystack)
        
    def create_context_with_needle(self, target_tokens: int, depth_percent: float) -> Tuple[str, int]:
        """Create a context with the needle inserted at the specified depth."""
        # Combine haystack texts to create base context
        haystack = " ".join(self.haystack_texts)
        
        # Simple token estimation (can be replaced with actual tokenizer)
        words_per_token = 0.75  # Rough estimate
        target_words = int(target_tokens * words_per_token)
        
        haystack_words = haystack.split()
        
        # Ensure we have enough words
        while len(haystack_words) < target_words:
            haystack_words.extend(haystack.split())
            
        # Trim to target length
        haystack_words = haystack_words[:target_words]
        
        # Calculate insertion position
        insertion_pos = int(len(haystack_words) * depth_percent)
        
        # Insert needle
        needle_words = self.config.needle_text.split()
        context_words = haystack_words[:insertion_pos] + needle_words + haystack_words[insertion_pos:]
        
        # Trim if necessary (accounting for needle)
        if len(context_words) > target_words:
            context_words = context_words[:target_words]
            
        context = " ".join(context_words)
        actual_tokens = len(context_words) // words_per_token  # Rough estimate
        
        return context, int(actual_tokens)
        
    def generate_test_cases(self) -> List[Dict]:
        """Generate all test cases."""
        test_cases = []
        
        for context_length in self.config.context_lengths:
            for depth_percent in self.config.depth_percents:
                for sample_idx in range(self.config.samples_per_needle):
                    context, actual_tokens = self.create_context_with_needle(
                        context_length, depth_percent
                    )
                    
                    test_case = {
                        "context": context,
                        "question": self.config.retrieval_question,
                        "needle": self.config.needle_text,
                        "expected_answer": self.config.needle_text,
                        "context_length": context_length,
                        "actual_tokens": actual_tokens,
                        "depth_percent": depth_percent,
                        "sample_idx": sample_idx
                    }
                    test_cases.append(test_case)
                    
        return test_cases
        
    def evaluate_model(self, model, tokenizer, test_cases: List[Dict], generation_config: Optional[Dict] = None) -> List[Dict]:
        """Evaluate model on test cases with configurable generation."""
        results = []
        
        # Default generation config
        gen_config = {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        # Override with provided config
        if generation_config:
            gen_config.update(generation_config)
        
        for test_case in tqdm(test_cases, desc="Evaluating"):
            # Format prompt
            prompt = f"{test_case['context']}\n\nQuestion: {test_case['question']}\nAnswer:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=test_case['context_length'])
            
            # Move inputs to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with config
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_config
                )
                
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Check if needle was found
            needle_found = self.config.needle_text.lower() in response.lower()
            
            result = {
                **test_case,
                "input_prompt": prompt,  # Save the full input
                "response": response,
                "needle_found": needle_found,
                "score": 1.0 if needle_found else 0.0,
                "input_tokens": inputs['input_ids'].shape[1],  # Token count
                "output_tokens": len(outputs[0]) - inputs['input_ids'].shape[1]  # Generated tokens
            }
            results.append(result)
            
        return results
        
    def run_evaluation(self, model, tokenizer, generation_config: Optional[Dict] = None) -> Dict:
        """Run complete evaluation with optional generation config."""
        # Load haystack data
        self.load_haystack_data()
        
        # Generate test cases
        test_cases = self.generate_test_cases()
        print(f"Generated {len(test_cases)} test cases")
        
        # Evaluate model with generation config
        results = self.evaluate_model(model, tokenizer, test_cases, generation_config)
        
        # Aggregate results
        summary = self.aggregate_results(results)
        
        # Save results
        if self.config.save_results:
            self.save_results(results, summary)
            
        return summary
        
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate evaluation results."""
        summary = {
            "total_samples": len(results),
            "overall_accuracy": np.mean([r["score"] for r in results]),
            "by_context_length": {},
            "by_depth": {}
        }
        
        # Aggregate by context length
        for context_length in self.config.context_lengths:
            context_results = [r for r in results if r["context_length"] == context_length]
            if context_results:
                summary["by_context_length"][context_length] = {
                    "accuracy": np.mean([r["score"] for r in context_results]),
                    "samples": len(context_results)
                }
                
        # Aggregate by depth
        for depth in self.config.depth_percents:
            depth_results = [r for r in results if r["depth_percent"] == depth]
            if depth_results:
                summary["by_depth"][depth] = {
                    "accuracy": np.mean([r["score"] for r in depth_results]),
                    "samples": len(depth_results)
                }
                
        return summary
        
    def save_results(self, results: List[Dict], summary: Dict):
        """Save evaluation results with inputs and outputs."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results with everything
        with open(results_dir / "detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save summary
        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save inputs and outputs separately for easier analysis
        inputs_outputs = []
        for r in results:
            inputs_outputs.append({
                "context_length": r["context_length"],
                "depth_percent": r["depth_percent"],
                "sample_idx": r.get("sample_idx", 0),
                "input_prompt": r.get("input_prompt", ""),
                "response": r["response"],
                "needle_found": r["needle_found"],
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0)
            })
        
        with open(results_dir / "inputs_outputs.json", 'w') as f:
            json.dump(inputs_outputs, f, indent=2)
            
        # Save a readable text version
        with open(results_dir / "generations.txt", 'w') as f:
            for i, r in enumerate(results):
                f.write(f"{'='*80}\n")
                f.write(f"Sample {i+1}: Context={r['context_length']}, Depth={r['depth_percent']}\n")
                f.write(f"Needle Found: {r['needle_found']}\n")
                f.write(f"Input Tokens: {r.get('input_tokens', 0)}, Output Tokens: {r.get('output_tokens', 0)}\n")
                f.write(f"-"*40 + " PROMPT " + "-"*40 + "\n")
                f.write(r.get("input_prompt", "")[:500] + "...\n" if len(r.get("input_prompt", "")) > 500 else r.get("input_prompt", "") + "\n")
                f.write(f"-"*40 + " RESPONSE " + "-"*38 + "\n")
                f.write(r["response"] + "\n")
                f.write("\n")
            
        print(f"Results saved to {results_dir}")
        print(f"  - detailed_results.json: Complete results with all data")
        print(f"  - inputs_outputs.json: Focused on prompts and generations")
        print(f"  - generations.txt: Human-readable format")


# Add torch import for model evaluation
try:
    import torch
except ImportError:
    print("Warning: PyTorch not imported. Model evaluation will not work.")
    torch = None