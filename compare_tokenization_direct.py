#!/usr/bin/env python3
"""
Direct comparison of tokenization between LlamaFactory's method and our script.
This script tokenizes the same data using both methods and compares byte-by-byte.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Import LlamaFactory's preprocessing
sys.path.insert(0, ".")
from llamafactory.data.processors.pretrain import preprocess_pretrain_dataset
from llamafactory.hparams import DataArguments

def tokenize_with_llamafactory(
    texts: List[str],
    tokenizer,
    cutoff_len: int = 2048,
    template: str = "default"
) -> Dict[str, List]:
    """Tokenize using LlamaFactory's method"""
    
    # Create data args
    data_args = DataArguments(
        cutoff_len=cutoff_len,
        template=template,
        packing=False
    )
    
    # Format as LlamaFactory expects
    examples = {
        "_prompt": [[{"content": text}] for text in texts]
    }
    
    # Use LlamaFactory's preprocessing
    result = preprocess_pretrain_dataset(examples, tokenizer, data_args)
    
    return result

def tokenize_with_our_method(
    texts: List[str],
    tokenizer,
    cutoff_len: int = 2048,
    template: str = "default"
) -> Dict[str, List]:
    """Tokenize using our pretokenization method"""
    
    # Determine EOS token
    if template == "llama3":
        eos_token = "<|end_of_text|>"
    else:
        eos_token = tokenizer.eos_token
    
    # Add EOS to texts
    text_examples = [text + eos_token for text in texts]
    
    # Add BOS for gemma
    if template == "gemma":
        text_examples = [tokenizer.bos_token + example for example in text_examples]
    
    # Tokenize
    result = tokenizer(
        text_examples,
        add_special_tokens=False,
        truncation=True,
        max_length=cutoff_len,
        padding=False
    )
    
    return result

def compare_tokenizations(tokens1: List[int], tokens2: List[int], name1: str = "Method1", name2: str = "Method2") -> Dict:
    """Compare two token sequences"""
    
    result = {
        "identical": tokens1 == tokens2,
        "len1": len(tokens1),
        "len2": len(tokens2),
        "differences": []
    }
    
    if len(tokens1) != len(tokens2):
        result["differences"].append(f"Length mismatch: {name1}={len(tokens1)}, {name2}={len(tokens2)}")
    
    min_len = min(len(tokens1), len(tokens2))
    for i in range(min_len):
        if tokens1[i] != tokens2[i]:
            result["differences"].append({
                "position": i,
                f"{name1}_token": tokens1[i],
                f"{name2}_token": tokens2[i]
            })
            if len(result["differences"]) > 10:  # Limit output
                break
    
    return result

def main():
    print("="*60)
    print("DIRECT TOKENIZATION COMPARISON TEST")
    print("="*60)
    
    # Setup - Use an open model for testing
    model_name = "gpt2"  # Open model, no authentication needed
    cutoff_len = 2048
    
    print(f"\nModel: {model_name}")
    print(f"Cutoff: {cutoff_len}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data - use local c4_demo
    print("Loading test data...")
    import json
    with open("./data/c4_demo.json", "r") as f:
        data = json.load(f)
    texts = [item["text"][:1000] if isinstance(item, dict) and "text" in item 
             else str(item)[:1000] for item in data[:5]]  # Use first 5 samples, 1000 chars each
    
    print(f"Testing with {len(texts)} samples")
    
    # Test each template
    templates = ["default", "llama3", "gemma"]
    
    all_results = {}
    
    for template in templates:
        print(f"\n{'='*40}")
        print(f"Testing template: {template}")
        print(f"{'='*40}")
        
        try:
            # Tokenize with LlamaFactory method
            print("Tokenizing with LlamaFactory method...")
            llamafactory_result = tokenize_with_llamafactory(texts, tokenizer, cutoff_len, template)
            
            # Tokenize with our method
            print("Tokenizing with our method...")
            our_result = tokenize_with_our_method(texts, tokenizer, cutoff_len, template)
            
            # Compare each sample
            template_results = []
            for i in range(len(texts)):
                lf_tokens = llamafactory_result["input_ids"][i]
                our_tokens = our_result["input_ids"][i]
                
                comparison = compare_tokenizations(
                    lf_tokens, our_tokens,
                    "LlamaFactory", "Our"
                )
                
                template_results.append(comparison)
                
                if comparison["identical"]:
                    print(f"  ✅ Sample {i}: IDENTICAL ({len(lf_tokens)} tokens)")
                else:
                    print(f"  ❌ Sample {i}: DIFFERENT")
                    print(f"     LlamaFactory: {len(lf_tokens)} tokens")
                    print(f"     Our method: {len(our_tokens)} tokens")
                    if comparison["differences"]:
                        print(f"     First difference: {comparison['differences'][0]}")
                
                # Show first few tokens for debugging
                if i == 0:
                    print(f"\n  First 20 tokens comparison:")
                    print(f"    LlamaFactory: {lf_tokens[:20]}")
                    print(f"    Our method:   {our_tokens[:20]}")
                    
                    # Decode to see text
                    lf_text = tokenizer.decode(lf_tokens[:50], skip_special_tokens=False)
                    our_text = tokenizer.decode(our_tokens[:50], skip_special_tokens=False)
                    print(f"\n  Decoded text (first 50 tokens):")
                    print(f"    LlamaFactory: '{lf_text[:100]}'...")
                    print(f"    Our method:   '{our_text[:100]}'...")
            
            # Calculate success rate
            identical_count = sum(1 for r in template_results if r["identical"])
            success_rate = identical_count / len(template_results) * 100
            
            all_results[template] = {
                "success_rate": success_rate,
                "identical_samples": identical_count,
                "total_samples": len(template_results),
                "sample_results": template_results
            }
            
            print(f"\n📊 Template '{template}' Summary:")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Identical: {identical_count}/{len(template_results)}")
            
        except Exception as e:
            print(f"❌ Error testing template '{template}': {e}")
            import traceback
            traceback.print_exc()
            all_results[template] = {"error": str(e)}
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for template, result in all_results.items():
        if "error" in result:
            print(f"{template}: ❌ ERROR - {result['error']}")
        else:
            print(f"{template}: {result['success_rate']:.1f}% identical ({result['identical_samples']}/{result['total_samples']})")
    
    # Save results
    output_file = Path("tokenization_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Check if all templates have 100% match
    all_perfect = all(
        result.get("success_rate") == 100 
        for result in all_results.values() 
        if "success_rate" in result
    )
    
    if all_perfect:
        print("\n✅ SUCCESS: All templates produce identical tokenization!")
    else:
        print("\n⚠️  WARNING: Some templates have tokenization differences")
    
    return all_perfect

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)