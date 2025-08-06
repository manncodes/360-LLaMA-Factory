# Needle-in-Haystack Performance Report

**Analysis Date**: 2025-08-05 22:18:00
**Methods Analyzed**: 5

## Overall Performance Summary

| Method | Overall Accuracy | Exact Match Rate | Total Examples |
|--------|------------------|------------------|----------------|
| 🥇 custom_test | 100.0% | 100.0% | 6 |
| 🥈 eval_test | 100.0% | 100.0% | 6 |
| 🥉 paulgraham | 100.0% | 100.0% | 6 |
| 4 paulgraham_v2 | 100.0% | 100.0% | 15 |
| 5 eval | 98.8% | 95.0% | 20 |

## Detailed Method Analysis

### Custom Test

- **Overall Accuracy**: 100.0%
- **Exact Match Rate**: 100.0%
- **Total Examples**: 6

**Performance by Context Length:**
- 135 tokens: 100.0% (3 examples)
- 282 tokens: 100.0% (3 examples)

**Performance by Needle Position:**
- 0.0% through context: 100.0% (2 examples)
- 50.0% through context: 100.0% (2 examples)
- 100.0% through context: 100.0% (2 examples)

### Eval

- **Overall Accuracy**: 98.8%
- **Exact Match Rate**: 95.0%
- **Total Examples**: 20

**Performance by Context Length:**
- 240 tokens: 100.0% (5 examples)
- 480 tokens: 95.0% (5 examples)
- 958 tokens: 100.0% (5 examples)
- 1918 tokens: 100.0% (5 examples)

**Performance by Needle Position:**
- 0.0% through context: 100.0% (4 examples)
- 25.0% through context: 93.8% (4 examples)
- 50.0% through context: 100.0% (4 examples)
- 75.0% through context: 100.0% (4 examples)
- 100.0% through context: 100.0% (4 examples)

### Eval Test

- **Overall Accuracy**: 100.0%
- **Exact Match Rate**: 100.0%
- **Total Examples**: 6

**Performance by Context Length:**
- 135 tokens: 100.0% (3 examples)
- 282 tokens: 100.0% (3 examples)

**Performance by Needle Position:**
- 0.0% through context: 100.0% (2 examples)
- 50.0% through context: 100.0% (2 examples)
- 100.0% through context: 100.0% (2 examples)

### Paulgraham

- **Overall Accuracy**: 100.0%
- **Exact Match Rate**: 100.0%
- **Total Examples**: 6

**Performance by Context Length:**
- 135 tokens: 100.0% (3 examples)
- 282 tokens: 100.0% (3 examples)

**Performance by Needle Position:**
- 0.0% through context: 100.0% (2 examples)
- 50.0% through context: 100.0% (2 examples)
- 100.0% through context: 100.0% (2 examples)

### Paulgraham V2

- **Overall Accuracy**: 100.0%
- **Exact Match Rate**: 100.0%
- **Total Examples**: 15

**Performance by Context Length:**
- 285 tokens: 100.0% (5 examples)
- 573 tokens: 100.0% (5 examples)
- 858 tokens: 100.0% (5 examples)

**Performance by Needle Position:**
- 0.0% through context: 100.0% (3 examples)
- 25.0% through context: 100.0% (3 examples)
- 50.0% through context: 100.0% (3 examples)
- 75.0% through context: 100.0% (3 examples)
- 100.0% through context: 100.0% (3 examples)

## Key Insights

- **Best Performing Method**: custom_test (100.0% accuracy)
- **Lowest Performing Method**: eval (98.8% accuracy)
- **Performance Gap**: 1.3% relative improvement from worst to best
- **Maximum Context Length Tested**: 1918 tokens

## Methodology

- **Task**: Needle-in-haystack retrieval
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Background Text**: Paul Graham essays
- **Evaluation**: Exact string matching for needle retrieval
