# RoPE Parameters Testing Suite

This directory contains test scripts for verifying RoPE (Rotary Position Embedding) parameter functionality in LlamaFactory.

## Test Scripts

### 1. `test_yaml_rope_params.py`
Tests that RoPE parameters can be properly loaded from YAML configuration files into EvaluationArguments.

**Usage:**
```bash
python test/rope_params/test_yaml_rope_params.py
```

**Tests:**
- Loading rope_scaling_type from YAML
- Loading rope_scaling_factor from YAML  
- Loading YARN-specific parameters (yarn_alpha, yarn_beta)
- Verifying parameters are correctly parsed by HfArgumentParser

### 2. `test_rope_params_end_to_end.py`
End-to-end test of RoPE parameters through the LlamaFactory training and evaluation pipelines.

**Usage:**
```bash
python test/rope_params/test_rope_params_end_to_end.py
```

**Tests:**
- Training pipeline: rope_scaling in ModelArguments (linear/dynamic)
- Evaluation pipeline: Extended RoPE parameters in EvaluationArguments
- Parameter propagation through the argument parsing system

## Supported RoPE Parameters

### Training (ModelArguments)
- `rope_scaling`: "linear" or "dynamic"

### Evaluation (EvaluationArguments)
- `rope_scaling_type`: linear, dynamic, yarn, longrope, llama3
- `rope_scaling_factor`: Scaling factor (float)
- `yarn_alpha`: YARN-specific alpha parameter
- `yarn_beta`: YARN-specific beta parameter

## Notes

- The `rope_theta` parameter is not available in the `llamafactory-native-rope-eval` branch
- For `rope_theta` support, use the `llama-pro-integration` branch

## Running All Tests

```bash
# Run YAML parameter loading tests
python test/rope_params/test_yaml_rope_params.py

# Run end-to-end verification
python test/rope_params/test_rope_params_end_to_end.py
```

Both test scripts provide colored output indicating success (green) or failure (red) for each test case.