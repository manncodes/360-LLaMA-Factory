#!/bin/bash

# Minimal Test - Smallest possible baseline case for local GPU
# Single method, single context length, minimal configuration

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}MINIMAL TEST: Single baseline case${NC}"
echo -e "${BLUE}Testing smallest possible configuration${NC}"
echo -e "${BLUE}============================================${NC}"

# Configuration - as minimal as possible
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="saves/methodwise/minimal_${TIMESTAMP}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Ensure directories exist
mkdir -p saves/methodwise
mkdir -p "$RESULTS_DIR"

echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"
echo -e "${GREEN}Model: $MODEL${NC}"

# Create minimal configuration
cat > "minimal_config.yaml" << EOF
### model
model_name_or_path: $MODEL

### method configuration
finetuning_type: full
cutoff_len: 512

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: llama3
lang: en

### output
save_dir: saves/methodwise/minimal_baseline

### eval
batch_size: 1

### needle haystack configuration - ABSOLUTE MINIMUM
needle_context_lengths: [256]
needle_depth_percents: [50]
needle_text: "The secret is MINIMAL_TEST."
needle_question: "What is the secret?"
needle_haystack_data_source: "paulgraham"

### optimization - conservative settings
flash_attn: disabled
use_cache: true
low_cpu_mem_usage: true
infer_dtype: float16
EOF

echo -e "${YELLOW}Created minimal_config.yaml with 256 token context${NC}"
echo -e "${YELLOW}Starting minimal evaluation...${NC}"

# Run minimal evaluation
start_time=$(date +%s)
python3 run_needle_eval.py minimal_config.yaml > "$RESULTS_DIR/minimal.log" 2>&1
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo -e "\n${BLUE}Results:${NC}"

# Check results
result_dir="saves/methodwise/minimal_baseline"
if [ $exit_code -eq 0 ] && [ -f "$result_dir/summary.json" ]; then
    echo -e "${GREEN}[SUCCESS] Minimal test passed in ${duration}s${NC}"
    
    # Show basic results
    python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    
    overall = data.get('overall', {})
    accuracy = overall.get('average_score', 0)
    total_examples = overall.get('total_examples', 0)
    
    print(f'Accuracy: {accuracy:.1%} ({total_examples} examples)')
    
    by_length = data.get('by_context_length', {})
    for length in sorted(by_length.keys(), key=int):
        stats = by_length[length]
        acc = stats.get('average_score', 0)
        count = stats.get('count', 0)
        print(f'  {length} tokens: {acc:.1%} ({count} tests)')
        
except Exception as e:
    print(f'Error reading results: {e}')
"
    
    echo -e "${GREEN}[VERDICT] Needle-in-haystack evaluation works on local GPU${NC}"
    
elif [ $exit_code -eq 124 ]; then
    echo -e "${RED}[TIMEOUT] Minimal test timed out${NC}"
    
else
    echo -e "${RED}[FAILED] Minimal test failed in ${duration}s${NC}"
    echo -e "${RED}Error details:${NC}"
    tail -10 "$RESULTS_DIR/minimal.log" | sed 's/^/  /'
    
    echo -e "\n${YELLOW}Checking for common issues:${NC}"
    
    # Check for CUDA errors
    if grep -q "CUDA error" "$RESULTS_DIR/minimal.log"; then
        echo -e "  ${RED}CUDA error detected - GPU memory or compatibility issue${NC}"
    fi
    
    # Check for model loading errors
    if grep -q "model_name_or_path" "$RESULTS_DIR/minimal.log"; then
        echo -e "  ${RED}Model path issue detected${NC}"
    fi
    
    # Check for template errors
    if grep -q "template" "$RESULTS_DIR/minimal.log"; then
        echo -e "  ${RED}Template issue detected${NC}"
    fi
fi

# Cleanup
rm -f minimal_config.yaml

echo -e "\n${BLUE}============================================${NC}"
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}Minimal test PASSED - larger benchmarks should work${NC}"
else
    echo -e "${RED}Minimal test FAILED - check GPU compatibility${NC}"
fi
echo -e "${BLUE}============================================${NC}"

echo -e "\n${GREEN}Log saved to: $RESULTS_DIR/minimal.log${NC}"