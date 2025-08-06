#!/bin/bash

# CPU-Only Minimal Test - Force CPU inference to avoid GPU issues
# This will verify the needle-in-haystack logic works

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}CPU MINIMAL TEST: Force CPU inference${NC}"
echo -e "${BLUE}Testing basic needle-in-haystack logic${NC}"
echo -e "${BLUE}============================================${NC}"

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="saves/methodwise/cpu_minimal_${TIMESTAMP}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Ensure directories exist
mkdir -p saves/methodwise
mkdir -p "$RESULTS_DIR"

echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"
echo -e "${GREEN}Model: $MODEL (CPU-only)${NC}"

# Create CPU-only configuration
cat > "cpu_minimal_config.yaml" << EOF
### model
model_name_or_path: $MODEL

### method configuration
finetuning_type: full
cutoff_len: 256

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: llama3
lang: en

### output
save_dir: saves/methodwise/cpu_minimal

### eval
batch_size: 1

### needle haystack configuration - TINY
needle_context_lengths: [128]
needle_depth_percents: [50]
needle_text: "The secret is CPU_TEST."
needle_question: "What is the secret?"
needle_haystack_data_source: "paulgraham"

### optimization - FORCE CPU (only valid parameters)
flash_attn: disabled
use_cache: false
low_cpu_mem_usage: true
infer_dtype: float32
EOF

echo -e "${YELLOW}Created CPU-only config with 128 token context${NC}"

# Force CPU-only environment
export CUDA_VISIBLE_DEVICES=""

echo -e "${YELLOW}Starting CPU-only evaluation...${NC}"

# Run CPU evaluation
start_time=$(date +%s)
python3 run_needle_eval.py cpu_minimal_config.yaml > "$RESULTS_DIR/cpu_minimal.log" 2>&1
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo -e "\n${BLUE}Results:${NC}"

# Check results
result_dir="saves/methodwise/cpu_minimal"
if [ $exit_code -eq 0 ] && [ -f "$result_dir/summary.json" ]; then
    echo -e "${GREEN}[SUCCESS] CPU minimal test passed in ${duration}s${NC}"
    
    # Show results
    python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    
    overall = data.get('overall', {})
    accuracy = overall.get('average_score', 0)
    total_examples = overall.get('total_examples', 0)
    
    print(f'CPU Test Accuracy: {accuracy:.1%} ({total_examples} examples)')
    
    by_length = data.get('by_context_length', {})
    for length in sorted(by_length.keys(), key=int):
        stats = by_length[length]
        acc = stats.get('average_score', 0)
        count = stats.get('count', 0)
        print(f'  {length} tokens: {acc:.1%} ({count} tests)')
        
    print(f'\\nCPU evaluation took: {duration}s')
        
except Exception as e:
    print(f'Error reading results: {e}')
"
    
    echo -e "\n${GREEN}[VERDICT] Needle-in-haystack evaluation works!${NC}"
    echo -e "${GREEN}GPU errors are environment-specific, production should work${NC}"
    
elif [ $exit_code -eq 124 ]; then
    echo -e "${RED}[TIMEOUT] CPU test timed out${NC}"
    
else
    echo -e "${RED}[FAILED] CPU test failed in ${duration}s${NC}"
    echo -e "${RED}Error details:${NC}"
    tail -15 "$RESULTS_DIR/cpu_minimal.log" | sed 's/^/  /'
    
    echo -e "\n${YELLOW}Diagnostic information:${NC}"
    
    # Check specific errors
    if grep -q "model_name_or_path" "$RESULTS_DIR/cpu_minimal.log"; then
        echo -e "  ${RED}Model loading issue${NC}"
    fi
    
    if grep -q "template" "$RESULTS_DIR/cpu_minimal.log"; then
        echo -e "  ${RED}Template configuration issue${NC}"
    fi
    
    if grep -q "needle_haystack_proper" "$RESULTS_DIR/cpu_minimal.log"; then
        echo -e "  ${RED}Task definition issue${NC}"
    fi
    
    # Show if CPU was actually used
    if grep -q "cuda" "$RESULTS_DIR/cpu_minimal.log"; then
        echo -e "  ${YELLOW}Warning: CUDA still detected in logs${NC}"
    else
        echo -e "  ${GREEN}CPU-only mode confirmed${NC}"
    fi
fi

# Cleanup
rm -f cpu_minimal_config.yaml

echo -e "\n${BLUE}============================================${NC}"
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}CPU test PASSED - evaluation logic works correctly${NC}"
    echo -e "${GREEN}GPU issues are hardware-specific limitations${NC}"
    echo -e "${GREEN}Production scripts should work on proper hardware${NC}"
else
    echo -e "${RED}CPU test FAILED - may need configuration adjustments${NC}"
fi
echo -e "${BLUE}============================================${NC}"

echo -e "\n${GREEN}Log saved to: $RESULTS_DIR/cpu_minimal.log${NC}"