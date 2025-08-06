#!/bin/bash

# Small Test Version for Local Verification
# Tests only baseline and linear at short contexts

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TEST: Method-wise Benchmark (Local)${NC}"
echo -e "${BLUE}Testing short contexts for verification${NC}"
echo -e "${BLUE}============================================${NC}"

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="saves/methodwise/test_${TIMESTAMP}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Ensure directories exist
mkdir -p saves/methodwise
mkdir -p "$RESULTS_DIR"
mkdir -p scripts/methods

echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"
echo -e "${GREEN}Model: $MODEL${NC}"

# Test configurations - small contexts only
declare -A METHODS
METHODS[baseline]="[512, 1024]"
METHODS[linear]="[512, 1024, 2048]"

# Create clean configuration
create_method_config() {
    local method=$1
    local context_lengths=$2
    local max_length=$(echo $context_lengths | grep -o '[0-9]\+' | sort -n | tail -1)
    
    cat > "${method}_config.yaml" << EOF
### model
model_name_or_path: $MODEL

### method configuration
finetuning_type: full
cutoff_len: $max_length

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: llama3
lang: en

### output
save_dir: saves/methodwise/$method

### eval
batch_size: 1

### needle haystack configuration - SHORT CONTEXTS FOR TESTING
needle_context_lengths: $context_lengths
needle_depth_percents: [50]
needle_text: "The test secret is TEST_${method^^}_SUCCESS."
needle_question: "What is the test secret mentioned in the document?"
needle_haystack_data_source: "paulgraham"

### optimization - only valid parameters
flash_attn: fa2
use_cache: true
low_cpu_mem_usage: true
EOF

    # Add method-specific RoPE configuration
    if [ "$method" != "baseline" ]; then
        echo "rope_scaling: $method" >> "${method}_config.yaml"
    fi
}

# Simple benchmark function
run_test_benchmark() {
    local method=$1
    local context_lengths=$2
    local method_num=$3
    local total_methods=$4
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Test $method_num/$total_methods: ${method^^}${NC}"
    echo -e "${YELLOW}Contexts: $context_lengths${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Create config
    create_method_config "$method" "$context_lengths"
    
    # Run evaluation with shorter timeout
    start_time=$(date +%s)
    echo "Starting evaluation..."
    timeout 300 python3 run_needle_eval.py "${method}_config.yaml" > "$RESULTS_DIR/${method}.log" 2>&1
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Check results
    result_dir="saves/methodwise/$method"
    if [ $exit_code -eq 0 ] && [ -f "$result_dir/summary.json" ]; then
        echo -e "${GREEN}[TEST SUCCESS] $method (${duration}s)${NC}"
        
        # Extract basic metrics
        python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    
    overall = data.get('overall', {})
    accuracy = overall.get('average_score', 0)
    total_examples = overall.get('total_examples', 0)
    
    print(f'  Test accuracy: {accuracy:.1%} ({total_examples} examples)')
    
    by_length = data.get('by_context_length', {})
    if by_length:
        for length in sorted(by_length.keys(), key=int):
            stats = by_length[length]
            acc = stats.get('average_score', 0)
            count = stats.get('count', 0)
            print(f'    {int(length):>4}T: {acc:.1%} ({count})')
    
    # Save simple CSV
    with open('$RESULTS_DIR/test_results.csv', 'a') as f:
        f.write(f'$method,SUCCESS,$duration,{accuracy:.3f},{total_examples}\\n')
        
except Exception as e:
    print(f'  Error: {e}')
    with open('$RESULTS_DIR/test_results.csv', 'a') as f:
        f.write('$method,ERROR,$duration,0,0\\n')
"
        
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}[TEST TIMEOUT] $method (5min limit)${NC}"
        echo "$method,TIMEOUT,$duration,0,0" >> "$RESULTS_DIR/test_results.csv"
        
    else
        echo -e "${RED}[TEST FAILED] $method (${duration}s)${NC}"
        echo "$method,FAILED,$duration,0,0" >> "$RESULTS_DIR/test_results.csv"
        echo -e "${RED}Error log:${NC}"
        tail -3 "$RESULTS_DIR/${method}.log" | sed 's/^/  /'
    fi
    
    # Cleanup
    rm -f "${method}_config.yaml"
}

# Initialize results
echo "Method,Status,Duration,Accuracy,Examples" > "$RESULTS_DIR/test_results.csv"

# Run test benchmarks (only 2 methods)
total_methods=${#METHODS[@]}
current_method=1

for method in baseline linear; do
    if [ -n "${METHODS[$method]}" ]; then
        run_test_benchmark "$method" "${METHODS[$method]}" $current_method $total_methods
        ((current_method++))
    fi
done

# Test summary
echo -e "\n${BLUE}============================================${NC}"
echo -e "${BLUE}Test Complete!${NC}"
echo -e "${BLUE}============================================${NC}"

echo -e "\n${YELLOW}TEST RESULTS:${NC}"
printf "%-12s | %-8s | %-8s | %-8s\n" "Method" "Status" "Duration" "Accuracy"
echo "-------------|----------|----------|----------"

if [ -f "$RESULTS_DIR/test_results.csv" ]; then
    tail -n +2 "$RESULTS_DIR/test_results.csv" | while IFS=',' read method status duration accuracy examples; do
        if [ "$status" = "SUCCESS" ]; then
            acc_pct=$(python3 -c "print(f'{float('$accuracy'):.1%}')" 2>/dev/null || echo "N/A")
            printf "${GREEN}%-12s${NC} | %-8s | %6ss | %-8s\n" "$method" "$status" "$duration" "$acc_pct"
        else
            printf "${RED}%-12s${NC} | %-8s | %6ss | %-8s\n" "$method" "$status" "$duration" "N/A"
        fi
    done
fi

echo -e "\n${GREEN}Test results saved to: $RESULTS_DIR${NC}"
echo -e "\n${BLUE}If tests pass, the production script should work on A100s${NC}"