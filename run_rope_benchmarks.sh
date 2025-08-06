#!/bin/bash

# Comprehensive RoPE scaling benchmark using needle-in-haystack
# This script tests different RoPE scaling methods

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}RoPE Scaling Methods Benchmark${NC}"
echo -e "${BLUE}Using Needle-in-Haystack Evaluation${NC}"
echo -e "${BLUE}================================${NC}"

# Create results directory
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"

# Function to run benchmark
run_benchmark() {
    local method=$1
    local config_file=$2
    
    echo -e "\n${YELLOW}Testing: $method${NC}"
    echo -e "${YELLOW}Config: $config_file${NC}"
    
    # Run the evaluation
    python3 run_needle_eval.py $config_file 2>&1 | tee $RESULTS_DIR/${method}_log.txt
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $method completed successfully${NC}"
        
        # Copy results
        if [ -d "saves/tinyllama/needle_haystack_${method}" ]; then
            cp -r saves/tinyllama/needle_haystack_${method}/* $RESULTS_DIR/${method}/ 2>/dev/null || true
        fi
    else
        echo -e "${RED}✗ $method failed${NC}"
    fi
}

# Run benchmarks
echo -e "\n${BLUE}Starting benchmarks...${NC}"

# 1. Baseline (no RoPE scaling)
run_benchmark "baseline" "needle_haystack_evaluation/baseline_config.yaml"

# 2. Linear RoPE scaling
run_benchmark "rope_linear" "needle_haystack_evaluation/rope_linear_config.yaml"

# 3. Dynamic NTK scaling
run_benchmark "rope_dynamic" "needle_haystack_evaluation/rope_dynamic_config.yaml"

# Generate comparison plot if all benchmarks completed
echo -e "\n${BLUE}Generating comparison plots...${NC}"
python3 plot_niah_comparison.py \
    --dirs saves/tinyllama/needle_haystack_baseline \
           saves/tinyllama/needle_haystack_rope_linear \
           saves/tinyllama/needle_haystack_rope_dynamic \
    --labels "Baseline" "Linear RoPE" "Dynamic NTK" \
    --output $RESULTS_DIR/comparison.png \
    2>&1 | tee $RESULTS_DIR/comparison_log.txt

echo -e "\n${BLUE}================================${NC}"
echo -e "${GREEN}Benchmark completed!${NC}"
echo -e "${GREEN}Results saved to: $RESULTS_DIR${NC}"
echo -e "${BLUE}================================${NC}"

# Show summary
echo -e "\n${BLUE}Summary:${NC}"
for dir in saves/tinyllama/needle_haystack_*; do
    if [ -d "$dir" ]; then
        method=$(basename $dir | sed 's/needle_haystack_//')
        if [ -f "$dir/needle_results.json" ]; then
            echo -e "  ${GREEN}$method: Results available${NC}"
        else
            echo -e "  ${RED}$method: No results${NC}"
        fi
    fi
done