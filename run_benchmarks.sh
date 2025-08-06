#!/bin/bash

# Benchmark runner for long context methods
# Run from 360-LLaMA-Factory directory

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Long Context Methods Benchmark${NC}"
echo -e "${BLUE}================================${NC}"

# Configuration
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

echo -e "${GREEN}Output directory: $OUTPUT_DIR${NC}"

# Function to run a single benchmark
run_benchmark() {
    local config_name=$1
    local config_file=$2
    local context_length=$3
    
    echo -e "\n${YELLOW}Testing: $config_name (Context: $context_length)${NC}"
    
    # Create test prompt
    echo "Summarize the following text in one sentence: $(python3 -c "print('word ' * $context_length)")" > /tmp/test_prompt.txt
    
    # Run inference and measure time
    start_time=$(date +%s.%N)
    
    timeout 60 python -m llamafactory.cli chat \
        --config $config_file \
        --input /tmp/test_prompt.txt \
        > $OUTPUT_DIR/${config_name}_${context_length}.out 2>&1
    
    exit_code=$?
    end_time=$(date +%s.%N)
    
    # Calculate elapsed time
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Success - Time: ${elapsed}s${NC}"
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}✗ Timeout after 60s${NC}"
    else
        echo -e "${RED}✗ Failed with exit code: $exit_code${NC}"
    fi
    
    # Log results
    echo "$config_name,$context_length,$elapsed,$exit_code" >> $OUTPUT_DIR/results.csv
}

# Initialize results file
echo "config,context_length,time_seconds,exit_code" > $OUTPUT_DIR/results.csv

# Test different context lengths
CONTEXT_LENGTHS=(1024 2048 4096 8192)

echo -e "\n${BLUE}Running benchmarks...${NC}"

# Baseline (no scaling)
for length in "${CONTEXT_LENGTHS[@]}"; do
    if [ $length -le 8192 ]; then
        run_benchmark "baseline" "benchmark_configs/baseline.yaml" $length
    fi
done

# Linear RoPE scaling
for length in "${CONTEXT_LENGTHS[@]}"; do
    run_benchmark "rope_linear" "benchmark_configs/rope_linear.yaml" $length
done

# Dynamic NTK scaling
for length in "${CONTEXT_LENGTHS[@]}"; do
    run_benchmark "rope_dynamic" "benchmark_configs/rope_dynamic.yaml" $length
done

# LongLoRA S²-Attn
for length in "${CONTEXT_LENGTHS[@]}"; do
    run_benchmark "longlora_s2attn" "benchmark_configs/longlora_s2attn.yaml" $length
done

echo -e "\n${BLUE}================================${NC}"
echo -e "${GREEN}Benchmark completed!${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"

# Generate summary
echo -e "\n${BLUE}Summary:${NC}"
python3 - <<EOF
import csv
import statistics

with open('$OUTPUT_DIR/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    results = {}
    
    for row in reader:
        config = row['config']
        if config not in results:
            results[config] = []
        if row['exit_code'] == '0':
            results[config].append(float(row['time_seconds']))
    
    print("\nAverage inference times:")
    for config, times in results.items():
        if times:
            avg_time = statistics.mean(times)
            print(f"  {config}: {avg_time:.2f}s")
        else:
            print(f"  {config}: No successful runs")
EOF

echo -e "${BLUE}================================${NC}"