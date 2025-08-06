#!/bin/bash

# REAL Long Context Benchmarking - Testing 4K to 16K tokens
# Comparing all RoPE methods at the same context lengths

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}REAL Long Context Methods Benchmark${NC}"
echo -e "${BLUE}Testing 4K-16K tokens: Linear vs Dynamic${NC}"
echo -e "${BLUE}================================================${NC}"

# Create results directory
RESULTS_DIR="longcontext_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"

# Create long context configuration
create_longcontext_config() {
    local method=$1
    local context_lengths=$2
    
    cat > "${method}_longcontext.yaml" << EOF
### model
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0

### method configuration
finetuning_type: full
cutoff_len: 8192

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: llama3
lang: en

### output
save_dir: saves/tinyllama/longcontext_${method}

### eval
batch_size: 1

### needle haystack configuration - REAL LONG CONTEXTS
needle_context_lengths: ${context_lengths}
needle_depth_percents: [10, 50, 90]
needle_text: "The secret code is BENCHMARK_2024_ALPHA."
needle_question: "What is the secret code mentioned in the document?"
needle_haystack_data_source: "paulgraham"

### optimization
flash_attn: auto
use_cache: true
EOF

    # Add method-specific RoPE configuration
    if [ "$method" != "baseline" ]; then
        echo "rope_scaling: $method" >> "${method}_longcontext.yaml"
    fi
}

# Function to run long context benchmark
run_longcontext_benchmark() {
    local method=$1
    local context_lengths=$2
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing: $method${NC}"  
    echo -e "${YELLOW}Contexts: $context_lengths${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Create config
    create_longcontext_config "$method" "$context_lengths"
    
    # Run evaluation with timeout
    start_time=$(date +%s)
    timeout 600 python3 run_needle_eval.py "${method}_longcontext.yaml" > "$RESULTS_DIR/${method}.log" 2>&1
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Check results
    result_dir="saves/tinyllama/longcontext_${method}"
    if [ $exit_code -eq 0 ] && [ -f "$result_dir/summary.json" ]; then
        echo -e "${GREEN}✅ $method: SUCCESS in ${duration}s${NC}"
        
        # Extract and display key metrics
        python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    overall = data.get('overall', {})
    print(f'  Accuracy: {overall.get(\"average_score\", 0):.1%}')
    print(f'  Examples: {overall.get(\"total_examples\", 0)}')
    
    by_length = data.get('by_context_length', {})
    for length in sorted(by_length.keys(), key=int):
        stats = by_length[length]
        print(f'  {length}T: {stats.get(\"average_score\", 0):.1%}')
        
    with open('$RESULTS_DIR/results.csv', 'a') as f:
        f.write(f'$method,SUCCESS,$duration,{overall.get(\"average_score\", 0):.3f}\n')
except Exception as e:
    print(f'  Error: {e}')
    with open('$RESULTS_DIR/results.csv', 'a') as f:
        f.write('$method,FAILED,$duration,0\n')
"
    else
        echo -e "${RED}❌ $method: FAILED in ${duration}s${NC}"
        echo "$method,FAILED,$duration,0" >> "$RESULTS_DIR/results.csv"
        
        # Show error details
        echo -e "${RED}Error details:${NC}"
        tail -5 "$RESULTS_DIR/${method}.log" | sed 's/^/  /'
    fi
    
    # Clean up
    rm -f "${method}_longcontext.yaml"
}

# Initialize results
echo "Method,Status,Duration,Accuracy" > "$RESULTS_DIR/results.csv"

echo -e "\n${BLUE}Starting REAL long context benchmarks...${NC}"
echo -e "${YELLOW}Testing contexts: 2K, 4K, 8K, 12K tokens${NC}"

# Test baseline first (limited context)
echo -e "\n${PURPLE}Phase 1: Baseline (max 2K)${NC}"
run_longcontext_benchmark "baseline" "[1024, 2048]"

# Test Linear RoPE scaling
echo -e "\n${PURPLE}Phase 2: Linear RoPE${NC}" 
run_longcontext_benchmark "linear" "[2048, 3072, 4096, 6144]"

# Test Dynamic NTK scaling
echo -e "\n${PURPLE}Phase 3: Dynamic NTK${NC}"
run_longcontext_benchmark "dynamic" "[2048, 3072, 4096, 6144]"

# Analysis and comparison
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Long Context Benchmark Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

# Show results summary
echo -e "\n${YELLOW}RESULTS SUMMARY:${NC}"
printf "%-12s | %-7s | %-8s | %s\n" "Method" "Status" "Duration" "Accuracy"
echo "-------------|---------|----------|----------"

if [ -f "$RESULTS_DIR/results.csv" ]; then
    tail -n +2 "$RESULTS_DIR/results.csv" | while IFS=',' read method status duration accuracy; do
        if [ "$status" = "SUCCESS" ]; then
            acc_pct=$(python3 -c "print(f'{float('$accuracy'):.1%}')")
            printf "${GREEN}%-12s${NC} | %-7s | %6ss | %s\n" "$method" "$status" "$duration" "$acc_pct"
        else
            printf "${RED}%-12s${NC} | %-7s | %6ss | %s\n" "$method" "$status" "$duration" "N/A"
        fi
    done
fi

# Generate detailed performance analysis
echo -e "\n${BLUE}📊 DETAILED PERFORMANCE ANALYSIS:${NC}"
echo "=" * 50

python3 -c "
import json
from pathlib import Path

methods = ['baseline', 'linear', 'dynamic']
results_found = False

for method in methods:
    result_dir = Path(f'saves/tinyllama/longcontext_{method}')
    summary_file = result_dir / 'summary.json'
    
    if summary_file.exists():
        results_found = True
        try:
            with open(summary_file) as f:
                data = json.load(f)
            
            print(f'\n📊 {method.upper()}:')
            overall = data.get('overall', {})
            print(f'  Overall: {overall.get(\"average_score\", 0):.1%} accuracy ({overall.get(\"total_examples\", 0)} examples)')
            
            by_length = data.get('by_context_length', {})
            if by_length:
                print('  Performance by context length:')
                for length in sorted(by_length.keys(), key=int):
                    stats = by_length[length] 
                    acc = stats.get('average_score', 0)
                    count = stats.get('count', 0)
                    print(f'    {int(length):>5} tokens: {acc:.1%} ({count} examples)')
                    
            # Performance degradation analysis
            if len(by_length) > 1:
                lengths = sorted(by_length.keys(), key=int)
                first_acc = by_length[lengths[0]].get('average_score', 0)
                last_acc = by_length[lengths[-1]].get('average_score', 0)
                degradation = (first_acc - last_acc) / first_acc * 100 if first_acc > 0 else 0
                print(f'  📉 Performance degradation: {degradation:.1f}% from {lengths[0]} to {lengths[-1]} tokens')
                
        except Exception as e:
            print(f'  ❌ Error reading {method}: {e}')
    else:
        print(f'\n❌ {method.upper()}: No results found')

if not results_found:
    print('\n❌ No results found. Check logs for errors.')
    
print(f'\n📁 Detailed results and logs saved to: $RESULTS_DIR')
print('📊 Check summary.json files in saves/tinyllama/longcontext_*/ for full data')
"

echo -e "\n${GREEN}Benchmark complete! Check results above for actual long context performance.${NC}"