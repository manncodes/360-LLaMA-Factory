#!/bin/bash

# Method-wise Long Context Benchmark
# Tests all RoPE scaling methods at progressively longer contexts

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Method-wise Long Context Benchmark${NC}"
echo -e "${BLUE}Testing contexts: 2K -> 4K -> 8K -> 16K -> 32K${NC}"
echo -e "${BLUE}============================================${NC}"

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="saves/methodwise/results_${TIMESTAMP}"
MODEL="/exp/model/Huggingface/meta-llama/Llama-3.2-1B"

# Ensure directories exist
mkdir -p saves/methodwise
mkdir -p "$RESULTS_DIR"
mkdir -p scripts/methods

echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"
echo -e "${GREEN}Model: $MODEL${NC}"

# Test configurations - only valid parameters
declare -A METHODS
METHODS[baseline]="[2048, 4096]"
METHODS[linear]="[2048, 4096, 8192, 16384]"
METHODS[dynamic]="[2048, 4096, 8192, 16384]"
METHODS[yarn]="[2048, 4096, 8192, 16384, 32768]"
METHODS[longrope]="[2048, 4096, 8192, 16384, 32768]"
METHODS[nope]="[2048, 4096, 8192, 16384, 32768]"

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

### needle haystack configuration
needle_context_lengths: $context_lengths
needle_depth_percents: [10, 50, 90]
needle_text: "The benchmark secret is METHOD_${method^^}_SUCCESS."
needle_question: "What is the benchmark secret mentioned in the document?"
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

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local method=$3
    local width=50
    
    local percent=$(( current * 100 / total ))
    local filled=$(( width * current / total ))
    local empty=$(( width - filled ))
    
    printf "\r[%*s%*s] %d%% - %s" $filled '' $empty '' $percent "$method"
    
    if [ $current -eq $total ]; then
        echo
    fi
}

# Benchmark function
run_method_benchmark() {
    local method=$1
    local context_lengths=$2
    local method_num=$3
    local total_methods=$4
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Method $method_num/$total_methods: ${method^^}${NC}"
    echo -e "${YELLOW}Contexts: $context_lengths${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Create config
    create_method_config "$method" "$context_lengths"
    
    # Show initial progress
    show_progress 0 100 "Starting $method"
    
    # Run evaluation
    start_time=$(date +%s)
    timeout 1800 python3 run_needle_eval.py "${method}_config.yaml" > "$RESULTS_DIR/${method}.log" 2>&1 &
    eval_pid=$!
    
    # Monitor progress
    while kill -0 $eval_pid 2>/dev/null; do
        sleep 5
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        progress=$(( elapsed * 100 / 1800 ))  # Assuming max 30min
        [ $progress -gt 100 ] && progress=100
        show_progress $progress 100 "Running $method (${elapsed}s)"
    done
    
    wait $eval_pid
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Complete progress bar
    show_progress 100 100 "$method complete"
    
    # Check results
    result_dir="saves/methodwise/$method"
    if [ $exit_code -eq 0 ] && [ -f "$result_dir/summary.json" ]; then
        echo -e "${GREEN}[SUCCESS] $method (${duration}s)${NC}"
        
        # Use analysis script
        python3 scripts/methods/analyze_results.py "$method" "$result_dir" > "$RESULTS_DIR/${method}_analysis.txt"
        
        # Extract metrics for CSV
        python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    
    overall = data.get('overall', {})
    accuracy = overall.get('average_score', 0)
    total_examples = overall.get('total_examples', 0)
    
    by_length = data.get('by_context_length', {})
    max_context = max([int(k) for k in by_length.keys()]) if by_length else 0
    
    # Save to CSV
    with open('$RESULTS_DIR/results.csv', 'a') as f:
        f.write(f'$method,SUCCESS,$duration,{accuracy:.3f},{total_examples},{max_context}\\n')
    
    # Save detailed JSON log
    result_data = {
        'method': '$method',
        'status': 'SUCCESS',
        'duration': $duration,
        'accuracy': accuracy,
        'total_examples': total_examples,
        'max_context': max_context,
        'timestamp': '$(date -Iseconds)',
        'detailed_results': data
    }
    
    with open('$RESULTS_DIR/detailed_results.jsonl', 'a') as f:
        f.write(json.dumps(result_data) + '\\n')
        
except Exception as e:
    print(f'Error: {e}')
    with open('$RESULTS_DIR/results.csv', 'a') as f:
        f.write('$method,ERROR,$duration,0,0,0\\n')
"
        
        # Show summary from analysis
        cat "$RESULTS_DIR/${method}_analysis.txt"
        
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}[TIMEOUT] $method (30min limit)${NC}"
        echo "$method,TIMEOUT,$duration,0,0,0" >> "$RESULTS_DIR/results.csv"
        
        # Log timeout to JSON
        python3 -c "
import json
result_data = {
    'method': '$method',
    'status': 'TIMEOUT',
    'duration': $duration,
    'timestamp': '$(date -Iseconds)',
    'error': 'Evaluation exceeded 30 minute timeout'
}
with open('$RESULTS_DIR/detailed_results.jsonl', 'a') as f:
    f.write(json.dumps(result_data) + '\\n')
"
        
    else
        echo -e "${RED}[FAILED] $method (${duration}s)${NC}"
        echo "$method,FAILED,$duration,0,0,0" >> "$RESULTS_DIR/results.csv"
        echo -e "${RED}Error log:${NC}"
        tail -5 "$RESULTS_DIR/${method}.log" | sed 's/^/  /'
        
        # Log error to JSON
        python3 -c "
import json
error_log = open('$RESULTS_DIR/${method}.log').read()[-500:]  # Last 500 chars
result_data = {
    'method': '$method',
    'status': 'FAILED',
    'duration': $duration,
    'timestamp': '$(date -Iseconds)',
    'error_log': error_log
}
with open('$RESULTS_DIR/detailed_results.jsonl', 'a') as f:
    f.write(json.dumps(result_data) + '\\n')
"
    fi
    
    # Cleanup
    rm -f "${method}_config.yaml"
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    sleep 2  # Brief pause between methods
}

# Initialize results
echo "Method,Status,Duration,Accuracy,Examples,MaxContext" > "$RESULTS_DIR/results.csv"
echo "# Method-wise benchmark started at $(date)" > "$RESULTS_DIR/README.md"
echo "{\"benchmark_start\": \"$(date -Iseconds)\", \"model\": \"$MODEL\"}" > "$RESULTS_DIR/detailed_results.jsonl"

# Run benchmarks
total_methods=${#METHODS[@]}
current_method=1

for method in baseline linear dynamic yarn longrope nope; do
    if [ -n "${METHODS[$method]}" ]; then
        run_method_benchmark "$method" "${METHODS[$method]}" $current_method $total_methods
        ((current_method++))
    fi
done

# Final analysis
echo -e "\n${BLUE}============================================${NC}"
echo -e "${BLUE}Benchmark Complete!${NC}"
echo -e "${BLUE}============================================${NC}"

# Results summary
echo -e "\n${YELLOW}RESULTS SUMMARY:${NC}"
printf "%-12s | %-8s | %-8s | %-8s | %-10s\n" "Method" "Status" "Duration" "Accuracy" "MaxContext"
echo "-------------|----------|----------|----------|------------"

if [ -f "$RESULTS_DIR/results.csv" ]; then
    tail -n +2 "$RESULTS_DIR/results.csv" | while IFS=',' read method status duration accuracy examples max_context; do
        if [ "$status" = "SUCCESS" ]; then
            acc_pct=$(python3 -c "print(f'{float('$accuracy'):.1%}')" 2>/dev/null || echo "N/A")
            printf "${GREEN}%-12s${NC} | %-8s | %6ss | %-8s | %-10s\n" "$method" "$status" "$duration" "$acc_pct" "$max_context"
        else
            printf "${RED}%-12s${NC} | %-8s | %6ss | %-8s | %-10s\n" "$method" "$status" "$duration" "N/A" "$max_context"
        fi
    done
fi

# Performance analysis using script
echo -e "\n${BLUE}PERFORMANCE ANALYSIS:${NC}"
python3 scripts/methods/analyze_results.py

echo -e "\n${GREEN}Results saved to: $RESULTS_DIR${NC}"
echo -e "${GREEN}JSON logs: $RESULTS_DIR/detailed_results.jsonl${NC}"
echo -e "${GREEN}CSV summary: $RESULTS_DIR/results.csv${NC}"

echo -e "\n${GREEN}Method-wise benchmark complete!${NC}"