#!/bin/bash

# Ultimate Long Context Benchmark - For Hardware with No Limitations
# Tests all methods at extreme context lengths: 4K to 128K tokens

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 ULTIMATE LONG CONTEXT BENCHMARK 🚀${NC}"
echo -e "${BLUE}Testing 4K-128K tokens on unlimited hardware${NC}"
echo -e "${BLUE}================================================${NC}"

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="ultimate_benchmark_${TIMESTAMP}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change to your preferred model

mkdir -p $RESULTS_DIR

echo -e "${GREEN}🎯 Configuration:${NC}"
echo -e "  Model: $MODEL"
echo -e "  Results: $RESULTS_DIR"
echo -e "  Hardware: Unlimited (no VRAM constraints)"

# Test configurations - progressively longer contexts
declare -A TEST_CONFIGS
TEST_CONFIGS["baseline"]="[2048, 4096]"
TEST_CONFIGS["linear"]="[2048, 4096, 8192, 16384, 32768, 65536]"
TEST_CONFIGS["dynamic"]="[2048, 4096, 8192, 16384, 32768, 65536]"
TEST_CONFIGS["yarn"]="[2048, 4096, 8192, 16384, 32768, 65536, 131072]"
TEST_CONFIGS["longrope"]="[2048, 4096, 8192, 16384, 32768, 65536, 131072]"
TEST_CONFIGS["nope"]="[2048, 4096, 8192, 16384, 32768, 65536, 131072]"

# Create dynamic configuration function
create_config() {
    local method=$1
    local context_lengths=$2
    local max_length=$3
    
    cat > "${method}_ultimate.yaml" << EOF
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
save_dir: saves/ultimate/${method}

### eval
batch_size: 1

### needle haystack configuration - EXTREME LONG CONTEXTS
needle_context_lengths: $context_lengths
needle_depth_percents: [5, 25, 50, 75, 95]  # Test across full range
needle_text: "The ultimate secret is LONGCONTEXT_BENCHMARK_SUCCESS_CODE_${method^^}."
needle_question: "What is the ultimate secret mentioned in the document?"
needle_haystack_data_source: "paulgraham"

### optimization for long contexts
flash_attn: fa2  # Force FlashAttention-2 for memory efficiency
use_cache: true
torch_dtype: bfloat16
gradient_checkpointing: true

### memory optimization
low_cpu_mem_usage: true
EOF

    # Add method-specific RoPE configuration
    if [ "$method" != "baseline" ]; then
        echo "rope_scaling: $method" >> "${method}_ultimate.yaml"
        
        # Add method-specific parameters
        case $method in
            "yarn")
                echo "yarn_factor: 8.0" >> "${method}_ultimate.yaml"
                echo "yarn_beta_fast: 32" >> "${method}_ultimate.yaml"
                echo "yarn_beta_slow: 1" >> "${method}_ultimate.yaml"
                ;;
            "longrope") 
                echo "longrope_factor: 32.0" >> "${method}_ultimate.yaml"
                ;;
        esac
    fi
}

# Enhanced benchmark function with progress tracking
run_ultimate_benchmark() {
    local method=$1
    local context_lengths=$2
    local method_num=$3
    local total_methods=$4
    
    echo -e "\n${CYAN}================================================${NC}"
    echo -e "${CYAN}🧪 Testing Method $method_num/$total_methods: ${method^^}${NC}"
    echo -e "${CYAN}Contexts: $context_lengths${NC}"
    echo -e "${CYAN}================================================${NC}"
    
    # Calculate max context length for configuration
    local max_context=$(echo $context_lengths | grep -o '[0-9]\+' | sort -n | tail -1)
    
    # Create configuration
    create_config "$method" "$context_lengths" "$max_context"
    
    # Create method results directory
    mkdir -p "$RESULTS_DIR/$method"
    
    # Run evaluation with extended timeout for long contexts
    local timeout_duration=3600  # 1 hour timeout for extreme contexts
    
    echo -e "${YELLOW}⏳ Starting evaluation (timeout: ${timeout_duration}s)...${NC}"
    start_time=$(date +%s)
    
    timeout $timeout_duration python3 run_needle_eval.py "${method}_ultimate.yaml" \
        > "$RESULTS_DIR/$method/full_log.txt" 2>&1
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Enhanced result analysis
    result_dir="saves/ultimate/${method}"
    if [ $exit_code -eq 0 ] && [ -f "$result_dir/summary.json" ]; then
        echo -e "${GREEN}✅ SUCCESS: $method completed in ${duration}s${NC}"
        
        # Extract comprehensive metrics
        python3 -c "
import json
import sys

try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    
    overall = data.get('overall', {})
    accuracy = overall.get('average_score', 0)
    total_examples = overall.get('total_examples', 0)
    
    print(f'  📊 Overall Performance: {accuracy:.1%} ({total_examples} examples)')
    
    # Performance by context length
    by_length = data.get('by_context_length', {})
    if by_length:
        print('  📏 Performance by Context Length:')
        lengths = sorted(by_length.keys(), key=int)
        
        for i, length in enumerate(lengths):
            stats = by_length[length]
            acc = stats.get('average_score', 0)
            count = stats.get('count', 0)
            
            # Color code performance
            if acc >= 0.9:
                color = '🟢'
            elif acc >= 0.7:
                color = '🟡' 
            else:
                color = '🔴'
                
            print(f'    {color} {int(length):>6} tokens: {acc:.1%} ({count} examples)')
        
        # Performance degradation analysis
        if len(lengths) > 1:
            first_acc = by_length[lengths[0]].get('average_score', 0)
            last_acc = by_length[lengths[-1]].get('average_score', 0)
            
            if first_acc > 0:
                degradation = (first_acc - last_acc) / first_acc * 100
                print(f'  📉 Performance Drop: {degradation:.1f}% from {lengths[0]} to {lengths[-1]} tokens')
                
                # Context scaling efficiency
                context_ratio = int(lengths[-1]) / int(lengths[0])
                print(f'  📈 Context Scaling: {context_ratio:.1f}x length increase')
    
    # Performance by needle position
    by_depth = data.get('by_depth_percent', {})
    if by_depth:
        print('  🎯 Performance by Needle Position:')
        for depth in sorted(by_depth.keys(), key=float):
            stats = by_depth[depth]
            acc = stats.get('average_score', 0)
            count = stats.get('count', 0)
            
            pos_icon = '🏁' if float(depth) == 0 else '🏃' if float(depth) == 50 else '🚩'
            print(f'    {pos_icon} {float(depth):>5.1f}% position: {acc:.1%} ({count} examples)')
    
    # Save to CSV
    with open('$RESULTS_DIR/results.csv', 'a') as f:
        f.write(f'$method,SUCCESS,$duration,{accuracy:.3f},{total_examples},{int(lengths[-1]) if len(lengths) > 0 else 0}\n')
        
except Exception as e:
    print(f'  ❌ Error analyzing results: {e}')
    with open('$RESULTS_DIR/results.csv', 'a') as f:
        f.write('$method,ERROR,$duration,0,0,0\n')
"
        
        # Copy detailed results
        cp -r "$result_dir"/* "$RESULTS_DIR/$method/" 2>/dev/null || true
        
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}⏰ TIMEOUT: $method exceeded ${timeout_duration}s limit${NC}"
        echo "$method,TIMEOUT,$duration,0,0,0" >> "$RESULTS_DIR/results.csv"
        
    else
        echo -e "${RED}❌ FAILED: $method failed in ${duration}s${NC}"
        echo "$method,FAILED,$duration,0,0,0" >> "$RESULTS_DIR/results.csv"
        
        # Show error summary
        echo -e "${RED}Error details:${NC}"
        tail -10 "$RESULTS_DIR/$method/full_log.txt" 2>/dev/null | sed 's/^/  /' || echo "  No error log available"
    fi
    
    # Cleanup config file
    rm -f "${method}_ultimate.yaml"
    
    # Memory cleanup suggestion
    echo -e "${BLUE}💾 Clearing GPU cache...${NC}"
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
}

# Initialize results tracking
echo "Method,Status,Duration,Accuracy,Examples,MaxContext" > "$RESULTS_DIR/results.csv"
echo "# Ultimate Long Context Benchmark Results" > "$RESULTS_DIR/README.md"
echo "Started: $(date)" >> "$RESULTS_DIR/README.md"

# Run all benchmarks
total_methods=${#TEST_CONFIGS[@]}
current_method=1

for method in baseline linear dynamic yarn longrope nope; do
    if [ -n "${TEST_CONFIGS[$method]}" ]; then
        run_ultimate_benchmark "$method" "${TEST_CONFIGS[$method]}" $current_method $total_methods
        ((current_method++))
    fi
done

# Generate final comprehensive analysis
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}🏆 ULTIMATE BENCHMARK COMPLETE 🏆${NC}"
echo -e "${BLUE}================================================${NC}"

# Results summary table
echo -e "\n${YELLOW}📊 FINAL RESULTS SUMMARY:${NC}"
printf "%-12s | %-8s | %-8s | %-8s | %-8s | %-10s\n" "Method" "Status" "Duration" "Accuracy" "Examples" "MaxContext"
echo "-------------|----------|----------|----------|----------|------------"

if [ -f "$RESULTS_DIR/results.csv" ]; then
    tail -n +2 "$RESULTS_DIR/results.csv" | while IFS=',' read method status duration accuracy examples max_context; do
        case $status in
            "SUCCESS")
                acc_pct=$(python3 -c "print(f'{float('$accuracy'):.1%}')" 2>/dev/null || echo "N/A")
                printf "${GREEN}%-12s${NC} | %-8s | %6ss | %-8s | %-8s | %-10s\n" "$method" "$status" "$duration" "$acc_pct" "$examples" "$max_context"
                ;;
            "FAILED"|"ERROR"|"TIMEOUT")
                printf "${RED}%-12s${NC} | %-8s | %6ss | %-8s | %-8s | %-10s\n" "$method" "$status" "$duration" "N/A" "N/A" "$max_context"
                ;;
        esac
    done
fi

# Generate comprehensive analysis report
python3 -c "
import json, csv, os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print('\n🔬 GENERATING COMPREHENSIVE ANALYSIS...')

# Load all results
results_data = []
methods_tested = []

try:
    with open('$RESULTS_DIR/results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_data.append(row)
            if row['Status'] == 'SUCCESS':
                methods_tested.append(row['Method'])

    print(f'📈 Successfully tested methods: {methods_tested}')
    
    # Analyze context scaling performance
    print('\n🚀 CONTEXT SCALING ANALYSIS:')
    
    scaling_data = {}
    for method in methods_tested:
        result_dir = Path(f'saves/ultimate/{method}')
        summary_file = result_dir / 'summary.json'
        
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                
            by_length = data.get('by_context_length', {})
            if by_length:
                scaling_data[method] = {}
                for length, stats in by_length.items():
                    scaling_data[method][int(length)] = stats.get('average_score', 0)
    
    # Find the method that scales best to long contexts
    if scaling_data:
        print('\\n🏆 SCALING CHAMPIONS:')
        
        # Find longest context tested
        max_contexts = {}
        for method, contexts in scaling_data.items():
            max_contexts[method] = max(contexts.keys()) if contexts else 0
        
        longest_context = max(max_contexts.values()) if max_contexts else 0
        
        if longest_context > 0:
            print(f'📏 Maximum context tested: {longest_context:,} tokens')
            
            # Performance at maximum context
            max_context_performance = {}
            for method, contexts in scaling_data.items():
                if longest_context in contexts:
                    max_context_performance[method] = contexts[longest_context]
            
            if max_context_performance:
                best_long_context = max(max_context_performance.keys(), 
                                      key=lambda k: max_context_performance[k])
                best_score = max_context_performance[best_long_context]
                
                print(f'🥇 Best at {longest_context:,} tokens: {best_long_context.upper()} ({best_score:.1%})')
                
                # Show all methods at max context
                print(f'\\n📊 All methods at {longest_context:,} tokens:')
                for method, score in sorted(max_context_performance.items(), 
                                          key=lambda x: x[1], reverse=True):
                    emoji = '🟢' if score >= 0.8 else '🟡' if score >= 0.5 else '🔴'
                    print(f'   {emoji} {method.upper()}: {score:.1%}')

    # Method comparison
    print('\\n🔬 METHOD EFFECTIVENESS:')
    successful_methods = [r for r in results_data if r['Status'] == 'SUCCESS']
    
    if successful_methods:
        # Sort by accuracy
        successful_methods.sort(key=lambda x: float(x['Accuracy']), reverse=True)
        
        print('\\n🏆 RANKING BY OVERALL ACCURACY:')
        for i, method_data in enumerate(successful_methods, 1):
            medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
            accuracy = float(method_data['Accuracy'])
            max_ctx = int(method_data['MaxContext'])
            
            print(f'   {medal} {method_data[\"Method\"].upper()}: {accuracy:.1%} (up to {max_ctx:,} tokens)')

    print(f'\\n📁 Detailed results saved to: $RESULTS_DIR')
    print('📊 Check individual method directories for full analysis')

except Exception as e:
    print(f'Error in analysis: {e}')
"

# Final summary
echo -e "\n${CYAN}================================================${NC}"
echo -e "${CYAN}🎯 BENCHMARK COMPLETE - CHECK RESULTS ABOVE${NC}"  
echo -e "${CYAN}================================================${NC}"

echo -e "\n${GREEN}📁 All results saved to: $RESULTS_DIR${NC}"
echo -e "${GREEN}📊 Detailed logs in: $RESULTS_DIR/[method]/full_log.txt${NC}"
echo -e "${GREEN}📈 Summary data: $RESULTS_DIR/results.csv${NC}"

# Instructions for further analysis
echo -e "\n${BLUE}🔍 For deeper analysis:${NC}"
echo -e "1. Check individual summary.json files in saves/ultimate/[method]/"
echo -e "2. View heatmap images: saves/ultimate/[method]/*.png"
echo -e "3. Compare detailed_results.json for needle position analysis"
echo -e "4. Run: python3 -m llamafactory.eval.plot_niah_comparison --dirs saves/ultimate/*"

echo -e "\n${PURPLE}🚀 ULTIMATE BENCHMARK COMPLETE! 🚀${NC}"