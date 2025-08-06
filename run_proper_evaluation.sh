#!/bin/bash

# Proper Needle-in-Haystack Evaluation for All RoPE Methods
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}ACTUAL Needle-in-Haystack Evaluation${NC}"
echo -e "${BLUE}Testing accuracy across different contexts${NC}"
echo -e "${BLUE}============================================${NC}"

# Create proper configs that will work
create_working_config() {
    local method=$1
    local output_dir=$2
    local context_lengths=$3
    
    cat > "eval_${method}.yaml" << EOF
### model
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0

### method configuration
finetuning_type: full
cutoff_len: 4096

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: llama3
lang: en

### output
save_dir: saves/tinyllama/needle_eval_${method}

### eval
batch_size: 1

### needle haystack configuration
needle_context_lengths: ${context_lengths}
needle_depth_percents: [0, 25, 50, 75, 100]
needle_text: "The magic number is 73628."
needle_question: "What is the magic number mentioned in the document?"
needle_haystack_data_source: "paulgraham"

### optimization
flash_attn: auto
use_cache: true
EOF

    # Add method-specific RoPE configuration
    if [ "$method" != "baseline" ]; then
        echo "rope_scaling: $method" >> "eval_${method}.yaml"
    fi
}

# Function to run evaluation and extract results
run_evaluation() {
    local method=$1
    local context_lengths=$2
    
    echo -e "\n${YELLOW}===========================================${NC}"
    echo -e "${YELLOW}Testing: $method${NC}"
    echo -e "${YELLOW}Context lengths: $context_lengths${NC}"
    echo -e "${YELLOW}===========================================${NC}"
    
    # Create config
    create_working_config "$method" "needle_eval_${method}" "$context_lengths"
    
    # Run evaluation
    timeout 300 python3 run_needle_eval.py "eval_${method}.yaml" > "eval_${method}.log" 2>&1
    
    # Check results
    result_dir="saves/tinyllama/needle_eval_${method}"
    if [ -f "$result_dir/summary.json" ]; then
        echo -e "${GREEN}✅ $method: Evaluation completed${NC}"
        
        # Extract key metrics
        python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    overall = data.get('overall', {})
    print(f'  Overall accuracy: {overall.get(\"average_score\", 0):.2%}')
    print(f'  Exact matches: {overall.get(\"exact_match_rate\", 0):.2%}')
    print(f'  Total examples: {overall.get(\"total_examples\", 0)}')
    
    # Show performance by context length
    by_length = data.get('by_context_length', {})
    if by_length:
        print('  Performance by context length:')
        for length, stats in sorted(by_length.items(), key=lambda x: int(x[0])):
            acc = stats.get('average_score', 0)
            print(f'    {length} tokens: {acc:.2%} ({stats.get(\"count\", 0)} examples)')
except Exception as e:
    print(f'  Error reading results: {e}')
"
    else
        echo -e "${RED}❌ $method: Evaluation failed${NC}"
        echo "  Check eval_${method}.log for details"
        tail -5 "eval_${method}.log" | sed 's/^/  /'
    fi
    
    # Clean up config file
    rm -f "eval_${method}.yaml"
}

# Test different methods with appropriate context lengths
echo -e "\n${BLUE}Starting evaluations...${NC}"

# Baseline - conservative context lengths
run_evaluation "baseline" "[512, 1024, 1536]"

# Linear RoPE - moderate extension 
run_evaluation "linear" "[512, 1024, 2048, 3072]"

# Dynamic RoPE - moderate extension
run_evaluation "dynamic" "[512, 1024, 2048, 3072]"

# Create summary
echo -e "\n${BLUE}============================================${NC}"
echo -e "${GREEN}Evaluation Summary${NC}"
echo -e "${BLUE}============================================${NC}"

echo "Method,Overall_Accuracy,Exact_Match_Rate,Total_Examples" > evaluation_summary.csv

for method in baseline linear dynamic; do
    result_dir="saves/tinyllama/needle_eval_${method}"
    if [ -f "$result_dir/summary.json" ]; then
        python3 -c "
import json
try:
    with open('$result_dir/summary.json', 'r') as f:
        data = json.load(f)
    overall = data.get('overall', {})
    print(f'$method,{overall.get(\"average_score\", 0):.3f},{overall.get(\"exact_match_rate\", 0):.3f},{overall.get(\"total_examples\", 0)}')
except:
    print('$method,0,0,0')
" >> evaluation_summary.csv
    else
        echo "$method,0,0,0" >> evaluation_summary.csv
    fi
done

echo -e "\n${GREEN}Results saved to: evaluation_summary.csv${NC}"
echo -e "${GREEN}Detailed results in: saves/tinyllama/needle_eval_*/summary.json${NC}"

# Show final summary
echo -e "\n${BLUE}Quick Summary:${NC}"
cat evaluation_summary.csv | while IFS=',' read -r method acc exact total; do
    if [ "$method" != "Method" ]; then
        if [ "$acc" != "0" ]; then
            echo -e "  ${GREEN}$method: ${acc%.*}% accuracy, $total examples${NC}"
        else
            echo -e "  ${RED}$method: Failed${NC}"
        fi
    fi
done