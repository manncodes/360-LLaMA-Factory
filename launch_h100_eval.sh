#!/bin/bash

# H100 8-GPU RoPE Evaluation Launch Script
# Optimized for maximum performance and memory efficiency

set -e

echo "🚀 Starting H100 8-GPU RoPE Evaluation"
echo "========================================"

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

# Default values
MODEL="meta-llama/Llama-2-70b-hf"
CONFIG="examples/rope_eval_h100_8gpu.yaml"
SP_SIZE=8
SP_MODE="zigzag-ring"
OUTPUT_DIR="./h100_results_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --sp-size)
            SP_SIZE="$2"
            shift 2
            ;;
        --sp-mode)
            SP_MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --extreme)
            CONFIG="examples/rope_eval_h100_extreme.yaml"
            echo "⚠️  EXTREME mode enabled - this will push H100s to their limits"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Model to evaluate (default: Llama-2-70b-hf)"
            echo "  --config CONFIG      Configuration file (default: rope_eval_h100_8gpu.yaml)"
            echo "  --sp-size SIZE       Sequence parallel size (default: 8)"
            echo "  --sp-mode MODE       Sequence parallel mode (default: zigzag-ring)"
            echo "  --output-dir DIR     Output directory"
            echo "  --extreme            Use extreme evaluation settings"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# System checks
echo "🔍 System Checks"
echo "----------------"

# Check NVIDIA driver
if ! nvidia-smi >/dev/null 2>&1; then
    echo "❌ NVIDIA driver not found or not working"
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✅ Found $GPU_COUNT GPUs"

if [ $GPU_COUNT -lt $SP_SIZE ]; then
    echo "❌ Not enough GPUs: need $SP_SIZE, found $GPU_COUNT"
    exit 1
fi

# Check memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "✅ System RAM: ${TOTAL_MEM}GB"

if [ $TOTAL_MEM -lt 64 ]; then
    echo "⚠️  Low system RAM (${TOTAL_MEM}GB < 64GB recommended)"
fi

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ PyTorch CUDA not available"
    exit 1
fi
echo "✅ PyTorch CUDA ready"

# Check dependencies
if ! python -c "from ring_flash_attn import zigzag_ring_flash_attn_func" 2>/dev/null; then
    echo "❌ Ring Flash Attention not available"
    echo "Install with: pip install ring-flash-attn --no-build-isolation"
    exit 1
fi
echo "✅ Ring Flash Attention ready"

# Configuration summary
echo ""
echo "📋 Configuration"
echo "----------------"
echo "Model: $MODEL"
echo "Config: $CONFIG"
echo "Sequence Parallel: $SP_SIZE GPUs, $SP_MODE mode"
echo "Output: $OUTPUT_DIR"
echo ""

# GPU status
echo "💾 GPU Status"
echo "-------------"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %s: %s - %s/%sMB, %s°C, %sW\n", $1, $2, $4, $3, $5, $6}'
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "$OUTPUT_DIR/launch_config.txt" << EOF
Launch Configuration
====================
Date: $(date)
Model: $MODEL
Config: $CONFIG
Sequence Parallel: $SP_SIZE GPUs, $SP_MODE mode
GPU Count: $GPU_COUNT
System RAM: ${TOTAL_MEM}GB
CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

Environment Variables:
PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF
NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE
NCCL_IB_DISABLE: $NCCL_IB_DISABLE
OMP_NUM_THREADS: $OMP_NUM_THREADS
EOF

echo "🚀 Launching Evaluation"
echo "=======================")

# Launch command
LAUNCH_CMD="torchrun --standalone --nproc_per_node=$SP_SIZE run_rope_needle_eval.py \
    --config $CONFIG \
    --model $MODEL \
    --sequence-parallel-size $SP_SIZE \
    --sequence-parallel-mode $SP_MODE \
    --output-dir $OUTPUT_DIR"

echo "Command: $LAUNCH_CMD"
echo ""

# Monitor GPU usage in background
(
    while true; do
        echo "$(date): GPU Memory Usage" >> "$OUTPUT_DIR/gpu_monitor.log"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUTPUT_DIR/gpu_monitor.log"
        sleep 30
    done
) &
MONITOR_PID=$!

# Trap to cleanup background process
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Execute evaluation
$LAUNCH_CMD 2>&1 | tee "$OUTPUT_DIR/evaluation.log"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "📊 Evaluation Complete"
echo "====================="
echo "Exit Code: $EXIT_CODE"
echo "Results: $OUTPUT_DIR"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    
    # Show final GPU state
    echo ""
    echo "🏁 Final GPU State"
    echo "------------------"
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        awk -F, '{printf "GPU %s: %s°C, %sW\n", $1, $2, $3}'
    
    # Show results summary if available
    if [ -f "$OUTPUT_DIR/summary_"*.json ]; then
        echo ""
        echo "📈 Results Summary"
        echo "------------------"
        python -c "
import json, glob
files = glob.glob('$OUTPUT_DIR/summary_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
    print(f'Total Evaluations: {data.get(\"total_evaluations\", 0)}')
    if 'by_technique' in data:
        for tech, stats in data['by_technique'].items():
            print(f'{tech}: {stats.get(\"avg_accuracy\", 0):.2%} accuracy')
" 2>/dev/null || echo "Results processing available in $OUTPUT_DIR"
        
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    echo "Check logs: $OUTPUT_DIR/evaluation.log"
fi

echo ""
echo "Done! 🎉"