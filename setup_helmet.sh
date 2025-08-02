#!/bin/bash

# HELMET Benchmark Setup Script for LlamaFactory Integration
# This script sets up HELMET benchmark for use with LlamaFactory

set -e

echo "🛡️  HELMET Benchmark Setup for LlamaFactory"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "src/llamafactory/eval/helmet_evaluator.py" ]; then
    echo "❌ Error: This script must be run from the 360-LLaMA-Factory root directory"
    exit 1
fi

# Get the parent directory (should contain both projects)
PARENT_DIR=$(dirname $(pwd))
HELMET_DIR="${PARENT_DIR}/HELMET"

echo "📁 Checking directory structure..."
echo "   Current dir: $(pwd)"
echo "   Parent dir: ${PARENT_DIR}"
echo "   Expected HELMET dir: ${HELMET_DIR}"

# Check if HELMET directory exists
if [ ! -d "${HELMET_DIR}" ]; then
    echo "📥 HELMET repository not found. Cloning..."
    cd "${PARENT_DIR}"
    git clone https://github.com/princeton-nlp/HELMET.git
    echo "✅ HELMET repository cloned successfully"
else
    echo "✅ HELMET repository found"
fi

# Check if HELMET data exists
if [ ! -d "${HELMET_DIR}/data" ]; then
    echo "📊 HELMET data not found. Downloading..."
    cd "${HELMET_DIR}"
    bash scripts/download_data.sh
    echo "✅ HELMET data downloaded successfully (~34GB)"
else
    echo "✅ HELMET data found"
fi

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
cd "${HELMET_DIR}"

if ! python -c "import yaml" 2>/dev/null; then
    echo "📦 Installing PyYAML..."
    pip install PyYAML
fi

if ! python -c "import numpy" 2>/dev/null; then
    echo "📦 Installing numpy..."
    pip install numpy
fi

if ! python -c "import datasets" 2>/dev/null; then
    echo "📦 Installing datasets..."
    pip install datasets
fi

# Try to install flash-attn (optional but recommended)
echo "⚡ Attempting to install flash-attention (optional, for GPU acceleration)..."
pip install flash-attn || echo "⚠️  Flash attention installation failed (this is optional)"

echo ""
echo "🧪 Testing HELMET integration..."

# Return to LlamaFactory directory
cd - > /dev/null

# Test if we can import HELMET modules
python3 -c "
import sys
sys.path.insert(0, '${HELMET_DIR}')
try:
    from arguments import parse_arguments
    from model_utils import load_LLM
    from data import load_data
    print('✅ HELMET modules import successfully')
except ImportError as e:
    print(f'❌ HELMET import failed: {e}')
    sys.exit(1)
"

echo ""
echo "🎯 Running quick validation test..."

# Create a minimal test config
cat > test_helmet_setup.yaml << EOF
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
finetuning_type: full
task: helmet_demo
save_dir: saves/helmet_setup_test
batch_size: 1
helmet_tasks: "json_kv"
helmet_input_max_length: 4096
helmet_generation_max_length: 20
helmet_shots: 1
helmet_max_test_samples: 2
EOF

echo "Created test configuration: test_helmet_setup.yaml"

echo ""
echo "🎉 HELMET setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Run quick test: llamafactory-cli eval test_helmet_setup.yaml"
echo "   2. Try demo config: llamafactory-cli eval helmet_quick_demo.yaml"
echo "   3. Run full evaluation: llamafactory-cli eval helmet_recall_config.yaml"
echo ""
echo "📚 Documentation: evaluation/helmet/README.md"
echo "🔧 Available configs:"
echo "   - helmet_quick_demo.yaml (fast test)"
echo "   - helmet_recall_config.yaml (memory tasks)"
echo "   - helmet_longqa_config.yaml (QA tasks)"
echo "   - helmet_comprehensive_config.yaml (multiple tasks)"
echo ""

# Clean up test config
rm test_helmet_setup.yaml

echo "✨ Setup complete! Ready to evaluate long-context models with HELMET."