#!/bin/bash

# HELMET Benchmark Setup Script for LlamaFactory Integration
# This script sets up HELMET benchmark for use with LlamaFactory

set -e

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration defaults (can be overridden by environment variables)
DEFAULT_HELMET_DATA_PATH="/exp/data/eval_data/HELMET/data"
HELMET_DATA_PATH="${HELMET_DATA_PATH:-$DEFAULT_HELMET_DATA_PATH}"
HELMET_REPO_URL="${HELMET_REPO_URL:-https://github.com/princeton-nlp/HELMET.git}"

echo -e "${CYAN}HELMET Benchmark Setup for LlamaFactory${NC}"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "src/llamafactory/eval/helmet_evaluator.py" ]; then
    echo -e "${RED}Error: This script must be run from the 360-LLaMA-Factory root directory${NC}"
    exit 1
fi

# Get the parent directory (should contain both projects)
PARENT_DIR=$(dirname $(pwd))
HELMET_DIR="${PARENT_DIR}/HELMET"

echo -e "${BLUE}Checking directory structure...${NC}"
echo "   Current dir: $(pwd)"
echo "   Parent dir: ${PARENT_DIR}"
echo "   Expected HELMET dir: ${HELMET_DIR}"
echo "   HELMET data path: ${HELMET_DATA_PATH}"

# Check if HELMET directory exists
if [ ! -d "${HELMET_DIR}" ]; then
    echo -e "${YELLOW}HELMET repository not found. Cloning...${NC}"
    cd "${PARENT_DIR}"
    if git clone "${HELMET_REPO_URL}"; then
        echo -e "${GREEN}HELMET repository cloned successfully${NC}"
    else
        echo -e "${RED}Failed to clone HELMET repository. Check your network/proxy settings.${NC}"
        echo -e "${YELLOW}Alternative: Download manually from ${HELMET_REPO_URL}${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}HELMET repository found${NC}"
fi

# Check if HELMET data exists (configurable path)
if [ -d "${HELMET_DATA_PATH}" ]; then
    echo -e "${GREEN}HELMET data found at: ${HELMET_DATA_PATH}${NC}"
    # Create symlink if data is not in the expected location
    if [ ! -d "${HELMET_DIR}/data" ] && [ "${HELMET_DATA_PATH}" != "${HELMET_DIR}/data" ]; then
        echo -e "${BLUE}Creating symlink to HELMET data...${NC}"
        ln -sf "${HELMET_DATA_PATH}" "${HELMET_DIR}/data"
        echo -e "${GREEN}Symlink created: ${HELMET_DIR}/data -> ${HELMET_DATA_PATH}${NC}"
    fi
elif [ -d "${HELMET_DIR}/data" ]; then
    echo -e "${GREEN}HELMET data found in repository${NC}"
else
    echo -e "${YELLOW}HELMET data not found at ${HELMET_DATA_PATH}${NC}"
    echo -e "${YELLOW}Please ensure HELMET data is available at the configured path${NC}"
    echo -e "${BLUE}To use a different path, set: export HELMET_DATA_PATH=/your/path${NC}"
    exit 1
fi

# Check Python dependencies (without installing)
echo -e "${BLUE}Checking Python dependencies...${NC}"

check_python_module() {
    if python -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}$1 available${NC}"
        return 0
    else
        echo -e "${RED}$1 missing${NC}"
        return 1
    fi
}

MISSING_DEPS=()

check_python_module "yaml" || MISSING_DEPS+=("PyYAML")
check_python_module "numpy" || MISSING_DEPS+=("numpy") 
check_python_module "datasets" || MISSING_DEPS+=("datasets")
check_python_module "torch" || MISSING_DEPS+=("torch")
check_python_module "transformers" || MISSING_DEPS+=("transformers")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo -e "${BLUE}Install with: pip install ${MISSING_DEPS[*]}${NC}"
else
    echo -e "${GREEN}All required dependencies available${NC}"
fi

echo ""
echo -e "${BLUE}Testing HELMET integration...${NC}"

# Return to LlamaFactory directory
cd "${PARENT_DIR}/360-LLaMA-Factory" > /dev/null

# Test if we can import HELMET modules
if python3 -c "
import sys
sys.path.insert(0, '${HELMET_DIR}')
try:
    from arguments import parse_arguments
    from model_utils import load_LLM
    from data import load_data
    print('HELMET modules import successfully')
except ImportError as e:
    print(f'HELMET import failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
    echo -e "${GREEN}HELMET modules import successfully${NC}"
else
    echo -e "${RED}HELMET import failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Running configuration validation...${NC}"

# Test YAML configuration loading
CONFIGS=("helmet_quick_demo.yaml" "helmet_recall_config.yaml" "helmet_longqa_config.yaml" "helmet_comprehensive_config.yaml")
for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        if python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null; then
            echo -e "${GREEN}$config valid${NC}"
        else
            echo -e "${RED}$config invalid${NC}"
        fi
    else
        echo -e "${YELLOW}$config not found${NC}"
    fi
done

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

echo -e "${BLUE}Created test configuration: test_helmet_setup.yaml${NC}"

echo ""
echo -e "${GREEN}HELMET setup validation completed${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "   1. Install missing dependencies if any"
echo "   2. Run quick test: llamafactory-cli eval test_helmet_setup.yaml"
echo "   3. Try demo config: llamafactory-cli eval helmet_quick_demo.yaml"
echo "   4. Run full evaluation: llamafactory-cli eval helmet_recall_config.yaml"
echo ""
echo -e "${CYAN}Documentation:${NC} evaluation/helmet/README.md"
echo -e "${CYAN}Available configs:${NC}"
echo "   - helmet_quick_demo.yaml (fast test)"
echo "   - helmet_recall_config.yaml (memory tasks)" 
echo "   - helmet_longqa_config.yaml (QA tasks)"
echo "   - helmet_comprehensive_config.yaml (multiple tasks)"
echo ""
echo -e "${CYAN}Environment variables:${NC}"
echo "   - HELMET_DATA_PATH: ${HELMET_DATA_PATH}"
echo "   - HELMET_REPO_URL: ${HELMET_REPO_URL}"
echo ""

# Clean up test config
rm test_helmet_setup.yaml

echo -e "${GREEN}Setup validation complete. Ready to evaluate long-context models with HELMET.${NC}"