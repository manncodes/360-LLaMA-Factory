#!/bin/bash

# Simple setup script to link existing LongBench v2 HuggingFace dataset
# The dataset is already downloaded at /exp/data/eval_data/LongBench-v2

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Paths
EXISTING_LONGBENCH="/exp/data/eval_data/LongBench-v2"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EVAL_DIR="$SCRIPT_DIR/evaluation/longbench"
SYMLINK_PATH="$EVAL_DIR/LongBench-v2"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}LongBench v2 Dataset Link Setup${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Check if the existing dataset path exists
if [ ! -d "$EXISTING_LONGBENCH" ]; then
    echo -e "${RED}✗ LongBench v2 dataset not found at: $EXISTING_LONGBENCH${NC}"
    echo -e "${YELLOW}  Please ensure the dataset is downloaded at this location${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found LongBench v2 dataset at: $EXISTING_LONGBENCH${NC}"

# Create evaluation directory if it doesn't exist
mkdir -p "$EVAL_DIR"

# Remove existing symlink if it exists
if [ -L "$SYMLINK_PATH" ]; then
    echo -e "${YELLOW}Removing existing symlink...${NC}"
    rm "$SYMLINK_PATH"
elif [ -e "$SYMLINK_PATH" ]; then
    echo -e "${RED}✗ $SYMLINK_PATH exists and is not a symlink${NC}"
    echo -e "${YELLOW}  Please remove it manually or use --force${NC}"
    exit 1
fi

# Create symlink
echo -e "${BLUE}Creating symlink...${NC}"
ln -s "$EXISTING_LONGBENCH" "$SYMLINK_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Created symlink: $SYMLINK_PATH -> $EXISTING_LONGBENCH${NC}"
else
    echo -e "${RED}✗ Failed to create symlink${NC}"
    exit 1
fi

# Update YAML configurations to use the dataset path
echo -e "${BLUE}Creating example configuration...${NC}"

cat > "$SCRIPT_DIR/longbench_hf_dataset.yaml" << EOF
# LongBench v2 Configuration using HuggingFace dataset
# Uses the dataset at: $EXISTING_LONGBENCH

model_name_or_path: meta-llama/Llama-2-7b-hf
finetuning_type: full
task: longbench_hf_dataset
save_dir: saves/longbench_hf_dataset
template: llama2
batch_size: 1

# LongBench settings
longbench_mode: standard
longbench_max_length: 32768
longbench_max_samples: 10
longbench_temperature: 0.1
longbench_max_new_tokens: 128

# Since we're using HuggingFace dataset directly, 
# we don't need the GitHub repo for the dataset itself.
# The evaluator will load from 'THUDM/LongBench-v2' on HuggingFace.
EOF

echo -e "${GREEN}✓ Created example configuration: longbench_hf_dataset.yaml${NC}"

# Check if we need prompt files from the GitHub repo
GITHUB_REPO="https://github.com/THUDM/LongBench.git"
PROMPTS_DIR="$EVAL_DIR/prompts"

if [ ! -d "$PROMPTS_DIR" ]; then
    echo -e "${BLUE}Downloading prompt templates from GitHub...${NC}"
    
    # Create a temporary directory
    TEMP_DIR=$(mktemp -d)
    
    # Clone only the prompts directory (sparse checkout)
    cd "$TEMP_DIR"
    git clone --no-checkout --depth 1 --filter=blob:none --sparse "$GITHUB_REPO" LongBench
    cd LongBench
    git sparse-checkout set prompts
    git checkout
    
    # Copy prompts to evaluation directory
    if [ -d "prompts" ]; then
        cp -r prompts "$EVAL_DIR/"
        echo -e "${GREEN}✓ Copied prompt templates to: $PROMPTS_DIR${NC}"
    else
        echo -e "${YELLOW}⚠ Could not find prompts directory in repository${NC}"
    fi
    
    # Clean up
    cd "$SCRIPT_DIR"
    rm -rf "$TEMP_DIR"
else
    echo -e "${GREEN}✓ Prompt templates already exist at: $PROMPTS_DIR${NC}"
fi

# Create a note about the dataset
cat > "$EVAL_DIR/DATASET_INFO.md" << EOF
# LongBench v2 Dataset Information

## Dataset Location
The LongBench v2 dataset is stored at: $EXISTING_LONGBENCH

This is the HuggingFace dataset version downloaded from:
https://huggingface.co/datasets/THUDM/LongBench-v2

## Dataset Structure
The dataset contains 503 challenging multiple-choice questions across 6 domains:
- Single-Document QA
- Multi-Document QA  
- Long In-context Learning
- Long-dialogue History Understanding
- Code Repo Understanding
- Long Structured Data Understanding

## Usage
The dataset is automatically loaded by the LongBench evaluator using:
\`\`\`python
from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench-v2', split='train')
\`\`\`

## Symlink
A symlink has been created at:
$SYMLINK_PATH -> $EXISTING_LONGBENCH

This allows the evaluation scripts to reference the dataset consistently.
EOF

echo -e "${GREEN}✓ Created dataset information file${NC}"

# Summary
echo
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo
echo -e "${BLUE}Dataset Location:${NC}"
echo -e "  HuggingFace Dataset: $EXISTING_LONGBENCH"
echo -e "  Symlink: $SYMLINK_PATH"
echo
echo -e "${BLUE}Configuration:${NC}"
echo -e "  The LongBench evaluator will automatically use the HuggingFace dataset"
echo -e "  No need to specify longbench_repo_path in YAML files"
echo
echo -e "${BLUE}Test the setup:${NC}"
echo -e "  llamafactory-cli eval longbench_minimal_test.yaml"
echo -e "  llamafactory-cli eval longbench_hf_dataset.yaml"
echo
echo -e "${GREEN}✓ LongBench v2 is ready to use!${NC}"