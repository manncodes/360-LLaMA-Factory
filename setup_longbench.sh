#!/bin/bash

# Setup script for LongBench v2 integration
# This script clones the LongBench repository and creates appropriate symlinks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
DEFAULT_CLONE_PATH="/exp/data/eval_data"
DEFAULT_LONGBENCH_DIR="LongBench"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOCAL_LONGBENCH_PATH="$SCRIPT_DIR/evaluation/longbench"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}LongBench v2 Setup for LlamaFactory${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Function to check if directory exists and is not empty
check_directory() {
    if [ -d "$1" ] && [ "$(ls -A $1)" ]; then
        return 0
    else
        return 1
    fi
}

# Parse command line arguments
CLONE_PATH="${1:-$DEFAULT_CLONE_PATH}"
FORCE_CLONE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path) CLONE_PATH="$2"; shift ;;
        --force) FORCE_CLONE=true ;;
        --help) 
            echo "Usage: $0 [--path /custom/path] [--force]"
            echo
            echo "Options:"
            echo "  --path    Custom path where LongBench will be cloned (default: $DEFAULT_CLONE_PATH)"
            echo "  --force   Force re-clone even if directory exists"
            echo
            echo "Examples:"
            echo "  $0                                    # Use default path"
            echo "  $0 --path /home/user/benchmarks      # Custom path"
            echo "  $0 --force                            # Force re-clone"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Ensure the clone path exists
echo -e "${YELLOW}Setting up directories...${NC}"
mkdir -p "$CLONE_PATH"
mkdir -p "$LOCAL_LONGBENCH_PATH"

# Full path to LongBench
LONGBENCH_FULL_PATH="$CLONE_PATH/$DEFAULT_LONGBENCH_DIR"

# Check if LongBench already exists
if check_directory "$LONGBENCH_FULL_PATH" && [ "$FORCE_CLONE" = false ]; then
    echo -e "${GREEN}✓ LongBench already exists at: $LONGBENCH_FULL_PATH${NC}"
    echo -e "${YELLOW}  Use --force to re-clone${NC}"
else
    # Clone LongBench repository
    echo -e "${YELLOW}Cloning LongBench v2 repository...${NC}"
    echo -e "${YELLOW}Target: $LONGBENCH_FULL_PATH${NC}"
    
    # Remove existing directory if force clone
    if [ "$FORCE_CLONE" = true ] && [ -d "$LONGBENCH_FULL_PATH" ]; then
        echo -e "${YELLOW}Removing existing directory...${NC}"
        rm -rf "$LONGBENCH_FULL_PATH"
    fi
    
    # Clone the repository
    git clone https://github.com/THUDM/LongBench.git "$LONGBENCH_FULL_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully cloned LongBench repository${NC}"
    else
        echo -e "${RED}✗ Failed to clone LongBench repository${NC}"
        exit 1
    fi
fi

# Create symlink in the evaluation directory
SYMLINK_PATH="$LOCAL_LONGBENCH_PATH/LongBench"

echo -e "${YELLOW}Creating symlink...${NC}"
if [ -L "$SYMLINK_PATH" ]; then
    echo -e "${YELLOW}Removing existing symlink...${NC}"
    rm "$SYMLINK_PATH"
elif [ -e "$SYMLINK_PATH" ]; then
    echo -e "${RED}✗ $SYMLINK_PATH exists and is not a symlink!${NC}"
    echo -e "${RED}  Please remove it manually or choose a different location${NC}"
    exit 1
fi

# Create the symlink
ln -s "$LONGBENCH_FULL_PATH" "$SYMLINK_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Created symlink: $SYMLINK_PATH -> $LONGBENCH_FULL_PATH${NC}"
else
    echo -e "${RED}✗ Failed to create symlink${NC}"
    exit 1
fi

# Copy prompt files to local directory as backup
if [ -d "$LONGBENCH_FULL_PATH/prompts" ]; then
    echo -e "${YELLOW}Copying prompt files as local backup...${NC}"
    cp -r "$LONGBENCH_FULL_PATH/prompts" "$LOCAL_LONGBENCH_PATH/" 2>/dev/null || true
    echo -e "${GREEN}✓ Prompt files copied to: $LOCAL_LONGBENCH_PATH/prompts${NC}"
fi

# Create example configuration
EXAMPLE_CONFIG="$SCRIPT_DIR/longbench_with_symlink.yaml"
cat > "$EXAMPLE_CONFIG" << EOF
# LongBench v2 Configuration using symlinked repository
# This configuration uses the default symlinked path

model_name_or_path: meta-llama/Llama-2-7b-hf
finetuning_type: full
task: longbench_symlink_test
save_dir: saves/longbench_symlink_test
template: llama2
batch_size: 1

# LongBench settings
longbench_mode: standard
longbench_max_length: 32768
longbench_max_samples: 10
longbench_temperature: 0.1

# Using default path (symlink created by setup script)
# No need to specify longbench_repo_path - it will use:
# evaluation/longbench/LongBench (symlink to $LONGBENCH_FULL_PATH)
EOF

echo -e "${GREEN}✓ Created example configuration: $EXAMPLE_CONFIG${NC}"

# Summary
echo
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo
echo -e "${BLUE}LongBench Location:${NC}"
echo -e "  Repository: $LONGBENCH_FULL_PATH"
echo -e "  Symlink:    $SYMLINK_PATH"
echo
echo -e "${BLUE}Usage Options:${NC}"
echo
echo -e "1. ${YELLOW}Use default (symlink):${NC}"
echo -e "   # No configuration needed - will use symlink automatically"
echo -e "   llamafactory-cli eval longbench_standard.yaml"
echo
echo -e "2. ${YELLOW}Use custom path:${NC}"
echo -e "   # Add to your YAML configuration:"
echo -e "   longbench_repo_path: $LONGBENCH_FULL_PATH"
echo
echo -e "3. ${YELLOW}Test the setup:${NC}"
echo -e "   llamafactory-cli eval longbench_with_symlink.yaml"
echo
echo -e "${GREEN}✓ LongBench v2 is ready to use!${NC}"

# Check if prompts exist
if [ -d "$LONGBENCH_FULL_PATH/prompts" ]; then
    PROMPT_COUNT=$(ls -1 "$LONGBENCH_FULL_PATH/prompts"/*.txt 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $PROMPT_COUNT prompt templates${NC}"
else
    echo -e "${YELLOW}⚠ Warning: prompts directory not found in LongBench repository${NC}"
    echo -e "${YELLOW}  You may need to check the repository structure${NC}"
fi