#!/bin/bash

# Setup script for LongBench v2 dataset symlink
# Links existing dataset to project directory

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Default paths
DEFAULT_DATASET_PATH="/exp/data/eval_data/LongBench-v2"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TARGET_DIR="$SCRIPT_DIR/data/longbench"

# Parse command line arguments
DATASET_PATH="${1:-$DEFAULT_DATASET_PATH}"
SYMLINK_PATH="$TARGET_DIR/LongBench-v2"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}LongBench v2 Dataset Setup${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Show usage if help requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [DATASET_PATH]"
    echo
    echo "Arguments:"
    echo "  DATASET_PATH    Path to LongBench v2 dataset (default: $DEFAULT_DATASET_PATH)"
    echo
    echo "Examples:"
    echo "  $0                                    # Use default path"
    echo "  $0 /custom/path/to/LongBench-v2      # Use custom path"
    echo "  $0 ~/data/LongBench-v2               # Use home directory path"
    exit 0
fi

echo "Using dataset path: $DATASET_PATH"
echo

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}✗ LongBench v2 dataset not found at: $DATASET_PATH${NC}"
    echo -e "${YELLOW}  Please ensure the dataset is available at this location${NC}"
    echo -e "${YELLOW}  Or specify a different path: $0 /path/to/LongBench-v2${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found LongBench v2 dataset at: $DATASET_PATH${NC}"

# Create target directory
mkdir -p "$TARGET_DIR"

# Remove existing symlink if it exists
if [ -L "$SYMLINK_PATH" ]; then
    echo -e "${YELLOW}Removing existing symlink...${NC}"
    rm "$SYMLINK_PATH"
elif [ -e "$SYMLINK_PATH" ]; then
    echo -e "${RED}✗ $SYMLINK_PATH exists and is not a symlink${NC}"
    echo -e "${YELLOW}  Please remove it manually${NC}"
    exit 1
fi

# Create symlink
echo -e "${BLUE}Creating symlink...${NC}"
ln -s "$DATASET_PATH" "$SYMLINK_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Created symlink: $SYMLINK_PATH -> $DATASET_PATH${NC}"
else
    echo -e "${RED}✗ Failed to create symlink${NC}"
    exit 1
fi

# Verify symlink works
if [ -d "$SYMLINK_PATH" ]; then
    echo -e "${GREEN}✓ Symlink verification successful${NC}"
    
    # Check for data files
    if ls "$SYMLINK_PATH"/*.json >/dev/null 2>&1; then
        file_count=$(ls -1 "$SYMLINK_PATH"/*.json 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ Found $file_count JSON data files${NC}"
    else
        echo -e "${YELLOW}⚠ No JSON files found in dataset directory${NC}"
    fi
else
    echo -e "${RED}✗ Symlink verification failed${NC}"
    exit 1
fi

# Create a simple test config
cat > "$SCRIPT_DIR/longbench_local.yaml" << EOF
# LongBench v2 Configuration using local dataset
# Uses the symlinked dataset at data/longbench/LongBench-v2

model_name_or_path: meta-llama/Llama-3.2-1B
finetuning_type: full
task: longbench
save_dir: saves/longbench_local
template: llama3

# LongBench specific settings
longbench_max_length: 8192    # Conservative context length  
longbench_max_samples: 3      # Only 3 samples for testing
longbench_use_local: true     # Force using local dataset
EOF

echo -e "${GREEN}✓ Created local configuration: longbench_local.yaml${NC}"

# Summary
echo
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo
echo -e "${BLUE}Dataset Location:${NC}"
echo -e "  Original: $DATASET_PATH"
echo -e "  Symlink:  $SYMLINK_PATH"
echo
echo -e "${BLUE}Usage:${NC}"
echo -e "  llamafactory-cli eval longbench_local.yaml"
echo
echo -e "${GREEN}✓ LongBench v2 is ready to use with local dataset!${NC}"