#!/bin/bash

# Advanced setup script for LongBench v2 with multiple options
# Supports: clone, symlink, copy, and environment variable setup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_FILE="$SCRIPT_DIR/.longbench_config"
LOCAL_EVAL_DIR="$SCRIPT_DIR/evaluation/longbench"

# Default values
DEFAULT_INSTALL_PATH="/exp/data/eval_data/LongBench"
DEFAULT_MODE="symlink"

echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   LongBench v2 Advanced Setup Script     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --mode MODE         Setup mode: clone, symlink, copy, check (default: $DEFAULT_MODE)
    --path PATH         Installation path (default: $DEFAULT_INSTALL_PATH)
    --source PATH       Source path for symlink/copy modes
    --force             Force operation even if files exist
    --env               Set up environment variable
    --config            Save configuration for future use
    --help              Show this help message

MODES:
    clone       Clone LongBench from GitHub to specified path
    symlink     Create symlink from existing LongBench to project
    copy        Copy LongBench to project (for environments without symlink support)
    check       Check current setup status

EXAMPLES:
    # Clone to default location and create symlink
    $0 --mode clone

    # Create symlink from existing installation
    $0 --mode symlink --source /data/LongBench

    # Clone to custom location with environment setup
    $0 --mode clone --path ~/benchmarks/LongBench --env

    # Check current setup
    $0 --mode check

EOF
}

# Parse arguments
MODE="$DEFAULT_MODE"
INSTALL_PATH="$DEFAULT_INSTALL_PATH"
SOURCE_PATH=""
FORCE=false
SETUP_ENV=false
SAVE_CONFIG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        --source)
            SOURCE_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --env)
            SETUP_ENV=true
            shift
            ;;
        --config)
            SAVE_CONFIG=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Load saved configuration if exists
if [ -f "$CONFIG_FILE" ] && [ "$MODE" = "symlink" ] && [ -z "$SOURCE_PATH" ]; then
    echo -e "${BLUE}Loading saved configuration...${NC}"
    source "$CONFIG_FILE"
    if [ -n "$LONGBENCH_PATH" ]; then
        SOURCE_PATH="$LONGBENCH_PATH"
        echo -e "${GREEN}✓ Using saved path: $SOURCE_PATH${NC}"
    fi
fi

# Function to check if LongBench is valid
check_longbench() {
    local path="$1"
    if [ -d "$path" ] && [ -d "$path/prompts" ] && [ -f "$path/README.md" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check current setup
check_setup() {
    echo -e "${BLUE}Checking LongBench setup...${NC}"
    echo
    
    # Check symlink
    local symlink_path="$LOCAL_EVAL_DIR/LongBench"
    if [ -L "$symlink_path" ]; then
        local target=$(readlink -f "$symlink_path")
        echo -e "${GREEN}✓ Symlink found:${NC}"
        echo -e "  $symlink_path -> $target"
        if check_longbench "$target"; then
            echo -e "  ${GREEN}✓ Valid LongBench installation${NC}"
        else
            echo -e "  ${RED}✗ Invalid or incomplete installation${NC}"
        fi
    else
        echo -e "${YELLOW}✗ No symlink found at: $symlink_path${NC}"
    fi
    echo
    
    # Check local prompts
    if [ -d "$LOCAL_EVAL_DIR/prompts" ]; then
        local prompt_count=$(ls -1 "$LOCAL_EVAL_DIR/prompts"/*.txt 2>/dev/null | wc -l || echo 0)
        echo -e "${GREEN}✓ Local prompts backup: $prompt_count files${NC}"
    else
        echo -e "${YELLOW}✗ No local prompts backup${NC}"
    fi
    echo
    
    # Check environment variable
    if [ -n "$LONGBENCH_PATH" ]; then
        echo -e "${GREEN}✓ Environment variable set: LONGBENCH_PATH=$LONGBENCH_PATH${NC}"
    else
        echo -e "${YELLOW}✗ Environment variable LONGBENCH_PATH not set${NC}"
    fi
    echo
    
    # Check saved config
    if [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}✓ Configuration file found${NC}"
        cat "$CONFIG_FILE" | sed 's/^/  /'
    else
        echo -e "${YELLOW}✗ No saved configuration${NC}"
    fi
}

# Function to clone LongBench
clone_longbench() {
    echo -e "${BLUE}Cloning LongBench repository...${NC}"
    
    if [ -d "$INSTALL_PATH" ] && [ "$FORCE" = false ]; then
        if check_longbench "$INSTALL_PATH"; then
            echo -e "${GREEN}✓ Valid LongBench already exists at: $INSTALL_PATH${NC}"
            echo -e "${YELLOW}  Use --force to re-clone${NC}"
            return 0
        fi
    fi
    
    # Create parent directory
    mkdir -p "$(dirname "$INSTALL_PATH")"
    
    # Remove existing if force
    if [ -d "$INSTALL_PATH" ] && [ "$FORCE" = true ]; then
        echo -e "${YELLOW}Removing existing directory...${NC}"
        rm -rf "$INSTALL_PATH"
    fi
    
    # Clone repository
    echo -e "${YELLOW}Cloning to: $INSTALL_PATH${NC}"
    git clone https://github.com/THUDM/LongBench.git "$INSTALL_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully cloned LongBench${NC}"
        SOURCE_PATH="$INSTALL_PATH"
        create_symlink
    else
        echo -e "${RED}✗ Failed to clone repository${NC}"
        exit 1
    fi
}

# Function to create symlink
create_symlink() {
    local symlink_path="$LOCAL_EVAL_DIR/LongBench"
    
    if [ -z "$SOURCE_PATH" ]; then
        echo -e "${RED}✗ Source path not specified for symlink${NC}"
        echo -e "${YELLOW}  Use --source /path/to/LongBench${NC}"
        exit 1
    fi
    
    if ! check_longbench "$SOURCE_PATH"; then
        echo -e "${RED}✗ Invalid LongBench at: $SOURCE_PATH${NC}"
        echo -e "${YELLOW}  Directory must contain prompts/ and README.md${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Creating symlink...${NC}"
    
    # Create directory if needed
    mkdir -p "$LOCAL_EVAL_DIR"
    
    # Remove existing symlink
    if [ -L "$symlink_path" ]; then
        rm "$symlink_path"
    elif [ -e "$symlink_path" ] && [ "$FORCE" = true ]; then
        rm -rf "$symlink_path"
    elif [ -e "$symlink_path" ]; then
        echo -e "${RED}✗ $symlink_path exists and is not a symlink${NC}"
        echo -e "${YELLOW}  Use --force to overwrite${NC}"
        exit 1
    fi
    
    # Create symlink
    ln -s "$SOURCE_PATH" "$symlink_path"
    echo -e "${GREEN}✓ Created symlink: $symlink_path -> $SOURCE_PATH${NC}"
    
    # Copy prompts as backup
    copy_prompts_backup
}

# Function to copy LongBench
copy_longbench() {
    if [ -z "$SOURCE_PATH" ]; then
        echo -e "${RED}✗ Source path not specified for copy${NC}"
        echo -e "${YELLOW}  Use --source /path/to/LongBench${NC}"
        exit 1
    fi
    
    if ! check_longbench "$SOURCE_PATH"; then
        echo -e "${RED}✗ Invalid LongBench at: $SOURCE_PATH${NC}"
        exit 1
    fi
    
    local target_path="$LOCAL_EVAL_DIR/LongBench"
    
    echo -e "${BLUE}Copying LongBench...${NC}"
    echo -e "${YELLOW}Source: $SOURCE_PATH${NC}"
    echo -e "${YELLOW}Target: $target_path${NC}"
    
    # Create directory
    mkdir -p "$LOCAL_EVAL_DIR"
    
    # Remove existing if force
    if [ -e "$target_path" ] && [ "$FORCE" = true ]; then
        rm -rf "$target_path"
    elif [ -e "$target_path" ]; then
        echo -e "${RED}✗ Target already exists${NC}"
        echo -e "${YELLOW}  Use --force to overwrite${NC}"
        exit 1
    fi
    
    # Copy files
    cp -r "$SOURCE_PATH" "$target_path"
    echo -e "${GREEN}✓ Copied LongBench to project${NC}"
}

# Function to copy prompts as backup
copy_prompts_backup() {
    if [ -d "$SOURCE_PATH/prompts" ]; then
        echo -e "${BLUE}Creating prompts backup...${NC}"
        mkdir -p "$LOCAL_EVAL_DIR/prompts"
        cp -r "$SOURCE_PATH/prompts"/* "$LOCAL_EVAL_DIR/prompts/" 2>/dev/null || true
        local count=$(ls -1 "$LOCAL_EVAL_DIR/prompts"/*.txt 2>/dev/null | wc -l || echo 0)
        echo -e "${GREEN}✓ Backed up $count prompt files${NC}"
    fi
}

# Function to setup environment
setup_environment() {
    echo -e "${BLUE}Setting up environment...${NC}"
    
    local env_line="export LONGBENCH_PATH=\"$SOURCE_PATH\""
    local shell_rc=""
    
    # Determine shell configuration file
    if [ -n "$BASH_VERSION" ]; then
        shell_rc="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        shell_rc="$HOME/.zshrc"
    else
        shell_rc="$HOME/.profile"
    fi
    
    # Check if already added
    if grep -q "LONGBENCH_PATH" "$shell_rc" 2>/dev/null; then
        echo -e "${YELLOW}Updating existing LONGBENCH_PATH in $shell_rc${NC}"
        sed -i.bak "/LONGBENCH_PATH/c\\$env_line" "$shell_rc"
    else
        echo -e "${YELLOW}Adding LONGBENCH_PATH to $shell_rc${NC}"
        echo "" >> "$shell_rc"
        echo "# LongBench v2 path for LlamaFactory" >> "$shell_rc"
        echo "$env_line" >> "$shell_rc"
    fi
    
    echo -e "${GREEN}✓ Environment variable added${NC}"
    echo -e "${YELLOW}  Run 'source $shell_rc' to apply changes${NC}"
}

# Function to save configuration
save_configuration() {
    echo -e "${BLUE}Saving configuration...${NC}"
    cat > "$CONFIG_FILE" << EOF
# LongBench configuration for LlamaFactory
# Generated on $(date)
LONGBENCH_PATH="$SOURCE_PATH"
INSTALL_MODE="$MODE"
EOF
    echo -e "${GREEN}✓ Configuration saved to: $CONFIG_FILE${NC}"
}

# Main execution
case "$MODE" in
    clone)
        clone_longbench
        ;;
    symlink)
        create_symlink
        ;;
    copy)
        copy_longbench
        ;;
    check)
        check_setup
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        show_usage
        exit 1
        ;;
esac

# Setup environment if requested
if [ "$SETUP_ENV" = true ] && [ -n "$SOURCE_PATH" ]; then
    setup_environment
fi

# Save configuration if requested
if [ "$SAVE_CONFIG" = true ] && [ -n "$SOURCE_PATH" ]; then
    save_configuration
fi

# Create example configuration
create_example_config() {
    local example_file="$SCRIPT_DIR/longbench_auto_setup.yaml"
    cat > "$example_file" << EOF
# LongBench v2 Configuration
# Auto-generated by setup script

model_name_or_path: meta-llama/Llama-2-7b-hf
finetuning_type: full
task: longbench_auto_setup
save_dir: saves/longbench_auto_setup
template: llama2
batch_size: 1

# LongBench settings
longbench_mode: standard
longbench_max_length: 32768
longbench_max_samples: 5

# Path configuration (choose one):

# Option 1: Use automatic detection (recommended)
# The system will check in order:
# 1. evaluation/longbench/LongBench (symlink/copy)
# 2. \$LONGBENCH_PATH environment variable
# 3. Default fallback paths

# Option 2: Explicit path
# longbench_repo_path: $SOURCE_PATH
EOF
    echo -e "${GREEN}✓ Created example: $example_file${NC}"
}

if [ -n "$SOURCE_PATH" ]; then
    create_example_config
fi

# Summary
echo
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Setup Complete!                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo
echo -e "${BLUE}Quick Test:${NC}"
echo -e "  llamafactory-cli eval longbench_minimal_test.yaml"
echo
if [ -n "$SOURCE_PATH" ]; then
    echo -e "${BLUE}LongBench Location:${NC}"
    echo -e "  $SOURCE_PATH"
fi
echo
echo -e "${GREEN}✓ LongBench v2 is ready to use!${NC}"