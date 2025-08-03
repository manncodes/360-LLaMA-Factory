# LongBench v2 Setup Guide

This guide explains how to set up LongBench v2 for use with 360-LLaMA-Factory.

## Quick Setup

### Option 1: Basic Setup (Recommended)
```bash
# Clone and create symlink with default settings
./setup_longbench.sh

# Or specify custom path
./setup_longbench.sh --path /your/custom/path
```

### Option 2: Advanced Setup
```bash
# Clone to specific location
./setup_longbench_advanced.sh --mode clone --path /exp/data/LongBench

# Create symlink from existing installation
./setup_longbench_advanced.sh --mode symlink --source /existing/LongBench

# Copy for environments without symlink support
./setup_longbench_advanced.sh --mode copy --source /existing/LongBench

# Check current setup status
./setup_longbench_advanced.sh --mode check
```

## Setup Options

### 1. Clone Mode
Downloads LongBench from GitHub and creates a symlink:
```bash
./setup_longbench_advanced.sh --mode clone --path /data/benchmarks/LongBench
```

### 2. Symlink Mode
Creates a symlink from an existing LongBench installation:
```bash
./setup_longbench_advanced.sh --mode symlink --source /shared/LongBench
```

### 3. Copy Mode
Copies LongBench into the project (for restricted environments):
```bash
./setup_longbench_advanced.sh --mode copy --source /shared/LongBench --force
```

### 4. Environment Setup
Automatically configure environment variables:
```bash
./setup_longbench_advanced.sh --mode clone --env --config
```

## Configuration in YAML

After setup, you have three options for using LongBench:

### 1. Automatic Detection (Default)
```yaml
# No configuration needed - uses symlink/copy automatically
model_name_or_path: meta-llama/Llama-2-7b-hf
task: longbench_test
# System will find LongBench automatically
```

### 2. Environment Variable
```bash
export LONGBENCH_PATH=/exp/data/eval_data/LongBench
```
```yaml
# Will use $LONGBENCH_PATH if set
model_name_or_path: meta-llama/Llama-2-7b-hf
task: longbench_test
```

### 3. Explicit Path
```yaml
model_name_or_path: meta-llama/Llama-2-7b-hf
task: longbench_test
longbench_repo_path: /exp/data/eval_data/LongBench
```

## Directory Structure

After setup, the structure will be:
```
360-LLaMA-Factory/
├── evaluation/
│   └── longbench/
│       ├── LongBench/          # Symlink to actual LongBench
│       ├── prompts/            # Backup of prompt templates
│       └── LLAMAFACTORY_README.md
├── setup_longbench.sh          # Basic setup script
├── setup_longbench_advanced.sh # Advanced setup script
└── .longbench_config          # Saved configuration
```

## Troubleshooting

### Permission Issues
```bash
# If you can't create symlinks
./setup_longbench_advanced.sh --mode copy --source /path/to/LongBench
```

### Corporate Proxy
```bash
# Set proxy before cloning
export https_proxy=http://proxy.company.com:8080
./setup_longbench.sh
```

### Verify Installation
```bash
# Check if setup is correct
./setup_longbench_advanced.sh --mode check

# Test with minimal config
llamafactory-cli eval longbench_minimal_test.yaml
```

## Advanced Usage

### Multiple LongBench Versions
```yaml
# Development version
longbench_repo_path: /data/LongBench-dev

# Stable version
longbench_repo_path: /data/LongBench-stable
```

### Cluster/HPC Setup
```bash
# Set up on shared filesystem
./setup_longbench_advanced.sh \
  --mode clone \
  --path /shared/benchmarks/LongBench \
  --env \
  --config
```

### Docker/Container Setup
```dockerfile
# In Dockerfile
COPY setup_longbench.sh /app/
RUN /app/setup_longbench.sh --path /data/LongBench
```

## Examples

### Quick Test After Setup
```bash
# 1. Run setup
./setup_longbench.sh

# 2. Test with 2 samples
llamafactory-cli eval longbench_minimal_test.yaml

# 3. Run full evaluation
llamafactory-cli eval longbench_standard.yaml
```

### Custom Path Example
```bash
# 1. Setup with custom path
./setup_longbench_advanced.sh \
  --mode clone \
  --path ~/my_benchmarks/LongBench \
  --env

# 2. Use in YAML
cat > my_config.yaml << EOF
model_name_or_path: meta-llama/Llama-2-7b-hf
task: longbench_custom
longbench_repo_path: ~/my_benchmarks/LongBench
longbench_max_samples: 10
EOF

# 3. Run evaluation
llamafactory-cli eval my_config.yaml
```

## Notes

- The setup scripts are idempotent - safe to run multiple times
- Prompts are backed up locally for reliability
- Configuration is saved for future reference
- Both scripts support `--force` to overwrite existing setups
- The advanced script supports environment variable configuration