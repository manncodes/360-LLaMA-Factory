# HELMET Setup for Corporate Environment

## Quick Setup for Your Corporate PC

### 1. Environment Configuration

Set the HELMET data path to your corporate location:

```bash
export HELMET_DATA_PATH="/exp/data/eval_data/HELMET/data"
```

Add this to your `.bashrc` or `.profile` for persistence:
```bash
echo 'export HELMET_DATA_PATH="/exp/data/eval_data/HELMET/data"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Manual HELMET Repository Setup

Since corporate proxy blocks git clone, manually download HELMET:

```bash
# Create parent directory structure
cd /path/to/your/workspace
mkdir -p HELMET

# Download HELMET repository manually (via web interface)
# Extract to: /path/to/your/workspace/HELMET/

# Or if you have access to wget/curl with proxy:
# wget https://github.com/princeton-nlp/HELMET/archive/refs/heads/main.zip
# unzip main.zip
# mv HELMET-main HELMET
```

### 3. Run Setup Script

```bash
cd 360-LLaMA-Factory
bash setup_helmet.sh
```

The script will:
- Use your configured HELMET_DATA_PATH
- Skip downloading (uses existing data)
- Create symlinks if needed
- Check dependencies without installing
- Validate configurations

### 4. Expected Output

```
HELMET Benchmark Setup for LlamaFactory
=============================================
Checking directory structure...
   Current dir: /path/to/360-LLaMA-Factory
   Parent dir: /path/to/workspace
   Expected HELMET dir: /path/to/workspace/HELMET
   HELMET data path: /exp/data/eval_data/HELMET/data

HELMET repository found
HELMET data found at: /exp/data/eval_data/HELMET/data
Creating symlink to HELMET data...
Symlink created: /path/to/workspace/HELMET/data -> /exp/data/eval_data/HELMET/data

Checking Python dependencies...
yaml available
numpy available
datasets available
torch available
transformers available
All required dependencies available

Testing HELMET integration...
HELMET modules import successfully

Running configuration validation...
helmet_quick_demo.yaml valid
helmet_recall_config.yaml valid
helmet_longqa_config.yaml valid
helmet_comprehensive_config.yaml valid

HELMET setup validation completed

Next steps:
   1. Install missing dependencies if any
   2. Run quick test: llamafactory-cli eval test_helmet_setup.yaml
   3. Try demo config: llamafactory-cli eval helmet_quick_demo.yaml
   4. Run full evaluation: llamafactory-cli eval helmet_recall_config.yaml

Environment variables:
   - HELMET_DATA_PATH: /exp/data/eval_data/HELMET/data
   - HELMET_REPO_URL: https://github.com/princeton-nlp/HELMET.git

Setup validation complete. Ready to evaluate long-context models with HELMET.
```

### 5. Quick Test

```bash
# Test integration
llamafactory-cli eval helmet_quick_demo.yaml

# Check results
ls -la saves/helmet_quick_demo/
cat saves/helmet_quick_demo/helmet_metrics.json
```

### 6. Configuration Options

You can customize paths via environment variables:

```bash
# Custom data path
export HELMET_DATA_PATH="/your/custom/path/to/helmet/data"

# Custom repository URL (if using internal mirror)
export HELMET_REPO_URL="https://internal-git.company.com/HELMET.git"

# Run setup
bash setup_helmet.sh
```

### 7. Troubleshooting

**If symlink creation fails:**
```bash
# Manual symlink creation
ln -sf /exp/data/eval_data/HELMET/data /path/to/workspace/HELMET/data
```

**If HELMET repository is missing:**
- Download manually from GitHub web interface
- Extract to parent directory as `HELMET/`
- Ensure files like `eval.py`, `arguments.py` exist

**If Python dependencies are missing:**
```bash
pip install PyYAML numpy datasets torch transformers
```

### 8. Directory Structure

Expected final structure:
```
workspace/
├── 360-LLaMA-Factory/          # Your project
│   ├── src/llamafactory/eval/helmet_evaluator.py
│   ├── helmet_*.yaml           # Config files
│   └── setup_helmet.sh         # Setup script
├── HELMET/                     # HELMET repository
│   ├── eval.py
│   ├── arguments.py
│   ├── model_utils.py
│   ├── data.py
│   └── data/                   # Symlink to actual data
└── /exp/data/eval_data/HELMET/data/  # Actual data location
    ├── json_kv/
    ├── ruler/
    └── ...
```

The setup script is now optimized for corporate environments with existing data and proxy restrictions.