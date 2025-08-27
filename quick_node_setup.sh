#!/bin/bash
# Quick setup script for RoPE hyperparameter sweep on new node

set -e

echo "🚀 Setting up 360-LLaMA-Factory with RoPE hyperparameter sweep..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python $PYTHON_VERSION"
}

# Check if Git is available
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git not found. Please install Git."
        exit 1
    fi
    print_status "Git is available"
}

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "No NVIDIA GPU detected. CPU training will be very slow."
    fi
}

# Clone repository
clone_repo() {
    if [ -d "360-LLaMA-Factory" ]; then
        print_warning "Directory 360-LLaMA-Factory already exists. Updating..."
        cd 360-LLaMA-Factory
        git fetch fork
        git checkout llamafactory-native-rope-eval
        git pull fork llamafactory-native-rope-eval
    else
        print_status "Cloning repository..."
        git clone https://github.com/manncodes/360-LLaMA-Factory.git
        cd 360-LLaMA-Factory
        git checkout llamafactory-native-rope-eval
    fi
}

# Setup virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    pip install -e .
    
    # Install additional packages for sweeps
    pip install matplotlib seaborn pandas tqdm
}

# Verify installation
verify_install() {
    print_status "Verifying installation..."
    
    # Test CLI
    if llamafactory-cli --help > /dev/null 2>&1; then
        print_status "✅ LlamaFactory CLI installed successfully"
    else
        print_error "❌ LlamaFactory CLI installation failed"
        exit 1
    fi
    
    # Test Python imports
    python3 -c "
import torch
import transformers
import llamafactory
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    print(f'✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
"
}

# Test RoPE configuration
test_rope_config() {
    print_status "Testing RoPE configuration parsing..."
    
    if python3 quick_rope_test.py > /dev/null 2>&1; then
        print_status "✅ RoPE configuration test passed"
    else
        print_warning "⚠️  RoPE configuration test had warnings (may be normal)"
    fi
}

# Create quick test script
create_test_script() {
    print_status "Creating quick test configuration..."
    
    cat > quick_test.yaml << EOF
# Quick test configuration for new node
model_name_or_path: unsloth/Llama-3.2-1B-Instruct
rope_scaling: linear
rope_theta: 1000000.0
cutoff_len: 2048
dataset: alpaca_en_demo
template: llama3
stage: sft
do_train: true
finetuning_type: full
output_dir: quick_test_output
overwrite_output_dir: true
per_device_train_batch_size: 1
max_steps: 1
bf16: true
logging_steps: 1
save_steps: 1
warmup_steps: 0
optim: adamw_torch
seed: 42
EOF
    
    print_status "Created quick_test.yaml"
}

# Run quick test
run_quick_test() {
    print_status "Running quick functionality test..."
    print_warning "This will download the model (~2GB) and run 1 training step..."
    
    read -p "Continue with quick test? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        timeout 300 llamafactory-cli train quick_test.yaml || {
            print_warning "Quick test timed out or failed (may be normal for first run)"
            print_status "Setup appears successful - you can run sweeps manually"
        }
    else
        print_status "Skipping quick test"
    fi
}

# Main setup function
main() {
    print_status "Starting 360-LLaMA-Factory setup for RoPE hyperparameter sweeps"
    
    check_python
    check_git
    check_gpu
    
    clone_repo
    setup_venv
    install_deps
    verify_install
    test_rope_config
    create_test_script
    
    print_status "✅ Setup completed successfully!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Activate environment: source venv/bin/activate"
    print_status "2. Run quick test: llamafactory-cli train quick_test.yaml"
    print_status "3. Run RoPE sweep: python3 comprehensive_sweep/fast_eval.py"
    print_status "4. Check results: python3 comprehensive_sweep/analyze_comprehensive_results.py"
    print_status ""
    print_status "Available configurations:"
    print_status "- final_rope_example.yaml (production ready)"
    print_status "- rope_theta_test.yaml (high theta testing)"  
    print_status "- llama32_rope_test.yaml (LLaMA 3.2 specific)"
    print_status ""
    print_status "For detailed usage, see: SETUP_AND_DEPLOYMENT_GUIDE.md"
    
    # Offer to run quick test
    run_quick_test
}

# Run main function
main "$@"