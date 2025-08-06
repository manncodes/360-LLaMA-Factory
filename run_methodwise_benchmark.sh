#!/bin/bash

# Simple orchestration script - all logic is in Python
# Method-wise Long Context Benchmark

MODEL="/exp/model/Huggingface/meta-llama/Llama-3.2-1B"

# Ensure required directories exist
mkdir -p saves/methodwise
mkdir -p scripts/methods

# Check if Python runner exists
if [ ! -f "scripts/methods/benchmark_runner.py" ]; then
    echo "Error: benchmark_runner.py not found in scripts/methods/"
    exit 1
fi

# Install required packages if needed
python3 -c "import tqdm, yaml" 2>/dev/null || pip install tqdm PyYAML

# Run the benchmark
echo "Starting method-wise benchmark..."
python3 scripts/methods/benchmark_runner.py "$MODEL"

echo "Benchmark complete!"