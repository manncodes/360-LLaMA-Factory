#!/bin/bash

# Simple orchestration script - all logic is in Python
# Method-wise Long Context Benchmark

MODEL="/exp/model/Huggingface/meta-llama/Llama-3.2-1B"
TEMPERATURE="${1:-0.0}"  # Default to deterministic (0.0) for reliable results

# Ensure required directories exist
mkdir -p saves/methodwise
mkdir -p scripts/methods

# Check if Python runner exists
if [ ! -f "scripts/methods/benchmark_runner.py" ]; then
    echo "Error: benchmark_runner.py not found in scripts/methods/"
    exit 1
fi

# Install required packages if needed
python3 -c "import yaml" 2>/dev/null || pip install PyYAML

# Run the benchmark
echo "Starting method-wise benchmark..."
echo "Temperature: $TEMPERATURE (0.0=deterministic, >0.0=sampling)"
python3 scripts/methods/benchmark_runner.py "$MODEL" "" "$TEMPERATURE"

echo "Benchmark complete!"