#!/bin/bash

# Test the new Python benchmark runner locally

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEMPERATURE="${1:-0.0}"  # Default to deterministic for testing

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

echo "Starting test benchmark with new Python runner..."
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"

# Run the benchmark with local model
python3 scripts/methods/benchmark_runner.py "$MODEL" "" "$TEMPERATURE"

echo "Test benchmark complete!"