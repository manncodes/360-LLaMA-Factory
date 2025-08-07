#!/bin/bash

echo "Searching for PaulGraham essays..."
echo "=================================="

# Search from current directory
if [ -d "./evaluation/needle_haystack/data/PaulGrahamEssays" ]; then
    echo "✓ Found at: ./evaluation/needle_haystack/data/PaulGrahamEssays"
    echo "  Use: --haystack-path ./evaluation/needle_haystack/data/PaulGrahamEssays"
    ls -la ./evaluation/needle_haystack/data/PaulGrahamEssays/*.txt | wc -l | xargs echo "  Essay count:"
fi

# Search in parent directories
PARENT="../"
for i in {1..3}; do
    PATH_CHECK="${PARENT}LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays"
    if [ -d "$PATH_CHECK" ]; then
        echo "✓ Found at: $PATH_CHECK"
        echo "  Use: --haystack-path $PATH_CHECK"
        ls -la $PATH_CHECK/*.txt | wc -l | xargs echo "  Essay count:"
    fi
    PARENT="../$PARENT"
done

# Search with find (slower but thorough)
echo ""
echo "Searching with find (may take a moment)..."
FOUND_PATHS=$(find . -type d -name "PaulGrahamEssays" 2>/dev/null)

if [ ! -z "$FOUND_PATHS" ]; then
    echo "Found PaulGraham directories:"
    echo "$FOUND_PATHS" | while read -r path; do
        COUNT=$(ls -la "$path"/*.txt 2>/dev/null | wc -l)
        echo "  $path (contains $COUNT .txt files)"
    done
else
    echo "No PaulGrahamEssays directories found in current directory tree"
fi

echo ""
echo "Usage examples:"
echo "  python run_rope_needle_eval.py --haystack-path ./evaluation/needle_haystack/data/PaulGrahamEssays"
echo "  python run_rope_needle_eval.py --haystack-path /absolute/path/to/PaulGrahamEssays"