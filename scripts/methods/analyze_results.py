#!/usr/bin/env python3

import json
import sys
from pathlib import Path

def analyze_method_results(method, result_dir):
    """Analyze results for a single method"""
    summary_file = Path(result_dir) / 'summary.json'
    
    if not summary_file.exists():
        return None
    
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        overall = data.get('overall', {})
        accuracy = overall.get('average_score', 0)
        total_examples = overall.get('total_examples', 0)
        
        print(f'  Overall: {accuracy:.1%} ({total_examples} examples)')
        
        by_length = data.get('by_context_length', {})
        if by_length:
            print('  By context length:')
            lengths = sorted(by_length.keys(), key=int)
            
            for length in lengths:
                stats = by_length[length]
                acc = stats.get('average_score', 0)
                count = stats.get('count', 0)
                
                status = 'GOOD' if acc >= 0.8 else 'FAIR' if acc >= 0.5 else 'POOR'
                print(f'    {status:4} {int(length):>6}T: {acc:.1%} ({count})')
            
            # Performance degradation
            if len(lengths) > 1:
                first_acc = by_length[lengths[0]].get('average_score', 0)
                last_acc = by_length[lengths[-1]].get('average_score', 0)
                
                if first_acc > 0:
                    drop = (first_acc - last_acc) / first_acc * 100
                    print(f'  Performance drop: {drop:.1f}% from {lengths[0]} to {lengths[-1]} tokens')
                else:
                    print(f'  Max context tested: {lengths[-1]} tokens')
        
        # Return data for CSV logging
        max_context = max([int(k) for k in by_length.keys()]) if by_length else 0
        return {
            'method': method,
            'status': 'SUCCESS',
            'accuracy': accuracy,
            'total_examples': total_examples,
            'max_context': max_context
        }
        
    except Exception as e:
        print(f'  Error: {e}')
        return {
            'method': method,
            'status': 'ERROR',
            'accuracy': 0,
            'total_examples': 0,
            'max_context': 0
        }

def generate_performance_analysis(results_dir):
    """Generate comprehensive performance analysis"""
    methods = ['baseline', 'linear', 'dynamic', 'yarn', 'longrope', 'nope']
    results_found = False
    
    print('\nPERFORMANCE ANALYSIS:')
    print('=' * 50)
    
    for method in methods:
        result_dir = Path(f'saves/methodwise/{method}')
        summary_file = result_dir / 'summary.json'
        
        if summary_file.exists():
            results_found = True
            try:
                with open(summary_file) as f:
                    data = json.load(f)
                
                print(f'\n{method.upper()}:')
                overall = data.get('overall', {})
                print(f'  Overall: {overall.get("average_score", 0):.1%} accuracy ({overall.get("total_examples", 0)} examples)')
                
                by_length = data.get('by_context_length', {})
                if by_length:
                    print('  Performance by context length:')
                    lengths = sorted(by_length.keys(), key=int)
                    
                    for length in lengths:
                        stats = by_length[length]
                        acc = stats.get('average_score', 0)
                        count = stats.get('count', 0)
                        print(f'    {int(length):>6} tokens: {acc:.1%} ({count})')
                    
                    # Performance degradation analysis
                    if len(lengths) > 1:
                        first_acc = by_length[lengths[0]].get('average_score', 0)
                        last_acc = by_length[lengths[-1]].get('average_score', 0)
                        
                        if first_acc > 0:
                            drop = (first_acc - last_acc) / first_acc * 100
                            print(f'  Performance drop: {drop:.1f}% from {lengths[0]} to {lengths[-1]} tokens')
                        else:
                            print(f'  Max context tested: {lengths[-1]} tokens')
                            
            except Exception as e:
                print(f'  Error reading {method}: {e}')
        else:
            print(f'\n{method.upper()}: No results found')
    
    if not results_found:
        print('\nNo successful results found')
        
    print(f'\nResults directory: {results_dir}')
    print('Detailed data in: saves/methodwise/[method]/summary.json')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        method = sys.argv[1]
        result_dir = sys.argv[2] if len(sys.argv) > 2 else f'saves/methodwise/{method}'
        result = analyze_method_results(method, result_dir)
        if result:
            print(json.dumps(result))
    else:
        results_dir = sys.argv[1] if len(sys.argv) > 1 else 'saves/methodwise'
        generate_performance_analysis(results_dir)