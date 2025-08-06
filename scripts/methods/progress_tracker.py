#!/usr/bin/env python3

import sys
import time

def show_progress_bar(current, total, method, width=50):
    """Display a progress bar for current method"""
    percent = float(current) / total
    filled_width = int(width * percent)
    
    bar = '#' * filled_width + '-' * (width - filled_width)
    percent_str = f"{percent * 100:.1f}%"
    
    sys.stdout.write(f'\r[{bar}] {percent_str} - {method}')
    sys.stdout.flush()
    
    if current == total:
        print()  # New line when complete

def log_method_start(method, method_num, total_methods, context_lengths):
    """Log the start of a method evaluation"""
    print(f'\nMethod {method_num}/{total_methods}: {method.upper()}')
    print(f'Testing contexts: {context_lengths}')
    print('-' * 50)

def log_method_complete(method, status, duration):
    """Log the completion of a method evaluation"""
    status_symbol = '[OK]' if status == 'SUCCESS' else '[FAIL]'
    print(f'{status_symbol} {method}: {status} in {duration}s')

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        current = int(sys.argv[1])
        total = int(sys.argv[2])
        method = sys.argv[3]
        show_progress_bar(current, total, method)