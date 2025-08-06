#!/usr/bin/env python3
"""
Monitor benchmark progress and show live updates
"""

import time
import os
import subprocess
import json
from pathlib import Path
import glob

def check_tmux_session():
    """Check if benchmark tmux session exists"""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], 
                              capture_output=True, text=True)
        return 'benchmark' in result.stdout
    except:
        return False

def get_tmux_output():
    """Get current tmux session output"""
    try:
        result = subprocess.run(['tmux', 'capture-pane', '-t', 'benchmark', '-p'],
                              capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Unable to capture tmux output"

def find_results_directory():
    """Find the current benchmark results directory"""
    results_dirs = glob.glob("complete_benchmark_*")
    if results_dirs:
        return max(results_dirs)  # Get the most recent one
    return None

def check_results_status():
    """Check status of benchmark results"""
    results_dir = find_results_directory()
    if not results_dir:
        return "No results directory found yet"
    
    status = {}
    methods = ['baseline', 'rope_linear', 'rope_dynamic', 'yarn', 'longrope', 'nope']
    
    for method in methods:
        method_dir = Path(results_dir) / method
        if method_dir.exists():
            if list(method_dir.glob("*.json")):
                status[method] = "✅ Completed"
            elif (method_dir / "benchmark_log.txt").exists():
                status[method] = "🔄 In Progress"
            else:
                status[method] = "📋 Scheduled"
        else:
            status[method] = "⏳ Waiting"
    
    return status

def monitor_benchmark():
    """Main monitoring loop"""
    print("🚀 Long Context Methods Benchmark Monitor")
    print("=" * 60)
    
    if not check_tmux_session():
        print("❌ No benchmark tmux session found. Starting benchmark...")
        os.system("tmux new-session -d -s benchmark './run_complete_benchmarks.sh'")
        time.sleep(2)
    
    last_output = ""
    iteration = 0
    
    while True:
        iteration += 1
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"🚀 Long Context Benchmark Monitor (Update #{iteration})")
        print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Check if tmux session is still running
        if not check_tmux_session():
            print("✅ Benchmark completed or stopped!")
            break
        
        # Show results status
        print("\n📊 Method Status:")
        status = check_results_status()
        if isinstance(status, dict):
            for method, state in status.items():
                print(f"   {method:12} : {state}")
        else:
            print(f"   {status}")
        
        # Show recent tmux output
        print("\n📺 Recent Output:")
        print("-" * 40)
        tmux_output = get_tmux_output()
        lines = tmux_output.split('\n')
        recent_lines = lines[-15:]  # Show last 15 lines
        for line in recent_lines:
            print(f"   {line}")
        
        # Show memory usage if available
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"\n🖥️  GPU Status: {gpu_info} (Used/Total MB, Util%)")
        except:
            pass
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop monitoring (benchmark will continue)")
        
        # Wait before next update
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_benchmark()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped. Benchmark continues in tmux session 'benchmark'")
        print("To reconnect: tmux attach -t benchmark")
        print("To check progress: tmux capture-pane -t benchmark -p")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")