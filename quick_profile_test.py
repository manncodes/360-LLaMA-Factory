#!/usr/bin/env python3
"""
Quick test to verify profiling functionality.
"""

import os
import torch
import torch.nn as nn

def test_profiling():
    """Quick test of PyTorch profiler with a simple model."""
    
    print("Testing PyTorch profiler...")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    
    # Simple data
    batch_size = 4
    data = torch.randn(batch_size, 128)
    if torch.cuda.is_available():
        data = data.cuda()
    
    # Setup profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    os.makedirs("quick_test_traces", exist_ok=True)
    
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("quick_test_traces"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )
    
    print("Running profiled iterations...")
    
    with profiler:
        for step in range(5):
            # Forward pass
            output = model(data)
            loss = output.mean()
            
            # Backward pass
            loss.backward()
            
            # Step profiler
            profiler.step()
            
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    # Export Chrome trace
    chrome_trace_path = "quick_test_traces/test_trace.json"
    try:
        profiler.export_chrome_trace(chrome_trace_path)
        print(f"\nProfiling test successful!")
        print(f"Chrome trace saved to: {chrome_trace_path}")
        return True
    except Exception as e:
        print(f"Failed to export Chrome trace: {e}")
        return False

if __name__ == "__main__":
    success = test_profiling()
    if success:
        print("\nProfiler is working correctly!")
        print("Open chrome://tracing and load quick_test_traces/test_trace.json")
    else:
        print("\nProfiler test failed.")