#!/usr/bin/env python3
"""
Example script demonstrating how to use the optimized diffusion model with
model warmup and mixed precision for maximum inference speed.

Usage:
    python examples/optimization_example.py
"""

import torch
import numpy as np
from ccai.models.trajectory_samplers_sac import TrajectorySampler

def main():
    print("=== Diffusion Model Optimization Example ===")
    
    # Model parameters
    T = 16  # Horizon
    dx = 15  # State dimension
    du = 21  # Action dimension  
    context_dim = 3  # Context dimension
    timesteps = 20  # Diffusion timesteps
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trajectory sampler
    print("\n1. Creating TrajectorySampler...")
    sampler = TrajectorySampler(
        T=T, dx=dx, du=du, context_dim=context_dim,
        type='diffusion',
        timesteps=timesteps,
        hidden_dim=64,
        new_projection=True,  # Enable new projection method
        dropout_p=0.25
    )
    
    # Move to device
    print(f"2. Moving model to {device}...")
    sampler = sampler.to(device)
    
    # Set normalization constants (usually done during training)
    print("3. Setting normalization constants...")
    x_mean = torch.zeros(dx + du, device=device)
    x_std = torch.ones(dx + du, device=device) 
    sampler.set_norm_constants(x_mean, x_std)
    
    # Enable all optimizations automatically
    print("\n4. Enabling optimizations...")
    sampler.auto_warmup_and_optimize(
        warmup_batch_size=16,
        enable_mixed_precision=True
    )
    
    # Check optimization status
    print("\n5. Optimization Status:")
    stats = sampler.get_optimization_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Benchmark inference speed
    print("\n6. Benchmarking inference speed...")
    
    # Create test inputs
    N = 16  # Number of samples
    start_state = torch.randn(1, dx, device=device)
    constraints = torch.ones(N, 3, device=device)
    
    # Warm up timing
    print("   Running warmup samples...")
    for _ in range(3):
        with torch.no_grad():
            _ = sampler.sample(N=N, H=T, start=start_state, constraints=constraints, 
                             skip_likelihood=True)
    
    # Time the inference
    num_runs = 10
    print(f"   Timing {num_runs} inference runs...")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        start_time.record()
    else:
        import time
        start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_runs):
            samples, _, likelihood = sampler.sample(
                N=N, H=T, start=start_state, constraints=constraints
            )
            
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    else:
        elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / num_runs
    print(f"   Average inference time: {avg_time:.4f} seconds")
    print(f"   Samples per second: {N / avg_time:.2f}")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Likelihood shape: {likelihood.shape}")
    
    # Test different configurations
    print("\n7. Testing different optimization configurations...")
    
    # Test without mixed precision
    print("   Testing without mixed precision...")
    sampler.set_mixed_precision(False)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        start_time.record()
    else:
        start_time = time.time()
        
    with torch.no_grad():
        for i in range(5):
            _ = sampler.sample(N=N, H=T, start=start_state, constraints=constraints, 
                             skip_likelihood=True)
            
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time_no_mp = start_time.elapsed_time(end_time) / 1000.0 / 5
    else:
        elapsed_time_no_mp = (time.time() - start_time) / 5
        
    print(f"   Without mixed precision: {elapsed_time_no_mp:.4f} seconds")
    
    # Re-enable mixed precision
    sampler.set_mixed_precision(True)
    
    # Speed comparison
    if avg_time > 0 and elapsed_time_no_mp > 0:
        speedup = elapsed_time_no_mp / avg_time
        print(f"   Mixed precision speedup: {speedup:.2f}x")
    
    print("\n=== Optimization Example Complete ===")
    print("\nTo use in your code:")
    print("1. Create TrajectorySampler")
    print("2. Move to device with .to(device)")
    print("3. Call sampler.auto_warmup_and_optimize()")
    print("4. Use sampler.sample() for fast inference")

if __name__ == "__main__":
    main() 