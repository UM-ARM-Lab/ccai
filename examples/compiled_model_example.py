#!/usr/bin/env python3
"""
Example script demonstrating the compiled model caching system.

This script shows how to:
1. Load models with automatic compilation caching
2. Clear compilation cache when needed
3. Monitor cache usage

The compilation cache will significantly speed up model loading times
by avoiding recompilation of PyTorch models.
"""

import torch
import os
import sys
from pathlib import Path

# Add the ccai module to the Python path
ccai_path = Path(__file__).parent.parent
sys.path.insert(0, str(ccai_path))

from ccai.models.temporal import TemporalUnet, TemporalUNetContext


def demo_compilation_caching():
    """Demonstrate the compilation caching system."""
    
    # Create a sample model
    model = TemporalUnet(
        horizon=32,
        transition_dim=16,
        cond_dim=3,
        dim=32,
        dim_mults=(1, 2, 4),
        attention=False,
        context_dropout_p=0.25,
        trajectory_condition=False
    )
    
    # Set up compilation cache directory
    cache_dir = ccai_path / "compiled_models_cache"
    model.set_compilation_cache_dir(cache_dir)
    
    print(f"Compilation cache directory: {cache_dir}")
    print(f"Cache directory exists: {cache_dir.exists()}")
    
    # Create sample inputs
    batch_size = 4
    horizon = 32
    transition_dim = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    t = torch.randn(batch_size, device=device)
    x = torch.randn(batch_size, horizon, transition_dim, device=device)
    context = torch.randn(batch_size, 3, device=device)
    
    print(f"\nUsing device: {device}")
    print(f"Model on device: {next(model.parameters()).device}")
    
    # First call - will compile and cache
    print("\n--- First call (will compile) ---")
    import time
    start_time = time.time()
    
    with torch.no_grad():
        output1, _ = model.compiled_conditional_test(t, x, context)
    
    first_call_time = time.time() - start_time
    print(f"First call time: {first_call_time:.2f} seconds")
    print(f"Output shape: {output1.shape}")
    
    # Second call - should load from cache
    print("\n--- Second call (should load from cache) ---")
    start_time = time.time()
    
    with torch.no_grad():
        output2, _ = model.compiled_conditional_test(t, x, context)
    
    second_call_time = time.time() - start_time
    print(f"Second call time: {second_call_time:.2f} seconds")
    print(f"Outputs match: {torch.allclose(output1, output2, atol=1e-6)}")
    
    if first_call_time > 0:
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
    
    # Show cache contents
    print(f"\n--- Cache contents ---")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pt"))
        print(f"Number of cached files: {len(cache_files)}")
        for cache_file in cache_files[:5]:  # Show first 5 files
            print(f"  - {cache_file.name}")
        if len(cache_files) > 5:
            print(f"  ... and {len(cache_files) - 5} more files")
    else:
        print("Cache directory does not exist yet")
    
    return model


def clear_cache_example(model):
    """Demonstrate cache clearing."""
    print("\n--- Cache clearing example ---")
    
    # Clear the cache
    model.clear_compilation_cache()
    print("Cleared compilation cache")
    
    # The cache directory will still exist but compiled methods cache will be cleared
    print("Note: Compiled methods will be recompiled on next use")


def cache_management_tips():
    """Print tips for managing the compilation cache."""
    print("\n--- Cache Management Tips ---")
    print("1. The cache directory is automatically created when first used")
    print("2. Cache files are named based on model class and method name")
    print("3. Cache is invalidated when model state changes (via hash)")
    print("4. You can manually clear cache using model.clear_compilation_cache()")
    print("5. Cache directory can be safely deleted to free disk space")
    print("6. Consider setting PYTORCH_COMPILE_CACHE_SIZE environment variable")
    print("7. Cache is per-model-instance, not global")


if __name__ == "__main__":
    print("Compiled Model Caching Demo")
    print("=" * 50)
    
    try:
        model = demo_compilation_caching()
        clear_cache_example(model)
        cache_management_tips()
        
        print("\n--- Summary ---")
        print("✓ Compilation caching is now enabled")
        print("✓ Models will compile once and reuse compiled versions")
        print("✓ This significantly speeds up model loading in production")
        print("✓ Cache is automatically managed and invalidated when needed")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc() 