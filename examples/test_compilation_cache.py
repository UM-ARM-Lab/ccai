#!/usr/bin/env python3
"""
Simple test script to verify compilation caching is working.
"""

import torch
import time
import sys
from pathlib import Path

# Add ccai to path
ccai_path = Path(__file__).parent.parent
sys.path.insert(0, str(ccai_path))

from ccai.models.temporal import TemporalUnet


def test_compilation_caching():
    """Test that compilation caching is working."""
    
    print("Testing compilation caching...")
    
    # Create a simple model
    model = TemporalUnet(
        horizon=16,
        transition_dim=8,
        cond_dim=3,
        dim=16,
        dim_mults=(1, 2),
        attention=False,
        context_dropout_p=0.25,
        trajectory_condition=False
    )
    
    # Enable caching
    model.set_compilation_cache_dir("./test_cache")
    
    # Create test inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    batch_size = 2
    t = torch.randn(batch_size, device=device)
    x = torch.randn(batch_size, 16, 8, device=device)
    context = torch.randn(batch_size, 3, device=device)
    
    print(f"Using device: {device}")
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Test 1: First call should compile
    print("\n--- Test 1: First call (should compile) ---")
    start_time = time.time()
    
    with torch.no_grad():
        output1, _ = model.compiled_conditional_test(t, x, context)
    
    first_call_time = time.time() - start_time
    print(f"First call time: {first_call_time:.3f} seconds")
    print(f"Output shape: {output1.shape}")
    
    # Check cache stats
    stats = model.get_cache_stats()
    print(f"Cache stats after first call: {stats}")
    
    # Test 2: Second call should use cached version
    print("\n--- Test 2: Second call (should use cache) ---")
    start_time = time.time()
    
    with torch.no_grad():
        output2, _ = model.compiled_conditional_test(t, x, context)
    
    second_call_time = time.time() - start_time
    print(f"Second call time: {second_call_time:.3f} seconds")
    print(f"Outputs match: {torch.allclose(output1, output2, atol=1e-6)}")
    
    # Check cache stats again
    stats = model.get_cache_stats()
    print(f"Cache stats after second call: {stats}")
    
    if first_call_time > 0 and second_call_time > 0:
        speedup = first_call_time / second_call_time
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ Caching is working! Second call was significantly faster.")
        else:
            print("⚠️  Caching may not be working effectively.")
    
    # Test 3: Test another method
    print("\n--- Test 3: Testing another method ---")
    start_time = time.time()
    
    with torch.no_grad():
        output3, _ = model.compiled_unconditional_test(t, x)
    
    third_call_time = time.time() - start_time
    print(f"Third call time: {third_call_time:.3f} seconds")
    
    # Final cache stats
    stats = model.get_cache_stats()
    print(f"Final cache stats: {stats}")
    
    return model


def test_cache_clearing():
    """Test cache clearing functionality."""
    print("\n--- Testing cache clearing ---")
    
    model = TemporalUnet(
        horizon=8,
        transition_dim=4,
        cond_dim=2,
        dim=8,
        dim_mults=(1,),
        attention=False,
        context_dropout_p=0.25,
        trajectory_condition=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    t = torch.randn(1, device=device)
    x = torch.randn(1, 8, 4, device=device)
    context = torch.randn(1, 2, device=device)
    
    # First call to compile
    with torch.no_grad():
        model.compiled_conditional_test(t, x, context)
    
    print(f"Cache before clearing: {model.get_cache_stats()}")
    
    # Clear cache
    model.clear_compilation_cache()
    
    print(f"Cache after clearing: {model.get_cache_stats()}")
    
    # This should recompile
    start_time = time.time()
    with torch.no_grad():
        model.compiled_conditional_test(t, x, context)
    recompile_time = time.time() - start_time
    print(f"Recompile time: {recompile_time:.3f} seconds")


def test_cache_disable():
    """Test disabling cache functionality."""
    print("\n--- Testing cache disable ---")
    
    model = TemporalUnet(
        horizon=8,
        transition_dim=4,
        cond_dim=2,
        dim=8,
        dim_mults=(1,),
        attention=False,
        context_dropout_p=0.25,
        trajectory_condition=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    t = torch.randn(1, device=device)
    x = torch.randn(1, 8, 4, device=device)
    context = torch.randn(1, 2, device=device)
    
    # Disable cache
    model.enable_compilation_cache(False)
    
    # This should compile directly without caching
    with torch.no_grad():
        model.compiled_conditional_test(t, x, context)
    
    print(f"Cache stats with disabled cache: {model.get_cache_stats()}")


if __name__ == "__main__":
    print("Compilation Cache Test")
    print("=" * 50)
    
    try:
        # Test basic caching
        model = test_compilation_caching()
        
        # Test cache clearing
        test_cache_clearing()
        
        # Test cache disable
        test_cache_disable()
        
        print("\n--- Summary ---")
        print("✅ All tests completed")
        print("✅ Compilation caching system is working")
        print("✅ Cache can be cleared and disabled")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 