#!/usr/bin/env python3
"""
Test script for optimized Allegro jacobian implementation.
Tests correctness and performance compared to pytorch_kinematics.
"""

import torch
import numpy as np
import time
import pytorch_kinematics as pk
from allegro_optimized_kinematics import OptimizedAllegroChain
import matplotlib.pyplot as plt

def load_allegro_urdf():
    """Load Allegro URDF file."""
    # Use the provided allegro_hand_right.urdf file
    import os
    urdf_path = "allegro_hand_right.urdf"
    
    if os.path.exists(urdf_path):
        with open(urdf_path, 'r') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Could not find Allegro URDF at {urdf_path}")

def test_jacobian_correctness():
    """Test that optimized jacobian matches pytorch_kinematics."""
    print("Testing Jacobian Correctness...")
    
    try:
        urdf_data = load_allegro_urdf()
        
        # Create both chains
        standard_chain = pk.build_chain_from_urdf(urdf_data)
        optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        
        standard_chain = standard_chain.to(device=device, dtype=dtype)
        optimized_chain = optimized_chain.to(device=device, dtype=dtype)
        
        print(f"Using device: {device}")
        print(f"Number of joints: {standard_chain.n_joints}")
        
        # Test cases
        test_cases = [
            {"batch_size": 1, "description": "Single batch element"},
            {"batch_size": 4, "description": "Small batch"},
            {"batch_size": 16, "description": "Medium batch"},
        ]
        
        for test_case in test_cases:
            batch_size = test_case["batch_size"]
            print(f"\n  Testing {test_case['description']} (batch_size={batch_size})")
            
            # Generate random joint angles
            th = torch.randn(batch_size, standard_chain.n_joints, device=device, dtype=dtype)
            
            # Test different link indices scenarios
            available_frames = list(standard_chain.frame_to_idx.keys())
            print(f"  Available frames: {len(available_frames)}")
            
            if len(available_frames) == 0:
                print("  No frames available for testing")
                continue
                
            # Test single link index
            link_idx = standard_chain.frame_to_idx[available_frames[0]]
            
            try:
                # Standard jacobian
                jac_standard = standard_chain.calc_jacobian(th, link_indices=link_idx)
                
                # Optimized jacobian
                jac_optimized = optimized_chain.jacobian(th, link_indices=link_idx)
                
                # Compare
                if jac_standard.shape == jac_optimized.shape:
                    diff = torch.abs(jac_standard - jac_optimized).max().item()
                    print(f"    Single link: Max difference = {diff:.2e}")
                    if diff < 1e-5:
                        print(f"    ✓ PASS: Single link jacobian matches")
                    else:
                        print(f"    ✗ FAIL: Single link jacobian differs by {diff}")
                else:
                    print(f"    ✗ FAIL: Shape mismatch - standard: {jac_standard.shape}, optimized: {jac_optimized.shape}")
                    
            except Exception as e:
                print(f"    ✗ ERROR in single link test: {e}")
            
            # Test multiple different link indices
            if len(available_frames) > 1 and batch_size > 1:
                try:
                    link_indices = [standard_chain.frame_to_idx[available_frames[i % len(available_frames)]] 
                                  for i in range(batch_size)]
                    
                    # For standard chain, we need to compute jacobians individually
                    jac_standard_list = []
                    for i in range(batch_size):
                        jac_i = standard_chain.calc_jacobian(th[i:i+1], link_indices=link_indices[i])
                        jac_standard_list.append(jac_i)
                    jac_standard_multi = torch.cat(jac_standard_list, dim=0)
                    
                    # Optimized jacobian - should handle multiple efficiently 
                    jac_optimized_multi = optimized_chain.jacobian(th, link_indices=link_indices)
                    
                    # Compare
                    if jac_standard_multi.shape == jac_optimized_multi.shape:
                        diff = torch.abs(jac_standard_multi - jac_optimized_multi).max().item()
                        print(f"    Multiple links: Max difference = {diff:.2e}")
                        if diff < 1e-5:
                            print(f"    ✓ PASS: Multiple link jacobian matches")
                        else:
                            print(f"    ✗ FAIL: Multiple link jacobian differs by {diff}")
                    else:
                        print(f"    ✗ FAIL: Multi-link shape mismatch - standard: {jac_standard_multi.shape}, optimized: {jac_optimized_multi.shape}")
                        
                except Exception as e:
                    print(f"    ✗ ERROR in multiple link test: {e}")
    
    except Exception as e:
        print(f"✗ ERROR: Could not load URDF or create chains: {e}")

def benchmark_performance():
    """Benchmark optimized vs standard jacobian performance."""
    print("\nBenchmarking Performance...")
    
    try:
        urdf_data = load_allegro_urdf()
        
        # Create both chains
        standard_chain = pk.build_chain_from_urdf(urdf_data)
        optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        
        standard_chain = standard_chain.to(device=device, dtype=dtype)
        optimized_chain = optimized_chain.to(device=device, dtype=dtype)
        
        available_frames = list(standard_chain.frame_to_idx.keys())
        if len(available_frames) == 0:
            print("No frames available for benchmarking")
            return
            
        link_idx = standard_chain.frame_to_idx[available_frames[0]]
        
        # Benchmark different batch sizes
        batch_sizes = [1, 4, 16, 64, 256, 1024]
        results = {"batch_sizes": [], "standard_times": [], "optimized_times": [], "speedups": []}
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            # Generate test data
            th = torch.randn(batch_size, standard_chain.n_joints, device=device, dtype=dtype)
            
            # Warm up
            for _ in range(3):
                try:
                    _ = standard_chain.calc_jacobian(th, link_indices=link_idx)
                    _ = optimized_chain.jacobian(th, link_indices=link_idx)
                except:
                    pass
            
            # Benchmark standard
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            n_trials = 10
            
            for _ in range(n_trials):
                try:
                    jac_standard = standard_chain.calc_jacobian(th, link_indices=link_idx)
                except Exception as e:
                    print(f"    Standard chain failed: {e}")
                    continue
                    
            torch.cuda.synchronize() if device == "cuda" else None
            standard_time = (time.time() - start_time) / n_trials
            
            # Benchmark optimized
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            for _ in range(n_trials):
                try:
                    jac_optimized = optimized_chain.jacobian(th, link_indices=link_idx)
                except Exception as e:
                    print(f"    Optimized chain failed: {e}")
                    continue
                    
            torch.cuda.synchronize() if device == "cuda" else None
            optimized_time = (time.time() - start_time) / n_trials
            
            speedup = standard_time / optimized_time if optimized_time > 0 else 0
            
            print(f"    Standard: {standard_time*1000:.2f} ms")
            print(f"    Optimized: {optimized_time*1000:.2f} ms") 
            print(f"    Speedup: {speedup:.2f}x")
            
            results["batch_sizes"].append(batch_size)
            results["standard_times"].append(standard_time * 1000)  # Convert to ms
            results["optimized_times"].append(optimized_time * 1000)
            results["speedups"].append(speedup)
        
        # Plot results
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time comparison
            ax1.loglog(results["batch_sizes"], results["standard_times"], 'o-', label='Standard', linewidth=2)
            ax1.loglog(results["batch_sizes"], results["optimized_times"], 's-', label='Optimized', linewidth=2)
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Time (ms)')
            ax1.set_title('Jacobian Computation Time')
            ax1.legend()
            ax1.grid(True)
            
            # Speedup
            ax2.semilogx(results["batch_sizes"], results["speedups"], 'o-', color='green', linewidth=2)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Speedup (x)')
            ax2.set_title('Performance Speedup')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('jacobian_benchmark.png', dpi=150, bbox_inches='tight')
            print(f"\n  Benchmark plot saved as 'jacobian_benchmark.png'")
            
        except Exception as e:
            print(f"Could not create plots: {e}")
            
        return results
        
    except Exception as e:
        print(f"✗ ERROR in benchmarking: {e}")

def test_with_locations():
    """Test jacobian computation with tool locations."""
    print("\nTesting with Tool Locations...")
    
    try:
        urdf_data = load_allegro_urdf()
        
        standard_chain = pk.build_chain_from_urdf(urdf_data)
        optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        standard_chain = standard_chain.to(device=device)
        optimized_chain = optimized_chain.to(device=device)
        
        batch_size = 4
        th = torch.randn(batch_size, standard_chain.n_joints, device=device)
        locations = torch.randn(batch_size, 3, device=device) * 0.1  # Small tool offsets
        
        available_frames = list(standard_chain.frame_to_idx.keys())
        if len(available_frames) == 0:
            print("No frames available for testing")
            return
            
        link_idx = standard_chain.frame_to_idx[available_frames[0]]
        
        # Test both with locations in EE frame and world frame
        for locations_in_ee_frame in [True, False]:
            print(f"  Testing locations_in_ee_frame = {locations_in_ee_frame}")
            
            try:
                # Standard (need to convert locations to tool transform)
                # Standard chain expects a single tensor of tool transforms
                tool_batch = torch.zeros(batch_size, 4, 4, device=device, dtype=th.dtype)
                for i in range(batch_size):
                    tool_tf = pk.Transform3d(pos=locations[i], device=device, dtype=th.dtype)
                    tool_batch[i] = tool_tf.get_matrix()
                
                jac_standard = standard_chain.calc_jacobian(
                    th, tool=tool_batch, link_indices=link_idx,
                    tool_in_ee_frame=locations_in_ee_frame
                )
                
                # Optimized
                jac_optimized = optimized_chain.jacobian(
                    th, locations=locations, link_indices=link_idx,
                    locations_in_ee_frame=locations_in_ee_frame
                )
                
                # Compare
                diff = torch.abs(jac_standard - jac_optimized).max().item()
                print(f"    Max difference = {diff:.2e}")
                
                if diff < 1e-4:  # Slightly looser tolerance for tool offset cases
                    print(f"    ✓ PASS: Jacobian with locations matches")
                else:
                    print(f"    ✗ FAIL: Jacobian with locations differs by {diff}")
                    
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
    
    except Exception as e:
        print(f"✗ ERROR in locations test: {e}")

def benchmark_finger_specific():
    """Benchmark finger-specific optimizations."""
    print("\nBenchmarking Finger-Specific Performance...")
    
    try:
        urdf_data = load_allegro_urdf()
        
        standard_chain = pk.build_chain_from_urdf(urdf_data)
        optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        standard_chain = standard_chain.to(device=device)
        optimized_chain = optimized_chain.to(device=device)
        
        # Find the frames that the analysis expects for optimized computation
        finger_frames = []
        if hasattr(optimized_chain, 'analysis') and 'kinematic_chains' in optimized_chain.analysis:
            for finger_name, data in optimized_chain.analysis['kinematic_chains'].items():
                ee_link = data.get('ee_link')
                if ee_link and ee_link in standard_chain.frame_to_idx:
                    finger_frames.append(ee_link)
        
        print(f"  Found optimized finger frames: {len(finger_frames)}")
        
        if len(finger_frames) == 0:
            print("  No optimized finger frames found")
            return
            
        # Test with multiple finger end-effectors
        batch_sizes = [16, 64, 256, 1024]
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            # Create diverse link indices (multiple different fingers)
            link_indices = []
            for i in range(batch_size):
                frame_name = finger_frames[i % len(finger_frames)]
                link_indices.append(standard_chain.frame_to_idx[frame_name])
            
            print(f"    Using link indices pattern: {set(link_indices)}")
            
            # Generate test data
            th = torch.randn(batch_size, standard_chain.n_joints, device=device)
            
            # Warm up
            for _ in range(3):
                try:
                    # Standard - try batched first
                    try:
                        link_indices_tensor = torch.tensor(link_indices, device=device)
                        _ = standard_chain.calc_jacobian(th, link_indices=link_indices_tensor)
                    except:
                        # Fall back to individual if needed
                        for i in range(min(4, batch_size)):  # Just warm up with a few
                            _ = standard_chain.calc_jacobian(th[i:i+1], link_indices=link_indices[i])
                    _ = optimized_chain.jacobian(th, link_indices=link_indices)
                except:
                    pass
            
            # Benchmark standard (batched computation with different link indices)
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            n_trials = 5
            
            standard_uses_batched = False
            for trial in range(n_trials):
                # Try batched computation first
                try:
                    link_indices_tensor = torch.tensor(link_indices, device=device)
                    jac_standard = standard_chain.calc_jacobian(th, link_indices=link_indices_tensor)
                    if trial == 0:
                        standard_uses_batched = True
                except:
                    # Fall back to individual computation if batched doesn't work
                    jac_list = []
                    for i in range(batch_size):
                        jac_i = standard_chain.calc_jacobian(th[i:i+1], link_indices=link_indices[i])
                        jac_list.append(jac_i)
                    jac_standard = torch.cat(jac_list, dim=0)
                    
            torch.cuda.synchronize() if device == "cuda" else None
            standard_time = (time.time() - start_time) / n_trials
            
            # Benchmark optimized (single call with batched inputs)
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            for _ in range(n_trials):
                jac_optimized = optimized_chain.jacobian(th, link_indices=link_indices)
                    
            torch.cuda.synchronize() if device == "cuda" else None
            optimized_time = (time.time() - start_time) / n_trials
            
            speedup = standard_time / optimized_time if optimized_time > 0 else 0
            
            method = "batched" if standard_uses_batched else "individual"
            print(f"    Standard ({method}): {standard_time*1000:.2f} ms")
            print(f"    Optimized (batched): {optimized_time*1000:.2f} ms") 
            print(f"    Speedup: {speedup:.2f}x")
            
            # Verify correctness
            diff = torch.abs(jac_standard - jac_optimized).max().item()
            if diff < 1e-5:
                print(f"    ✓ Results match (diff = {diff:.2e})")
            else:
                print(f"    ✗ Results differ (diff = {diff:.2e})")
        
    except Exception as e:
        print(f"✗ ERROR in finger-specific benchmarking: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED ALLEGRO JACOBIAN TESTING")
    print("=" * 60)
    
    # Run correctness tests
    test_jacobian_correctness()
    
    # Run performance benchmarks
    benchmark_results = benchmark_performance()
    
    # Test with tool locations
    test_with_locations()
    
    # Benchmark finger-specific optimizations
    benchmark_finger_specific()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    
    if benchmark_results:
        print(f"\nSummary:")
        max_speedup = max(benchmark_results["speedups"]) if benchmark_results["speedups"] else 0
        avg_speedup = np.mean(benchmark_results["speedups"]) if benchmark_results["speedups"] else 0
        print(f"  Max speedup: {max_speedup:.2f}x")
        print(f"  Average speedup: {avg_speedup:.2f}x") 