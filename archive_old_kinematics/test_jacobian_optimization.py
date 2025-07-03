#!/usr/bin/env python3
"""
Test script for optimized Jacobian implementation.

This script tests the accuracy and performance of the optimized Jacobian
calculation for the Allegro hand compared to standard pytorch_kinematics.
"""

import torch
import time
import numpy as np
import pytorch_kinematics as pk
from allegro_optimized_kinematics import OptimizedAllegroChain
import matplotlib.pyplot as plt


def test_jacobian_accuracy():
    """Test accuracy of optimized Jacobian against standard implementation."""
    print("🧪 Testing Jacobian accuracy...")
    
    # Load Allegro hand
    with open('allegro_hand_right.urdf', 'r') as f:
        urdf_data = f.read()
    
    # Create both implementations
    opt_chain = OptimizedAllegroChain(urdf_data=urdf_data)
    std_chain = pk.build_chain_from_urdf(urdf_data, use_optimized_allegro=False)
    
    # Test configurations
    n_tests = 20
    max_error = 0.0
    errors = []
    
    for i in range(n_tests):
        # Random joint configuration
        q = torch.randn(1, 16) * 0.5
        
        # Test each finger end-effector
        for finger_name, config in opt_chain.analysis['kinematic_chains'].items():
            ee_link = config['ee_link']
            
            # Standard Jacobian
            frame_idx = torch.tensor([std_chain.frame_to_idx[ee_link]])
            std_jac = std_chain.calc_jacobian(q, link_indices=frame_idx)[0]  # [6, 16]
            
            # Optimized Jacobian
            opt_results = opt_chain.jacobian(q, [ee_link])
            opt_jac = opt_results[ee_link][0]  # [6, 16]
            
            # Compare
            error = torch.abs(opt_jac - std_jac).max().item()
            errors.append(error)
            max_error = max(max_error, error)
            
            if error > 1e-5:
                print(f"❌ Test {i}, {finger_name}: error = {error:.2e}")
                return False
            else:
                print(f"✅ Test {i}, {finger_name}: error = {error:.2e}")
    
    print(f"✅ All accuracy tests passed! Maximum error: {max_error:.2e}")
    return True


def benchmark_jacobian_performance():
    """Benchmark Jacobian computation performance."""
    print("\n📊 Benchmarking Jacobian performance...")
    
    # Load Allegro hand
    with open('allegro_hand_right.urdf', 'r') as f:
        urdf_data = f.read()
    
    opt_chain = OptimizedAllegroChain(urdf_data=urdf_data)
    std_chain = pk.build_chain_from_urdf(urdf_data, use_optimized_allegro=False)
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 25, 50, 100, 200]
    end_effectors = [config['ee_link'] for config in opt_chain.analysis['kinematic_chains'].values()]
    
    std_times = []
    opt_times = []
    speedups = []
    
    n_trials = 30
    
    for batch_size in batch_sizes:
        print(f"\n Testing batch size {batch_size}...")
        
        # Generate test data
        q = torch.randn(batch_size, 16) * 0.5
        
        # Benchmark standard Jacobian
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(n_trials):
            for ee_link in end_effectors:
                frame_idx = torch.tensor([std_chain.frame_to_idx[ee_link]])
                std_jac = std_chain.calc_jacobian(q, link_indices=frame_idx)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        std_time = (time.time() - start_time) / n_trials
        
        # Benchmark optimized Jacobian
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(n_trials):
            opt_results = opt_chain.jacobian(q, end_effectors)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        opt_time = (time.time() - start_time) / n_trials
        
        speedup = std_time / opt_time
        std_times.append(std_time)
        opt_times.append(opt_time)
        speedups.append(speedup)
        
        print(f"Benchmark Results (batch_size={batch_size}, trials={n_trials}):")
        print(f"  Standard Jacobian: {std_time:.4f}s")
        print(f"  Optimized Jacobian: {opt_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(batch_sizes, std_times, 'o-', label='Standard Jacobian', linewidth=2)
    plt.plot(batch_sizes, opt_times, 's-', label='Optimized Jacobian', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Jacobian Computation Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(batch_sizes, speedups, 'o-', color='green', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor')
    plt.title('Jacobian Speedup vs Batch Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.loglog(batch_sizes, std_times, 'o-', label='Standard Jacobian')
    plt.loglog(batch_sizes, opt_times, 's-', label='Optimized Jacobian')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Jacobian Performance (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jacobian_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved benchmark plot to jacobian_benchmark.png")
    
    # Summary
    avg_speedup = np.mean(speedups)
    max_speedup = max(speedups)
    
    print(f"\n🏆 JACOBIAN OPTIMIZATION SUMMARY:")
    print(f"   Average speedup: {avg_speedup:.2f}x")
    print(f"   Best speedup: {max_speedup:.2f}x")
    
    return avg_speedup > 1.0


def test_jacobian_interface_compatibility():
    """Test interface compatibility with standard pytorch_kinematics."""
    print("\n🔌 Testing Jacobian interface compatibility...")
    
    # Load Allegro hand
    with open('allegro_hand_right.urdf', 'r') as f:
        urdf_data = f.read()
    
    opt_chain = OptimizedAllegroChain(urdf_data=urdf_data)
    
    # Test various call patterns
    q = torch.randn(3, 16) * 0.3
    
    # Test 1: No arguments (should compute all end-effectors)
    try:
        result1 = opt_chain.jacobian(q)
        assert len(result1) == 4  # 4 fingers
        print("✅ Default jacobian() call works")
    except Exception as e:
        print(f"❌ Default jacobian() call failed: {e}")
        return False
    
    # Test 2: Specific frame names
    try:
        ee_link = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
        result2 = opt_chain.jacobian(q, [ee_link])
        assert ee_link in result2
        assert result2[ee_link].shape == (3, 6, 16)
        print("✅ Specific frame jacobian() call works")
    except Exception as e:
        print(f"❌ Specific frame jacobian() call failed: {e}")
        return False
    
    # Test 3: With tool locations
    try:
        locations = torch.tensor([[0.01, 0.0, 0.02]] * 3)  # Tool offset for each batch
        result3 = opt_chain.jacobian(q, locations=locations)
        print("✅ Jacobian with tool locations works")
    except Exception as e:
        print(f"❌ Jacobian with tool locations failed: {e}")
        return False
    
    print("✅ All interface compatibility tests passed!")
    return True


def main():
    """Run all Jacobian optimization tests."""
    print("🚀 Starting Jacobian optimization test suite\n")
    
    # Test accuracy
    accuracy_passed = test_jacobian_accuracy()
    
    if not accuracy_passed:
        print("❌ Accuracy tests failed - stopping here!")
        return
    
    # Test interface compatibility
    interface_passed = test_jacobian_interface_compatibility()
    
    if not interface_passed:
        print("❌ Interface compatibility tests failed!")
        return
    
    # Benchmark performance
    performance_improved = benchmark_jacobian_performance()
    
    # Final assessment
    print(f"\n🏁 FINAL ASSESSMENT:")
    print(f"   ✅ Accuracy: {'Passed' if accuracy_passed else 'Failed'}")
    print(f"   ✅ Interface: {'Compatible' if interface_passed else 'Incompatible'}")
    print(f"   {'✅' if performance_improved else '⚠️'} Performance: {'Improved' if performance_improved else 'Needs work'}")
    
    if accuracy_passed and interface_passed and performance_improved:
        print("🎉 All tests passed! Jacobian optimization is working correctly.")
    else:
        print("⚠️  Some tests failed or need improvement.")


if __name__ == "__main__":
    main() 