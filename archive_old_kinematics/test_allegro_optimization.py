"""
Test and benchmark optimized Allegro kinematics

This script validates that the optimized forward kinematics produces
identical results to standard pytorch_kinematics while providing
significant performance improvements.
"""

import torch
import numpy as np
import time
import pytorch_kinematics as pk
from allegro_optimized_kinematics import OptimizedAllegroChain
import allegro_optimized_wrapper as aow
from allegro_optimized_wrapper import benchmark_allegro_fk
import matplotlib.pyplot as plt
from typing import Dict, List


def load_allegro_urdf() -> str:
    """Load Allegro URDF data."""
    with open("allegro_hand_right.urdf", 'r') as f:
        return f.read()


def test_accuracy(urdf_data: str, num_tests: int = 100, tolerance: float = 1e-6) -> bool:
    """
    Test that optimized FK produces identical results to standard FK.
    
    Args:
        urdf_data: Allegro URDF string
        num_tests: Number of random configurations to test
        tolerance: Numerical tolerance for equality
        
    Returns:
        True if all tests pass, False otherwise
    """
    print(f"🧪 Testing accuracy with {num_tests} random configurations...")
    
    # Create both chain types
    optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
    standard_chain = pk.build_chain_from_urdf(urdf_data, use_optimized_allegro=False)
    
    # End-effector links to test
    ee_links = [
        'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',  # index
        'allegro_hand_naka_finger_finger_1_aftc_base_link',      # middle
        'allegro_hand_kusuri_finger_finger_2_aftc_base_link',    # ring
        'allegro_hand_oya_finger_3_aftc_base_link'               # thumb
    ]
    
    ee_indices = torch.tensor([standard_chain.frame_to_idx[link] for link in ee_links])
    
    all_passed = True
    max_error = 0.0
    
    for i in range(num_tests):
        # Generate random joint configuration
        q = torch.randn(1, 16) * 0.8  # Single configuration
        
        # Compute FK with both methods
        optimized_result = optimized_chain.forward_kinematics(q, ee_links)
        standard_result = standard_chain.forward_kinematics(q, ee_indices)
        
        # Compare results for each end-effector
        for ee_link in ee_links:
            opt_matrix = optimized_result[ee_link].get_matrix()[0]  # Remove batch dim
            std_matrix = standard_result[ee_link].get_matrix()[0]
            
            # Compute maximum element-wise error
            error = torch.abs(opt_matrix - std_matrix).max().item()
            max_error = max(max_error, error)
            
            if error > tolerance:
                print(f"❌ Test {i} failed for {ee_link}: error = {error:.2e}")
                all_passed = False
                break
        
        if not all_passed:
            break
    
    if all_passed:
        print(f"✅ All accuracy tests passed! Maximum error: {max_error:.2e}")
    else:
        print(f"❌ Accuracy test failed! Maximum error: {max_error:.2e}")
    
    return all_passed


def benchmark_batch_sizes(urdf_data: str, batch_sizes: List[int], num_trials: int = 50) -> Dict:
    """
    Benchmark performance across different batch sizes.
    
    Args:
        urdf_data: Allegro URDF string
        batch_sizes: List of batch sizes to test
        num_trials: Number of trials per batch size
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"📊 Benchmarking across batch sizes: {batch_sizes}")
    
    results = {
        'batch_sizes': batch_sizes,
        'standard_times': [],
        'optimized_times': [],
        'speedups': []
    }
    
    for batch_size in batch_sizes:
        print(f"\n Testing batch size {batch_size}...")
        benchmark_result = benchmark_allegro_fk(urdf_data, batch_size, num_trials)
        
        results['standard_times'].append(benchmark_result.standard_time)
        results['optimized_times'].append(benchmark_result.optimized_time)
        results['speedups'].append(benchmark_result.speedup)
    
    return results


def plot_benchmark_results(results: Dict, save_path: str = "allegro_benchmark.png"):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    batch_sizes = results['batch_sizes']
    standard_times = results['standard_times']
    optimized_times = results['optimized_times']
    speedups = results['speedups']
    
    # Plot timing comparison
    ax1.plot(batch_sizes, standard_times, 'o-', label='Standard FK', linewidth=2, markersize=6)
    ax1.plot(batch_sizes, optimized_times, 's-', label='Optimized FK', linewidth=2, markersize=6)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Forward Kinematics Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot speedup
    ax2.plot(batch_sizes, speedups, 'o-', color='green', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('Optimization Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved benchmark plot to {save_path}")
    
    return fig


def test_interface_compatibility(urdf_data: str):
    """Test that the optimized wrapper maintains interface compatibility."""
    print("🔌 Testing interface compatibility...")
    
    # Test standard usage patterns from the codebase
    chain = aow.build_chain_from_urdf(urdf_data)
    
    # Test basic attributes
    assert hasattr(chain, 'n_joints'), "Missing n_joints attribute"
    assert hasattr(chain, 'frame_to_idx'), "Missing frame_to_idx attribute"
    assert hasattr(chain, 'get_frame_indices'), "Missing get_frame_indices method"
    
    # Test forward kinematics with frame indices
    q = torch.randn(5, 16) * 0.5
    
    # Test with frame names (optimized path)
    ee_links = ['allegro_hand_hitosashi_finger_finger_0_aftc_base_link']
    result1 = chain.forward_kinematics(q, ee_links)
    assert isinstance(result1, dict), "FK should return dict"
    assert ee_links[0] in result1, "FK result should contain requested frame"
    
    # Test with frame indices (standard path)
    frame_idx = chain.get_frame_indices(ee_links[0])
    result2 = chain.forward_kinematics(q, frame_idx)
    assert isinstance(result2, dict), "FK should return dict"
    
    # Test jacobian computation
    jacobian = chain.jacobian(q, link_indices=frame_idx)
    assert jacobian.shape == (5, 6, 16), f"Unexpected jacobian shape: {jacobian.shape}"
    
    # Test device/dtype handling
    chain.to(device='cpu', dtype=torch.float64)
    assert chain.dtype == torch.float64, "dtype not updated correctly"
    
    print("✅ Interface compatibility tests passed!")


def test_partial_joint_input():
    """Test SerialChain wrapper with partial joint inputs."""
    print("🔧 Testing partial joint input handling...")
    
    urdf_data = load_allegro_urdf()
    
    # Create serial chain for index finger
    from allegro_optimized_wrapper import build_serial_chain_from_urdf
    ee_link = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    serial_chain = build_serial_chain_from_urdf(urdf_data, ee_link)
    
    # Test with 4 joint angles (finger-specific)
    q_finger = torch.randn(3, 4) * 0.5  # 3 configurations, 4 joints
    
    # Should work with partial input
    result = serial_chain.forward_kinematics(q_finger, end_only=True)
    assert result.get_matrix().shape == (3, 4, 4), f"Unexpected result shape: {result.get_matrix().shape}"
    
    # Test jacobian with partial input
    jac = serial_chain.jacobian(q_finger)
    assert jac.shape == (3, 6, 4), f"Unexpected jacobian shape: {jac.shape}"
    
    print("✅ Partial joint input tests passed!")


def comprehensive_test_suite():
    """Run comprehensive test suite."""
    print("🚀 Starting comprehensive Allegro optimization test suite\n")
    
    # Load URDF
    urdf_data = load_allegro_urdf()
    print("✓ Loaded Allegro URDF")
    
    # Test 1: Accuracy verification
    accuracy_passed = test_accuracy(urdf_data, num_tests=50)
    
    if not accuracy_passed:
        print("❌ Accuracy tests failed - stopping here!")
        return False
    
    # Test 2: Interface compatibility
    test_interface_compatibility(urdf_data)
    
    # Test 3: Partial joint input handling
    test_partial_joint_input()
    
    # Test 4: Performance benchmarking
    batch_sizes = [1, 5, 10, 25, 50, 100, 200]
    results = benchmark_batch_sizes(urdf_data, batch_sizes, num_trials=30)
    
    # Test 5: Plot results
    plot_benchmark_results(results)
    
    # Summary
    avg_speedup = np.mean(results['speedups'])
    best_speedup = max(results['speedups'])
    
    print(f"\n🏆 OPTIMIZATION SUMMARY:")
    print(f"   Average speedup: {avg_speedup:.2f}x")
    print(f"   Best speedup: {best_speedup:.2f}x")
    print(f"   Accuracy: Passed (within 1e-6 tolerance)")
    print(f"   Interface: Fully compatible")
    
    # Performance expectations
    if avg_speedup >= 2.0:
        print("✅ Excellent optimization performance!")
    elif avg_speedup >= 1.5:
        print("✅ Good optimization performance!")
    elif avg_speedup >= 1.2:
        print("⚠️  Modest optimization performance")
    else:
        print("❌ Poor optimization performance")
    
    return True


def simple_demo():
    """Simple demonstration of optimized kinematics."""
    print("🎮 Simple Allegro Optimization Demo\n")
    
    # Load URDF and create optimized chain
    urdf_data = load_allegro_urdf()
    chain = aow.build_chain_from_urdf(urdf_data)  # Automatic optimization
    
    # Test configuration
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4,  # index finger
                      0.0, 0.1, 0.2, 0.3,   # middle finger  
                      -0.1, 0.0, 0.1, 0.2,  # ring finger
                      0.5, 0.3, 0.1, 0.0]], # thumb
                    dtype=torch.float32)
    
    # Compute forward kinematics for specific end-effectors
    ee_links = [
        'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',  # index
        'allegro_hand_naka_finger_finger_1_aftc_base_link',      # middle
        'allegro_hand_kusuri_finger_finger_2_aftc_base_link',    # ring
        'allegro_hand_oya_finger_3_aftc_base_link'               # thumb
    ]
    
    result = chain.forward_kinematics(q, ee_links)
    
    print("End-effector positions:")
    finger_names = ['Index', 'Middle', 'Ring', 'Thumb']
    for i, (link_name, transform) in enumerate(result.items()):
        pos = transform.get_matrix()[0, :3, 3]  # Extract position
        print(f"  {finger_names[i]}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Quick benchmark
    print(f"\nQuick performance test...")
    benchmark_result = benchmark_allegro_fk(urdf_data, batch_size=20, num_trials=20)
    print(f"🏁 Speedup: {benchmark_result.speedup:.1f}x faster than standard!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        simple_demo()
    else:
        comprehensive_test_suite() 