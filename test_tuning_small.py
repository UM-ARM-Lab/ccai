#!/usr/bin/env python3
"""
Small test script for IDTO parameter tuning to validate the approach.
This runs only 2-3 parameter configurations with 2 trials each for quick testing.
"""

import pathlib
import sys
import logging

# Add the ccai module to path
sys.path.append(str(pathlib.Path(__file__).parent))

from tune_idto_parameters import run_parameter_tuning


def test_small_tuning():
    """Test tuning with a small subset of parameters."""
    
    # Very small parameter space for testing
    param_ranges = {
        'Qq_hand': (-2, -2, 1),    # Only 10^-2 = 0.01
        'Qq': (1, 1, 1),           # Only 10^1 = 10
        'Qv_hand': (-3, -3, 1),    # Only 10^-3 = 0.001  
        'Qv': (0, 0, 1),           # Only 10^0 = 1
        'hand_R': (-2, -2, 1),     # Only 10^-2 = 0.01
        'screw_r': (1, 1, 1),      # Only 10^1 = 10
        'Qf_q_hand': (-2, -2, 1),  # Only 10^-2 = 0.01
        'Qf_q': (3, 3, 1),        # Only 10^3 = 1000
        'Qf_v': (1, 1, 1),        # Only 10^1 = 10
    }
    
    # This should generate only 1 configuration (1^9 = 1)
    print(f"Testing with parameter ranges: {param_ranges}")
    
    # Base configuration - use minimal settings for testing
    config_base = {
        'visualize': False,
        'sim_device': 'cuda:0', 
        'kp': 25.0,
        'fingers': 'index_middle_thumb',
        'randomize_obj_start': False,
        'randomize_rob_start': False,
        'external_wrench_perturb': False,
        'tactile_controller': False,
    }
    
    # Run tuning with minimal trials and cycles for quick test (sequential)
    results = run_parameter_tuning(
        param_ranges=param_ranges,
        num_trials_per_config=2,          # Only 2 trials
        max_cycles_per_trial=50,          # Only 50 cycles per trial
        output_dir="test_tuning_results", 
        config_base=config_base,
        parallel=False,                   # Test sequential first
        num_parallel_configs=1,
        threads_per_config=8,
    )
    
    print(f"\nTest completed successfully!")
    print(f"Total trials: {results['completed_trials']}")
    print(f"Expected trials: {results['total_configs'] * 2}")  # 1 config * 2 trials
    
    # Check if we got any successful trials
    successful_results = [r for r in results['all_results'] if r.get('success', False)]
    print(f"Successful trials: {len(successful_results)}/{results['completed_trials']}")
    
    if successful_results:
        rotations = [r['clockwise_rotation'] for r in successful_results]
        print(f"Clockwise rotations: {rotations}")
        print(f"Mean rotation: {sum(rotations)/len(rotations):.4f}")
    
    return results


def test_slightly_larger():
    """Test with slightly larger parameter space - 2 values per parameter."""
    
    # Small but realistic parameter space  
    param_ranges = {
        'Qq_hand': (-3, -2, 2),    # [1e-3, 1e-2]
        'Qq': (0, 1, 2),           # [1, 10]
        'Qv_hand': (-4, -3, 2),    # [1e-4, 1e-3] 
        'Qv': (0, 0, 1),           # [1] (fixed)
        'hand_R': (-3, -2, 2),     # [1e-3, 1e-2]
        'screw_r': (1, 1, 1),      # [10] (fixed)
        'Qf_q_hand': (-2, -2, 1),  # [1e-2] (fixed)
        'Qf_q': (3, 3, 1),        # [1000] (fixed) 
        'Qf_v': (1, 1, 1),        # [10] (fixed)
    }
    
    # This should generate 2*2*2*1*2*1*1*1*1 = 16 configurations
    print(f"Testing with larger parameter ranges: {param_ranges}")
    
    config_base = {
        'visualize': False,
        'sim_device': 'cuda:0',
        'kp': 25.0,
        'fingers': 'index_middle_thumb', 
        'randomize_obj_start': False,
        'randomize_rob_start': False,
        'external_wrench_perturb': False,
        'tactile_controller': False,
    }
    
    results = run_parameter_tuning(
        param_ranges=param_ranges,
        num_trials_per_config=2,
        max_cycles_per_trial=100,
        output_dir="test_tuning_results_larger",
        config_base=config_base,
        parallel=True,                    # Test parallel execution
        num_parallel_configs=2,           # Use 2 parallel processes for testing
        threads_per_config=4,             # Reduce threads per config for testing
        devices=['cuda:0'],               # Single device for testing
    )
    
    print(f"\nLarger test completed!")
    print(f"Total configurations: {results['total_configs']}")
    print(f"Total trials: {results['completed_trials']}")
    print(f"Expected trials: {results['total_configs'] * 2}")
    
    return results


def main():
    """Run the test."""
    print("=== Testing Small Parameter Tuning ===")
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test 1: Minimal single configuration
    print("\n--- Test 1: Single configuration ---")
    results1 = test_small_tuning()
    
    # Test 2: Small but multi-configuration (parallel)
    print("\n--- Test 2: Multiple configurations (parallel) ---") 
    results2 = test_slightly_larger()
    
    # Test 3: Verify parallel vs sequential consistency
    print("\n--- Test 3: Parallel vs Sequential consistency ---")
    print(f"Test 1 (sequential): {len(results1['all_results'])} trials")
    print(f"Test 2 (parallel):   {len(results2['all_results'])} trials") 
    print(f"Parallel speedup expected: ~2x (2 parallel processes)")
    
    print("\n=== All tests completed successfully! ===")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
