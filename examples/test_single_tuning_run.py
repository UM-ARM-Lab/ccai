#!/usr/bin/env python3
"""
Test script to verify the modified tune_tactile_controller.py works correctly.
Tests the global environment initialization and single configuration testing.
"""

import sys
import pathlib
import os

# Change to the correct directory
os.chdir(pathlib.Path(__file__).parent.parent.absolute())

# Add the ccai path to sys.path
CCAI_PATH = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(CCAI_PATH))
sys.path.append('.')

def test_global_environment_initialization():
    """Test that the global environment initialization works."""
    
    print("Testing global environment initialization...")
    
    try:
        from tune_tactile_controller import initialize_global_environment, GLOBAL_ENV
        
        # Test initialization
        success = initialize_global_environment()
        
        if success and GLOBAL_ENV is not None:
            print("✓ Global environment initialization successful")
            return True
        else:
            print("✗ Global environment initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Global environment initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_trial():
    """Test that a single trial works with the global environment."""
    
    print("Testing single trial execution...")
    
    try:
        from tune_tactile_controller import run_tactile_trial
        
        # Test configuration
        test_config = {
            'K_e': 100.0,
            'w_q': 10.0,
            'w_p': 1.0,
            'w_f': 1.0,
            'w_u': 1.0,
            'w_ori': 0.1,
        }
        
        # Run a single trial
        result = run_tactile_trial(test_config)
        
        if result and 'performance_score' in result:
            print(f"✓ Single trial successful: performance_score = {result['performance_score']:.4f}")
            return True
        else:
            print("✗ Single trial failed")
            return False
            
    except Exception as e:
        print(f"✗ Single trial error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ray_tune_integration():
    """Test that Ray Tune integration works with single configuration testing."""
    
    print("Testing Ray Tune integration...")
    
    try:
        import ray
        from ray import tune
        from tune_tactile_controller import objective
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Test configuration
        test_config = {
            'K_e': 100.0,
            'w_q': 10.0,
            'w_p': 1.0,
            'w_f': 1.0,
            'w_u': 1.0,
            'w_ori': 0.1,
        }
        
        # Run single configuration test
        storage_path = pathlib.Path("./ray_results").resolve()
        
        analysis = tune.run(
            objective,
            config=test_config,
            num_samples=1,
            name="test_single_config_integration",
            storage_path=str(storage_path),
            resources_per_trial={"cpu": 1, "gpu": 0},
            max_failures=1,
            raise_on_failed_trial=False,
            max_concurrent_trials=1  # Ensure only 1 configuration is tested at a time
        )
        
        if analysis.trials and analysis.trials[0].last_result:
            print("✓ Ray Tune integration successful")
            return True
        else:
            print("✗ Ray Tune integration failed")
            return False
            
    except Exception as e:
        print(f"✗ Ray Tune integration error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main():
    """Main test function."""
    
    print("="*60)
    print("SINGLE TUNING RUN TEST")
    print("="*60)
    
    # Test 1: Global environment initialization
    print("\n" + "-"*40)
    print("TEST 1: Global environment initialization")
    print("-"*40)
    success1 = test_global_environment_initialization()
    
    # Test 2: Single trial execution
    print("\n" + "-"*40)
    print("TEST 2: Single trial execution")
    print("-"*40)
    success2 = test_single_trial()
    
    # Test 3: Ray Tune integration
    print("\n" + "-"*40)
    print("TEST 3: Ray Tune integration")
    print("-"*40)
    success3 = test_ray_tune_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Global environment initialization: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Single trial execution: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Ray Tune integration: {'✓ PASS' if success3 else '✗ FAIL'}")
    
    if success1 and success2 and success3:
        print("\n✅ All tests passed! The modified tuning script is working correctly.")
        print("Key improvements:")
        print("  - Environment created once before tuning starts")
        print("  - Only 1 hyperparameter configuration tested at a time")
        print("  - Global environment and models shared across trials")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 