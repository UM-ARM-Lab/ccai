#!/usr/bin/env python3
"""
Simple test script for Ray Tune functionality.
Tests basic Ray Tune setup without complex imports.
"""

import os
import sys
import pathlib
import numpy as np
from typing import Dict, Any

# Ray Tune imports
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def test_ray_tune_basic():
    """Test basic Ray Tune functionality."""
    
    print("Testing basic Ray Tune functionality...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Define a simple test configuration
    test_config = {
        'x': 1.0,
        'y': 2.0,
    }
    
    def simple_objective(config_params):
        """Simple objective function."""
        try:
            # Simple computation
            x = config_params['x']
            y = config_params['y']
            score = x * x + y * y
            
            # Report results using the correct Ray Tune API
            from ray.air import session
            session.report({
                'score': score,
                'x': x,
                'y': y
            })
            
        except Exception as e:
            print(f"Simple experiment failed: {e}")
            import traceback
            traceback.print_exc()
            # Report poor performance for failed experiments
            from ray.air import session
            session.report({
                'score': -10.0,
                'x': 0.0,
                'y': 0.0
            })
    
    try:
        # Create absolute path for storage
        storage_path = pathlib.Path("./ray_results").resolve()
        
        # Run single configuration test
        analysis = tune.run(
            simple_objective,
            config=test_config,
            num_samples=1,  # Only 1 sample
            name="test_simple_config",
            storage_path=str(storage_path),  # Use absolute path as string
            resources_per_trial={"cpu": 1, "gpu": 0},
            max_failures=1,
            raise_on_failed_trial=False
        )
        
        # Get results
        if analysis.trials:
            trial = analysis.trials[0]
            if trial.last_result:
                print(f"\nTest completed successfully!")
                print(f"Result: {trial.last_result}")
                return True
            else:
                print("Test completed but no results available")
                return False
        else:
            print("No trials completed")
            return False
            
    except Exception as e:
        print(f"Ray Tune test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main():
    """Main test function."""
    
    print("="*60)
    print("SIMPLE RAY TUNE TEST")
    print("="*60)
    
    # Test basic Ray Tune functionality
    print("\n" + "-"*40)
    print("TEST: Basic Ray Tune functionality")
    print("-"*40)
    success = test_ray_tune_basic()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Basic Ray Tune: {'✓ PASS' if success else '✗ FAIL'}")
    
    if success:
        print("\n✅ Ray Tune is working correctly!")
        return 0
    else:
        print("\n❌ Ray Tune test failed. Check the output above for issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 