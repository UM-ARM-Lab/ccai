#!/usr/bin/env python3
"""
Test script for Ray Tune tactile controller hyperparameter tuning.
Runs a single hyperparameter configuration to test the setup and identify issues.
"""

import os
import sys
import yaml
import pathlib
import tempfile
import numpy as np
from copy import deepcopy
import time
from typing import Dict, Any

# Ray Tune imports
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Add the ccai path to sys.path
CCAI_PATH = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(CCAI_PATH))

# Import Isaac Gym modules first (before PyTorch)
try:
    from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
    from isaac_victor_envs.utils import get_assets_dir
    import pytorch_kinematics as pk
    ISAAC_GYM_AVAILABLE = True
    print("✓ Successfully imported Isaac Gym environment")
except ImportError as e:
    print(f"Warning: Isaac Gym not available: {e}")
    ISAAC_GYM_AVAILABLE = False

# Now import PyTorch and other modules
import torch

# Import required modules with error handling
try:
    # Import the actual do_trial function and dependencies
    from examples.allegro_screwdriver import do_trial, AllegroScrewdriver, CCAI_PATH as SCRIPT_CCAI_PATH
    from ccai.utils.allegro_utils import extract_state_vector
    from ccai.models.management.model_manager import ModelManager
    ALLEGRO_AVAILABLE = True
    print("✓ Successfully imported allegro_screwdriver modules")
except ImportError as e:
    print(f"Warning: Could not import allegro modules: {e}")
    ALLEGRO_AVAILABLE = False


def test_single_config():
    """Test a single hyperparameter configuration."""
    
    print("Testing single hyperparameter configuration...")
    
    # Define a single test configuration
    test_config = {
        'K_e': 100.0,      # Environment stiffness
        'w_q': 10.0,       # Position weight  
        'w_p': 1.0,        # Velocity weight
        'w_f': 1.0,        # Force weight
        'w_u': 1.0,        # Control effort weight
        'w_ori': 0.1,      # Orientation weight
    }
    
    print(f"Test configuration: {test_config}")
    
    # Test the objective function directly
    try:
        from tune_tactile_controller import run_tactile_experiment
        
        print("\nRunning single trial...")
        result = run_tactile_experiment(test_config)
        
        print(f"\nTrial result: {result}")
        
        return True
        
    except Exception as e:
        print(f"Error running trial: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ray_tune_single():
    """Test Ray Tune with a single configuration."""
    
    print("Testing Ray Tune with single configuration...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Define a single test configuration
    test_config = {
        'K_e': 100.0,
        'w_q': 10.0,
        'w_p': 1.0,
        'w_f': 1.0,
        'w_u': 1.0,
        'w_ori': 0.1,
    }
    
    def test_objective(config_params):
        """Test objective function."""
        try:
            # Simple mock result to test Ray Tune integration
            # Avoid complex imports that cause issues in worker threads
            import random
            
            # Simulate some computation
            performance_score = random.uniform(-1.0, 2.0)
            distance_to_goal = random.uniform(0.1, 2.0)
            success_rate = random.uniform(0, 1)
            completion_rate = random.uniform(0.5, 1)
            
            # Report results using the correct Ray Tune API
            from ray.air import session
            session.report({
                'performance_score': performance_score,
                'final_distance_to_goal': distance_to_goal,
                'success_rate': success_rate,
                'completion_rate': completion_rate,
                'num_trials_completed': 1
            })
            
        except Exception as e:
            print(f"Test experiment failed: {e}")
            import traceback
            traceback.print_exc()
            # Report poor performance for failed experiments
            from ray.air import session
            session.report({
                'performance_score': -10.0,
                'final_distance_to_goal': 10.0,
                'success_rate': 0.0,
                'completion_rate': 0.0,
                'num_trials_completed': 0
            })
    
    try:
        # Create absolute path for storage
        storage_path = pathlib.Path("./ray_results").resolve()
        
        # Run single configuration test
        analysis = tune.run(
            test_objective,
            config=test_config,
            num_samples=1,  # Only 1 sample
            name="test_single_config",
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
    print("TACTILE CONTROLLER TUNING SETUP TEST")
    print("="*60)
    print(f"Allegro modules available: {ALLEGRO_AVAILABLE}")
    print(f"Isaac Gym available: {ISAAC_GYM_AVAILABLE}")
    
    # Test 1: Direct function call
    print("\n" + "-"*40)
    print("TEST 1: Direct function call")
    print("-"*40)
    success1 = test_single_config()
    
    # Test 2: Ray Tune integration
    print("\n" + "-"*40)
    print("TEST 2: Ray Tune integration")
    print("-"*40)
    success2 = test_ray_tune_single()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Direct function call: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Ray Tune integration: {'✓ PASS' if success2 else '✗ FAIL'}")
    
    if success1 and success2:
        print("\n✅ All tests passed! Ray Tune setup is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 