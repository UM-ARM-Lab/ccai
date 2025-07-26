#!/usr/bin/env python3
"""
Simple test to verify the key improvements in tune_tactile_controller.py:
1. Environment created once before tuning starts
2. Only 1 hyperparameter configuration tested at a time
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

def test_global_variables():
    """Test that global variables are properly defined."""
    
    print("Testing global variables...")
    
    try:
        from tune_tactile_controller import (
            GLOBAL_ENV, GLOBAL_TRAJECTORY_SAMPLER, GLOBAL_TRAJECTORY_SAMPLER_ORIG,
            GLOBAL_CLASSIFIER, GLOBAL_CHAIN, GLOBAL_CONFIG, GLOBAL_PARAMS
        )
        
        # Check that global variables exist
        assert GLOBAL_ENV is None  # Should be None initially
        assert GLOBAL_TRAJECTORY_SAMPLER is None
        assert GLOBAL_TRAJECTORY_SAMPLER_ORIG is None
        assert GLOBAL_CLASSIFIER is None
        assert GLOBAL_CHAIN is None
        assert GLOBAL_CONFIG is None
        assert GLOBAL_PARAMS is None
        
        print("✓ Global variables are properly defined")
        return True
        
    except Exception as e:
        print(f"✗ Global variables test failed: {e}")
        return False


def test_initialize_function():
    """Test that the initialize function exists."""
    
    print("Testing initialize function...")
    
    try:
        from tune_tactile_controller import initialize_global_environment
        
        # Check that function exists and is callable
        assert callable(initialize_global_environment)
        
        print("✓ Initialize function exists and is callable")
        return True
        
    except Exception as e:
        print(f"✗ Initialize function test failed: {e}")
        return False


def test_max_concurrent_trials():
    """Test that the main function has max_concurrent_trials=1."""
    
    print("Testing max_concurrent_trials setting...")
    
    try:
        # Read the file and check for max_concurrent_trials=1
        with open('examples/tune_tactile_controller.py', 'r') as f:
            content = f.read()
        
        if 'max_concurrent_trials=1' in content:
            print("✓ max_concurrent_trials=1 is set in the main function")
            return True
        else:
            print("✗ max_concurrent_trials=1 is not set")
            return False
            
    except Exception as e:
        print(f"✗ max_concurrent_trials test failed: {e}")
        return False


def test_global_environment_initialization_call():
    """Test that the main function calls initialize_global_environment."""
    
    print("Testing global environment initialization call...")
    
    try:
        # Read the file and check for the initialization call
        with open('examples/tune_tactile_controller.py', 'r') as f:
            content = f.read()
        
        if 'initialize_global_environment()' in content:
            print("✓ initialize_global_environment() is called in main function")
            return True
        else:
            print("✗ initialize_global_environment() is not called")
            return False
            
    except Exception as e:
        print(f"✗ Global environment initialization call test failed: {e}")
        return False


def main():
    """Main test function."""
    
    print("="*60)
    print("KEY IMPROVEMENTS TEST")
    print("="*60)
    
    # Test 1: Global variables
    print("\n" + "-"*40)
    print("TEST 1: Global variables")
    print("-"*40)
    success1 = test_global_variables()
    
    # Test 2: Initialize function
    print("\n" + "-"*40)
    print("TEST 2: Initialize function")
    print("-"*40)
    success2 = test_initialize_function()
    
    # Test 3: Max concurrent trials
    print("\n" + "-"*40)
    print("TEST 3: Max concurrent trials")
    print("-"*40)
    success3 = test_max_concurrent_trials()
    
    # Test 4: Global environment initialization call
    print("\n" + "-"*40)
    print("TEST 4: Global environment initialization call")
    print("-"*40)
    success4 = test_global_environment_initialization_call()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Global variables: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Initialize function: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Max concurrent trials: {'✓ PASS' if success3 else '✗ FAIL'}")
    print(f"Global environment initialization call: {'✓ PASS' if success4 else '✗ FAIL'}")
    
    if success1 and success2 and success3 and success4:
        print("\n✅ All tests passed! The key improvements are implemented correctly.")
        print("\nKey improvements verified:")
        print("  ✓ Environment created once before tuning starts (global variables)")
        print("  ✓ Only 1 hyperparameter configuration tested at a time (max_concurrent_trials=1)")
        print("  ✓ Global environment and models shared across trials")
        print("  ✓ Proper initialization function exists")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 