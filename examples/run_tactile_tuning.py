#!/usr/bin/env python3
"""
Simple wrapper script to run tactile controller hyperparameter tuning.
"""

import sys
import argparse
import pathlib

# Add the ccai path to sys.path
CCAI_PATH = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(CCAI_PATH))

def main():
    parser = argparse.ArgumentParser(description="Run tactile controller hyperparameter tuning")
    parser.add_argument("--test", action="store_true", help="Run setup test only")
    parser.add_argument("--samples", type=int, default=15, help="Number of hyperparameter samples to try")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per configuration")
    parser.add_argument("--local-mode", action="store_true", help="Run Ray in local mode for debugging", default=True)
    
    args = parser.parse_args()
    
    if args.test:
        print("Running setup test...")
        try:
            from test_tactile_tuning_setup import main as test_main
            return test_main()
        except ImportError:
            print("Test script not found. Running basic import test...")
            try:
                from tune_tactile_controller import main as tune_main
                print("✓ Tuning script imports successfully")
                return 0
            except Exception as e:
                print(f"✗ Import failed: {e}")
                return 1
    else:
        print("Starting tactile controller hyperparameter tuning...")
        print(f"Configuration: {args.samples} samples, {args.trials} trials per config")
        
        try:
            # Import and modify the tuning script
            from tune_tactile_controller import main as tune_main, ray
            
            # If local mode requested, set it up
            if args.local_mode:
                if ray.is_initialized():
                    ray.shutdown()
                ray.init(local_mode=True, ignore_reinit_error=True)
                print("Running in local mode for debugging")
            
            # Note: The current tune_tactile_controller.py doesn't accept parameters
            # This would need to be modified to accept num_samples and num_trials
            print("Note: Using default configuration from tune_tactile_controller.py")
            print("To customize samples/trials, edit the script directly")
            
            return tune_main()
            
        except Exception as e:
            print(f"✗ Tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 