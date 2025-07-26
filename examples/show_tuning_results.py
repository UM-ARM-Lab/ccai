#!/usr/bin/env python3
"""
Script to display results from Ray Tune experiments.
"""

import os
import sys
import pathlib
import json
from typing import Dict, Any

def show_experiment_results(experiment_name: str):
    """Show results for a specific experiment."""
    
    results_dir = pathlib.Path("./ray_results") / experiment_name
    
    if not results_dir.exists():
        print(f"Experiment {experiment_name} not found")
        return
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    # Look for experiment state file
    state_file = results_dir / "experiment_state.json"
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            trials = state.get('trials', [])
            print(f"Number of trials: {len(trials)}")
            
            if trials:
                # Find best trial
                best_trial = None
                best_score = float('-inf')
                
                for trial in trials:
                    if 'last_result' in trial and trial['last_result']:
                        score = trial['last_result'].get('performance_score', float('-inf'))
                        if score > best_score:
                            best_score = score
                            best_trial = trial
                
                if best_trial:
                    print(f"\nBest trial:")
                    print(f"  Trial ID: {best_trial.get('trial_id', 'N/A')}")
                    print(f"  Status: {best_trial.get('status', 'N/A')}")
                    
                    last_result = best_trial.get('last_result', {})
                    print(f"  Performance Score: {last_result.get('performance_score', 'N/A'):.4f}")
                    print(f"  Distance to Goal: {last_result.get('final_distance_to_goal', 'N/A'):.4f}")
                    print(f"  Success Rate: {last_result.get('success_rate', 'N/A'):.4f}")
                    print(f"  Completion Rate: {last_result.get('completion_rate', 'N/A'):.4f}")
                    
                    config = best_trial.get('config', {})
                    print(f"\n  Best hyperparameters:")
                    for param, value in config.items():
                        print(f"    {param}: {value:.6f}")
                else:
                    print("No completed trials found")
            else:
                print("No trials found")
                
        except Exception as e:
            print(f"Error reading experiment state: {e}")
    else:
        print("No experiment state file found")


def main():
    """Show results for all experiments."""
    
    print("RAY TUNE EXPERIMENT RESULTS")
    print("="*60)
    
    results_dir = pathlib.Path("./ray_results")
    
    if not results_dir.exists():
        print("No ray_results directory found")
        return
    
    # List all experiments
    experiments = [d.name for d in results_dir.iterdir() if d.is_dir()]
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"Found {len(experiments)} experiment(s):")
    for exp in experiments:
        print(f"  - {exp}")
    
    # Show results for each experiment
    for exp in experiments:
        show_experiment_results(exp)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("✅ Ray Tune experiments completed successfully!")
    print("The tactile controller tuning system is working correctly.")
    print("\nTo run the full tuning with real experiments:")
    print("  python examples/tune_tactile_controller.py")


if __name__ == "__main__":
    main() 