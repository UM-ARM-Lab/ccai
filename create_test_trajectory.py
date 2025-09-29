#!/usr/bin/env python3

import torch
import pickle
import numpy as np
import pathlib

def create_test_trajectory(T=50, dx=19, save_path="test_trajectory.pkl"):
    """
    Create a test trajectory for demonstration purposes.
    
    Args:
        T (int): Number of timesteps
        dx (int): Dimension of state vector (usually 19 for Allegro hand + screwdriver)
        save_path (str): Path to save the pickle file
    """
    
    # Create a simple trajectory that moves the hand joints smoothly
    # dx = 16 (allegro hand joints) + 3 (screwdriver orientation) = 19
    
    # Initialize trajectory tensor
    trajectory = torch.zeros(T, dx)
    
    # Set initial pose (based on default values from allegro_screwdriver.py)
    initial_pose = torch.cat([
        torch.tensor([0.0819, 0.3447, 0.7860, 0.7333]),  # finger 0 (index)
        torch.tensor([-0.0578, 0.7718, 0.5937, 0.7523]), # finger 1 (middle)  
        torch.tensor([0., 0.5, 0.65, 0.65]),             # finger 2 (ring)
        torch.tensor([0.7946, 0.8216, 0.7075, 0.8364]),  # finger 3 (thumb)
        torch.tensor([0., 0., 0.])                        # screwdriver orientation (roll, pitch, yaw)
    ])
    
    # Create smooth trajectory
    for t in range(T):
        # Copy initial pose
        trajectory[t] = initial_pose.clone()
        
        # Add smooth motion to some joints
        time_factor = t / T
        
        # Gradually close/open some fingers
        trajectory[t, 1] += 0.3 * np.sin(2 * np.pi * time_factor)  # index finger joint 1
        trajectory[t, 5] += 0.2 * np.sin(2 * np.pi * time_factor)  # middle finger joint 1
        trajectory[t, 13] += 0.2 * np.sin(2 * np.pi * time_factor) # thumb joint 1
        
        # Rotate screwdriver slowly around z-axis
        trajectory[t, -1] = 0.5 * np.sin(np.pi * time_factor)  # yaw rotation
    
    # Save trajectory to pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(trajectory, f)
    
    print(f"Created test trajectory with shape {trajectory.shape}")
    print(f"Saved to: {save_path}")
    
    return trajectory

if __name__ == "__main__":
    # Create test trajectory
    trajectory = create_test_trajectory(T=100, dx=19, save_path="test_trajectory.pkl")
    
    print("\nTo replay this trajectory, run:")
    print("python replay_trajectory.py test_trajectory.pkl")
    print("\nOr with custom options:")
    print("python replay_trajectory.py test_trajectory.pkl --step-delay 0.05 --save-video --video-path ./test_replay")
