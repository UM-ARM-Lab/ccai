#!/usr/bin/env python3
"""
Test script for the inverse kinematics functionality.
This script demonstrates how to use the InverseObjectPoseCalculator to get
screwdriver position from arm configuration.
"""

import torch
import numpy as np
import sys
import os

# Add the hardware directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hardware'))

from hardware_env_inverse import InverseObjectPoseCalculator

def test_inverse_kinematics():
    """Test the inverse kinematics functionality."""
    print("Testing Inverse Kinematics Functionality")
    print("=" * 50)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the inverse calculator
    try:
        inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver_catching', device=device)
        print("✓ Successfully initialized InverseObjectPoseCalculator")
    except Exception as e:
        print(f"✗ Failed to initialize InverseObjectPoseCalculator: {e}")
        return False
    
    # Test with different arm configurations
    test_configs = [
        # Configuration from original code (in degrees, converted to radians)
        torch.tensor([50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78], device=device) / 180 * np.pi,
        # Zero configuration
        torch.zeros(7, device=device),
        # Random configuration
        torch.randn(7, device=device) * 0.5,
    ]
    
    # Example arm base pose in mocap world frame
    arm_base_trans = np.array([0.1, 0.2, 0.3])  # Example position
    arm_base_euler = np.array([0.1, 0.2, 0.3])  # Example orientation
    
    for i, arm_config in enumerate(test_configs):
        print(f"\nTest {i+1}: Arm configuration {arm_config.cpu().numpy() / np.pi * 180} degrees")
        
        try:
            # Calculate screwdriver pose with arm base pose
            screwdriver_trans, screwdriver_euler = inverse_calc.get_screwdriver_pose_from_arm_config(
                arm_config, arm_base_trans, arm_base_euler
            )
            print(f"  ✓ Screwdriver position: {screwdriver_trans}")
            print(f"  ✓ Screwdriver orientation: {screwdriver_euler}")
            
            # Calculate hand pose without arm base pose
            hand_trans, hand_euler = inverse_calc.get_screwdriver_pose_from_arm_config(arm_config)
            print(f"  ✓ Hand position (arm mocap frame): {hand_trans}")
            print(f"  ✓ Hand orientation (arm mocap frame): {hand_euler}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    return True

def compare_with_forward_kinematics():
    """Compare inverse calculation with forward kinematics to verify correctness."""
    print("\nVerifying Inverse Kinematics with Forward Kinematics")
    print("=" * 60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    try:
        inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver_catching', device=device)
        
        # Test configuration
        arm_config = torch.tensor([50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78], device=device) / 180 * np.pi
        
        # Get hand pose using forward kinematics
        hand_pose = inverse_calc.chain.forward_kinematics(arm_config.unsqueeze(0))
        hand_matrix = hand_pose.get_matrix()[0]
        hand_trans_fk = hand_matrix[:3, 3].cpu().numpy()
        hand_rot_fk = hand_matrix[:3, :3].cpu().numpy()
        
        # Get hand pose using our inverse method (without arm base pose)
        hand_trans_inv, hand_euler_inv = inverse_calc.get_screwdriver_pose_from_arm_config(arm_config)
        
        print(f"Forward kinematics hand position: {hand_trans_fk}")
        print(f"Inverse method hand position: {hand_trans_inv}")
        print(f"Position difference: {np.linalg.norm(hand_trans_fk - hand_trans_inv)}")
        
        if np.linalg.norm(hand_trans_fk - hand_trans_inv) < 1e-6:
            print("✓ Hand positions match!")
        else:
            print("✗ Hand positions don't match - there might be an issue")
            
    except Exception as e:
        print(f"✗ Error in comparison: {e}")

if __name__ == "__main__":
    success = test_inverse_kinematics()
    if success:
        compare_with_forward_kinematics()
    else:
        print("Tests failed!")
        sys.exit(1)
