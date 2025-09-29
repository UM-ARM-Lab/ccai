#!/usr/bin/env python3
"""
Calculate screwdriver pose for the given arm configuration.
"""
import numpy as np
import sys
import os

# Add the hardware directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hardware'))

from hardware.hardware_env_inverse import InverseObjectPoseCalculator
from hardware.hardware_env import ObjectPoseReader

# This import needs to be last
import torch

def calculate_screwdriver_pose():
    """Calculate screwdriver pose for the given arm configuration using mocap data."""
    
    # Your arm configuration in radians - ensure it's on CPU
    arm_config = torch.tensor([-0.3837, 0.5034, 0.2209, -1.7871, -1.0786, -1.5783, -1.4622], device='cpu')
    
    device = 'cpu'  # Force CPU to avoid device mismatch issues
    print(f"Using device: {device}")
    print(f"Arm configuration (radians): {arm_config.numpy()}")
    print(f"Arm configuration (degrees): {arm_config.numpy() / np.pi * 180}")
    
    # Initialize the inverse calculator on CPU
    inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver_catching', device=device)
    print("✓ Successfully initialized InverseObjectPoseCalculator")
    
    # Use ObjectPoseReader exactly like in hardware_env.py to get mocap data
    import rospy
    
    # Initialize ROS node if not already done
    rospy.init_node('screwdriver_pose_calculator', anonymous=True)
    
    print("Waiting for mocap data...")
    
    # Use ObjectPoseReader exactly like in hardware_env.py
    obj_reader = ObjectPoseReader(obj='blue_screwdriver_catching', mode='relative', device=device)
    
    # Wait for mocap data to be available
    timeout = 10  # seconds
    start_time = rospy.Time.now()
    while not hasattr(obj_reader, 'arm_base') or obj_reader.arm_base is None:
        if (rospy.Time.now() - start_time).to_sec() > timeout:
            print("✗ No mocap data received within timeout. Using default values.")
            arm_base_trans = np.array([0.0, 0.0, 0.0])
            arm_base_euler = np.array([0.0, 0.0, 0.0])
            break
        rospy.sleep(0.1)
    
    if hasattr(obj_reader, 'arm_base') and obj_reader.arm_base is not None:
        print("✓ Received mocap data for arm base")
        # Get arm base pose using the exact same method as hardware_env.py
        arm_base_euler, arm_base_trans = obj_reader.euler_trans_from_segment(obj_reader.arm_base.segments[0])
    else:
        print("✗ No mocap data available. Using default values.")
        arm_base_trans = np.array([0.0, 0.0, 0.0])
        arm_base_euler = np.array([0.0, 0.0, 0.0])
    
    print(f"\nArm base position (mocap world): {arm_base_trans}")
    print(f"Arm base orientation (mocap world): {arm_base_euler}")
    print(f"Arm base orientation (degrees): {arm_base_euler / np.pi * 180}")
    
    # Calculate screwdriver pose in mocap world frame
    screwdriver_trans, screwdriver_euler = inverse_calc.get_screwdriver_pose_from_arm_config(
        arm_config, arm_base_trans, arm_base_euler
    )
    
    print("\n" + "="*60)
    print("SCREWDRIVER POSE IN MOCAP WORLD FRAME")
    print("="*60)
    print(f"Screwdriver position (x, y, z): {screwdriver_trans}")
    print(f"Screwdriver orientation (roll, pitch, yaw): {screwdriver_euler}")
    print(f"Screwdriver orientation (degrees): {screwdriver_euler / np.pi * 180}")
    
    # Also calculate hand pose in arm mocap frame (without arm base pose)
    hand_trans, hand_euler = inverse_calc.get_screwdriver_pose_from_arm_config(arm_config)
    
    print("\n" + "="*60)
    print("HAND POSE IN ARM MOCAP FRAME")
    print("="*60)
    print(f"Hand position (x, y, z): {hand_trans}")
    print(f"Hand orientation (roll, pitch, yaw): {hand_euler}")
    print(f"Hand orientation (degrees): {hand_euler / np.pi * 180}")

if __name__ == "__main__":
    print("Calculating Screwdriver Pose for Given Arm Configuration")
    print("="*70)
    
    calculate_screwdriver_pose()
    
