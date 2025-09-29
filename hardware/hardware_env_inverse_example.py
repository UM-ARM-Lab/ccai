#!/usr/bin/env python3
"""
Example usage of the inverse kinematics functionality.
This shows how to integrate the InverseObjectPoseCalculator with the existing hardware environment.
"""

import torch
import numpy as np
import rospy
import pathlib
from hardware_env_inverse import InverseObjectPoseCalculator, ObjectPoseReader
import pytorch_kinematics as pk

urdf_path = "/home/abhinav/Documents/git_packages/isaacgym-arm-envs/isaac_victor_envs/assets/xela_models/victor_allegro_stalk.urdf"
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

def regularized_ik(n_tgts, device='cuda:0'):
    """
    Generate an initial guess for the IK solver that keeps the arm configuration close to previously used config
    """
    init_dof = torch.zeros((n_tgts, 7), device=device)
    # The arm config from ICRA 25 in degrees. To avoid strange local minima
    init_dof += torch.tensor([[50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78]], device=device)
    init_dof = init_dof / 180 * np.pi
    init_dof += torch.randn_like(init_dof) * 0.01
    return init_dof

class InverseHardwareEnv:
    """
    Extended hardware environment that can work in both directions:
    1. Forward: mocap data -> arm configuration (original functionality)
    2. Inverse: arm configuration -> expected mocap data (new functionality)
    """
    
    def __init__(self, device='cuda:0'):
        self.device = device
        
        # Initialize both forward and inverse calculators
        self.obj_reader = ObjectPoseReader(obj='blue_screwdriver_catching', mode='relative', device=device)
        self.inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver_catching', device=device)
        
        # Initialize kinematics chain for IK
        self.chain = pk.build_serial_chain_from_urdf(open(urdf_path, mode='rb').read(), 
                                                   'allegro_hand_base_link', 
                                                   root_link_name='victor_right_arm_link_1')
        self.chain = self.chain.to(device=device)
        lim = torch.tensor(self.chain.get_joint_limits(serial=True), device=device)
        self.ik = pk.PseudoInverseIK(self.chain, max_iterations=100, num_retries=20,
                                    joint_limits=lim.T,
                                    early_stopping_any_converged=True,
                                    early_stopping_no_improvement="any",
                                    debug=False,
                                    config_sampling_method=lambda n: regularized_ik(n, device),
                                    lr=0.2)
    
    def get_arm_config_from_mocap(self):
        """
        Original functionality: Get arm configuration from mocap data
        """
        if hasattr(self.obj_reader, 'arm_base') and self.obj_reader.arm_base is not None:
            # Get target IK pose from mocap data
            tgt_ik_pose = self.obj_reader.get_target_IK_pose()
            sol = self.ik.solve(tgt_ik_pose.to(self.chain.device))
            converged_sol = sol.solutions[sol.converged]
            
            if converged_sol.shape[0] > 0:
                return converged_sol[0]
        return None
    
    def get_expected_mocap_from_arm_config(self, arm_config, arm_base_trans=None, arm_base_euler=None):
        """
        New functionality: Get expected mocap data from arm configuration
        """
        return self.inverse_calc.get_screwdriver_pose_from_arm_config(
            arm_config, arm_base_trans, arm_base_euler
        )
    
    def verify_arm_config(self, arm_config, tolerance=1e-3):
        """
        Verify that an arm configuration produces the expected screwdriver pose
        by comparing with mocap data if available.
        """
        if not hasattr(self.obj_reader, 'arm_base') or self.obj_reader.arm_base is None:
            print("No mocap data available for verification")
            return False
        
        # Get expected screwdriver pose from arm config
        arm_base_euler, arm_base_trans = self.obj_reader.euler_trans_from_segment(self.obj_reader.arm_base.segments[0])
        expected_trans, expected_euler = self.get_expected_mocap_from_arm_config(
            arm_config, arm_base_trans, arm_base_euler
        )
        
        # Get actual screwdriver pose from mocap
        actual_trans, actual_euler = self.obj_reader.get_state()
        
        # Compare positions
        pos_error = np.linalg.norm(expected_trans - actual_trans)
        rot_error = np.linalg.norm(expected_euler - actual_euler)
        
        print(f"Position error: {pos_error:.6f}")
        print(f"Rotation error: {rot_error:.6f}")
        
        if pos_error < tolerance and rot_error < tolerance:
            print("✓ Arm configuration matches mocap data within tolerance")
            return True
        else:
            print("✗ Arm configuration does not match mocap data")
            return False

def main():
    """
    Example usage demonstrating both forward and inverse functionality
    """
    print("Inverse Hardware Environment Example")
    print("=" * 50)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the environment
    env = InverseHardwareEnv(device=device)
    
    # Example 1: Test inverse calculation with a known arm configuration
    print("\n1. Testing inverse calculation...")
    arm_config = torch.tensor([50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78], device=device) / 180 * np.pi
    
    # Test with example arm base pose
    arm_base_trans = np.array([0.1, 0.2, 0.3])
    arm_base_euler = np.array([0.1, 0.2, 0.3])
    
    screwdriver_trans, screwdriver_euler = env.get_expected_mocap_from_arm_config(
        arm_config, arm_base_trans, arm_base_euler
    )
    
    print(f"Expected screwdriver position: {screwdriver_trans}")
    print(f"Expected screwdriver orientation: {screwdriver_euler}")
    
    # Example 2: Test without arm base pose (hand pose in arm mocap frame)
    print("\n2. Testing hand pose calculation...")
    hand_trans, hand_euler = env.get_expected_mocap_from_arm_config(arm_config)
    print(f"Hand position (arm mocap frame): {hand_trans}")
    print(f"Hand orientation (arm mocap frame): {hand_euler}")
    
    # Example 3: Test with multiple arm configurations
    print("\n3. Testing multiple arm configurations...")
    test_configs = [
        torch.zeros(7, device=device),
        torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5], device=device),
        torch.randn(7, device=device) * 0.3,
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nConfiguration {i+1}: {config.cpu().numpy() / np.pi * 180} degrees")
        try:
            trans, euler = env.get_expected_mocap_from_arm_config(config, arm_base_trans, arm_base_euler)
            print(f"  Screwdriver position: {trans}")
            print(f"  Screwdriver orientation: {euler}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")

if __name__ == "__main__":
    # Initialize ROS if not already done
    try:
        rospy.init_node('inverse_hardware_env_example', anonymous=True)
    except rospy.exceptions.ROSException:
        print("ROS already initialized or not available")
    
    main()
