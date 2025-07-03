#!/usr/bin/env python3
"""Quick debug test to see which code paths are being taken."""

import torch
import pytorch_kinematics as pk
from allegro_optimized_kinematics import OptimizedAllegroChain

def quick_debug():
    # Load URDF
    with open("allegro_hand_right.urdf", 'r') as f:
        urdf_data = f.read()
    
    # Create chains
    standard_chain = pk.build_chain_from_urdf(urdf_data)
    optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    standard_chain = standard_chain.to(device=device)
    optimized_chain = optimized_chain.to(device=device)
    
    # Find finger end-effector frames
    finger_ee_frames = []
    for frame_name in standard_chain.frame_to_idx.keys():
        if 'ee' in frame_name.lower():  # end-effector frames
            finger_ee_frames.append(frame_name)
    
    print(f"Found finger end-effectors: {finger_ee_frames}")
    
    # Debug: Check what the analysis contains
    print("\n=== Analysis Content ===")
    if hasattr(optimized_chain, 'analysis'):
        print("Analysis keys:", list(optimized_chain.analysis.keys()))
        if 'kinematic_chains' in optimized_chain.analysis:
            print("Kinematic chains:")
            for finger_name, data in optimized_chain.analysis['kinematic_chains'].items():
                print(f"  {finger_name}: ee_link = {data.get('ee_link', 'NOT_SET')}")
    else:
        print("No analysis found!")
    
    # Find the frames that the analysis expects
    analysis_finger_frames = []
    if hasattr(optimized_chain, 'analysis') and 'kinematic_chains' in optimized_chain.analysis:
        for finger_name, data in optimized_chain.analysis['kinematic_chains'].items():
            ee_link = data.get('ee_link')
            if ee_link and ee_link in standard_chain.frame_to_idx:
                analysis_finger_frames.append(ee_link)
    
    print(f"Analysis expects frames: {analysis_finger_frames}")
    
    # Test with analysis frames if available, otherwise use ee frames
    test_frames = analysis_finger_frames if analysis_finger_frames else finger_ee_frames
    print(f"Using frames for test: {test_frames}")
    
    # Test with small batch
    batch_size = 4
    th = torch.randn(batch_size, standard_chain.n_joints, device=device)
    
    # Create diverse link indices
    link_indices = []
    for i in range(batch_size):
        frame_name = test_frames[i % len(test_frames)]
        link_indices.append(standard_chain.frame_to_idx[frame_name])
    
    print(f"Link indices: {link_indices}")
    print(f"Frame names: {[standard_chain.idx_to_frame[idx] for idx in link_indices]}")
    
    # Test optimized jacobian with debug output
    print("\n=== Testing Optimized Jacobian ===")
    jac_optimized = optimized_chain.jacobian(th, link_indices=link_indices)
    print(f"Output shape: {jac_optimized.shape}")

if __name__ == "__main__":
    quick_debug() 