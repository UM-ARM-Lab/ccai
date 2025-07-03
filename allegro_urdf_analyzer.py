"""
Allegro URDF Analyzer - Extract kinematic structure for optimization

This script analyzes the Allegro hand URDF to extract:
1. Joint chains for each finger
2. Static transform matrices
3. Joint axes and types
4. Frame relationships

This information is used to create optimized forward kinematics.
"""

import torch
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import pytorch_kinematics as pk

def parse_origin(origin_elem) -> Tuple[List[float], List[float]]:
    """Parse origin element from URDF."""
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    
    if origin_elem is not None:
        if 'xyz' in origin_elem.attrib:
            xyz = [float(x) for x in origin_elem.attrib['xyz'].split()]
        if 'rpy' in origin_elem.attrib:
            rpy = [float(x) for x in origin_elem.attrib['rpy'].split()]
    
    return xyz, rpy

def create_transform_matrix(xyz: List[float], rpy: List[float]) -> torch.Tensor:
    """Create 4x4 homogeneous transform matrix from xyz translation and rpy rotation."""
    T = torch.eye(4, dtype=torch.float32)
    
    # Set translation
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1] 
    T[2, 3] = xyz[2]
    
    # Apply RPY rotation (ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll))
    roll, pitch, yaw = rpy
    
    # Individual rotation matrices
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    
    # Combined rotation matrix
    R = torch.tensor([
        [c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r],
        [s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r - c_y*s_r],
        [-s_p,    c_p*s_r,                c_p*c_r               ]
    ], dtype=torch.float32)
    
    T[:3, :3] = R
    return T

def analyze_allegro_urdf(urdf_path: str) -> Dict:
    """
    Analyze Allegro URDF and extract kinematic structure.
    
    Returns:
        Dictionary containing finger chains, static transforms, and joint info
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Parse all joints
    joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        joint_type = joint.get('type')
        
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        origin_elem = joint.find('origin')
        xyz, rpy = parse_origin(origin_elem)
        
        axis_elem = joint.find('axis')
        axis = [0, 0, 1]  # Default
        if axis_elem is not None:
            axis_str = axis_elem.attrib['xyz']
            axis = [float(x) for x in axis_str.split()]
        
        joints[name] = {
            'type': joint_type,
            'parent': parent,
            'child': child,
            'xyz': xyz,
            'rpy': rpy,
            'axis': axis,
            'static_transform': create_transform_matrix(xyz, rpy)
        }
    
    # Finger definitions - only include transforms that matter for FK
    finger_configs = {
        'index': {
            'joints': [
                'allegro_hand_hitosashi_finger_finger_joint_0',
                'allegro_hand_hitosashi_finger_finger_joint_1', 
                'allegro_hand_hitosashi_finger_finger_joint_2',
                'allegro_hand_hitosashi_finger_finger_joint_3'
            ],
            'fixed_joints': [
                'allegro_hand_hitosashi_finger_finger_0_aftc_fixed'
            ],
            'ee_link': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'joint_indices': [0, 1, 2, 3]
        },
        'middle': {
            'joints': [
                'allegro_hand_naka_finger_finger_joint_4',
                'allegro_hand_naka_finger_finger_joint_5',
                'allegro_hand_naka_finger_finger_joint_6', 
                'allegro_hand_naka_finger_finger_joint_7'
            ],
            'fixed_joints': [
                'allegro_hand_naka_finger_finger_1_aftc_fixed'
            ],
            'ee_link': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'joint_indices': [4, 5, 6, 7]
        },
        'ring': {
            'joints': [
                'allegro_hand_kusuri_finger_finger_joint_8',
                'allegro_hand_kusuri_finger_finger_joint_9',
                'allegro_hand_kusuri_finger_finger_joint_10',
                'allegro_hand_kusuri_finger_finger_joint_11'
            ],
            'fixed_joints': [
                'allegro_hand_kusuri_finger_finger_2_aftc_fixed'
            ],
            'ee_link': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'joint_indices': [8, 9, 10, 11]
        },
        'thumb': {
            'joints': [
                'allegro_hand_oya_finger_joint_12',
                'allegro_hand_oya_finger_joint_13',
                'allegro_hand_oya_finger_joint_14',
                'allegro_hand_oya_finger_joint_15'
            ],
            'fixed_joints': [
                'allegro_hand_oya_finger_3_aftc_fixed'
            ],
            'ee_link': 'allegro_hand_oya_finger_3_aftc_base_link',
            'joint_indices': [12, 13, 14, 15]
        }
    }
    
    # Build kinematic chains for each finger
    kinematic_chains = {}
    for finger_name, config in finger_configs.items():
        chain = []
        
        # Add revolute joints
        for joint_name in config['joints']:
            if joint_name in joints:
                chain.append(joints[joint_name])
        
        # Add fixed joints  
        for joint_name in config['fixed_joints']:
            if joint_name in joints:
                chain.append(joints[joint_name])
        
        kinematic_chains[finger_name] = {
            'chain': chain,
            'ee_link': config['ee_link'], 
            'joint_indices': config['joint_indices']
        }
    
    # Compute static transform products for optimization
    static_transforms = {}
    for finger_name, data in kinematic_chains.items():
        chain = data['chain']
        
        # Pre-compute cumulative static transforms
        cumulative_transforms = []
        T_cumulative = torch.eye(4, dtype=torch.float32)
        
        for i, joint in enumerate(chain):
            T_cumulative = T_cumulative @ joint['static_transform']
            cumulative_transforms.append(T_cumulative.clone())
        
        static_transforms[finger_name] = cumulative_transforms
    
    return {
        'joints': joints,
        'finger_configs': finger_configs,
        'kinematic_chains': kinematic_chains,
        'static_transforms': static_transforms
    }

def print_analysis(analysis: Dict):
    """Print analysis results."""
    print("=== Allegro Hand URDF Analysis ===\n")
    
    for finger_name, data in analysis['kinematic_chains'].items():
        print(f"== {finger_name.upper()} FINGER ==")
        print(f"End effector: {data['ee_link']}")
        print(f"Joint indices: {data['joint_indices']}")
        print(f"Chain length: {len(data['chain'])} transforms")
        
        for i, joint in enumerate(data['chain']):
            print(f"  {i}: {joint.get('type', 'unknown')} - xyz: {joint['xyz']}, rpy: {joint['rpy']}")
            if joint['type'] == 'revolute':
                print(f"      axis: {joint['axis']}")
        print()

if __name__ == "__main__":
    urdf_path = "allegro_hand_right.urdf"
    analysis = analyze_allegro_urdf(urdf_path)
    print_analysis(analysis)
    
    # Save analysis for optimization
    torch.save(analysis, "allegro_kinematic_analysis.pt")
    print("✓ Saved analysis to allegro_kinematic_analysis.pt") 