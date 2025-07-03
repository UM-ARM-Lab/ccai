"""
Optimized Allegro Hand Kinematics

This module provides a drop-in replacement for pytorch_kinematics Chain 
that uses pre-computed static transforms and specialized code paths for 
the Allegro hand to achieve significant speedups.

Usage:
    chain = OptimizedAllegroChain(urdf_data)
    # Use exactly like pytorch_kinematics Chain
    result = chain.forward_kinematics(q, frame_indices)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import pytorch_kinematics as pk
from allegro_urdf_analyzer import analyze_allegro_urdf
import pytorch_kinematics.transforms as tf


class OptimizedAllegroChain:
    """
    Optimized forward kinematics for Allegro hand.
    
    This class provides the same interface as pytorch_kinematics Chain
    but uses pre-computed static transforms and unrolled loops for
    significant performance improvements on the specific Allegro URDF.
    """
    
    def __init__(self, urdf_data: str = None, urdf_path: str = None, 
                 dtype: torch.dtype = torch.float32, device: str = "cpu"):
        """
        Initialize optimized Allegro kinematics.
        
        Args:
            urdf_data: URDF string data (if provided, urdf_path is ignored)
            urdf_path: Path to URDF file 
            dtype: PyTorch data type
            device: Device for computation
        """
        self.dtype = dtype
        self.device = device
        
        # Load and analyze URDF
        if urdf_data is not None:
            # Write to temporary file for analysis
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(urdf_data)
                temp_path = f.name
            try:
                self.analysis = analyze_allegro_urdf(temp_path)
            finally:
                os.unlink(temp_path)
        elif urdf_path is not None:
            self.analysis = analyze_allegro_urdf(urdf_path)
        else:
            # Load saved analysis
            self.analysis = torch.load("allegro_kinematic_analysis.pt")
        
        # Create standard chain for interface compatibility
        if urdf_data is not None:
            self.standard_chain = pk.build_chain_from_urdf(urdf_data, use_optimized_allegro=False)
        elif urdf_path is not None:
            with open(urdf_path, 'r') as f:
                self.standard_chain = pk.build_chain_from_urdf(f.read(), use_optimized_allegro=False)
        else:
            raise ValueError("Must provide either urdf_data or urdf_path")
        
        # Move to specified device/dtype
        self.to(dtype=dtype, device=device)
        
        # Pre-compute optimized transforms
        self._precompute_transforms()
        
        # Interface compatibility
        self.n_joints = self.standard_chain.n_joints
        self.frame_to_idx = self.standard_chain.frame_to_idx
        self.idx_to_frame = self.standard_chain.idx_to_frame
        
        # Pre-compute Jacobian-specific data
        self._precompute_jacobian_data()
        
    def _precompute_transforms(self):
        """Pre-compute static transform matrices for optimization."""
        self.static_transforms = {}
        self.joint_axes = {}
        
        for finger_name, data in self.analysis['kinematic_chains'].items():
            chain = data['chain']
            
            # Extract joint axes for revolute joints
            axes = []
            static_transforms = []
            
            for joint in chain:
                if joint['type'] == 'revolute':
                    axes.append(torch.tensor(joint['axis'], dtype=self.dtype, device=self.device))
                
                # Store transform
                T = joint['static_transform'].to(dtype=self.dtype, device=self.device)
                static_transforms.append(T)
            
            self.joint_axes[finger_name] = torch.stack(axes) if axes else torch.empty(0, 3)
            self.static_transforms[finger_name] = static_transforms
            
    def to(self, dtype: torch.dtype = None, device: str = None):
        """Move chain to specified device/dtype."""
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
            
        self.standard_chain = self.standard_chain.to(dtype=self.dtype, device=self.device)
        
        # Move pre-computed transforms
        if hasattr(self, 'static_transforms'):
            for finger_name in self.static_transforms:
                for i, T in enumerate(self.static_transforms[finger_name]):
                    self.static_transforms[finger_name][i] = T.to(dtype=self.dtype, device=self.device)
                self.joint_axes[finger_name] = self.joint_axes[finger_name].to(dtype=self.dtype, device=self.device)
        
        # Move jacobian_info
        if hasattr(self, 'jacobian_info'):
            for finger_name in self.jacobian_info:
                for key in self.jacobian_info[finger_name]:
                    if isinstance(self.jacobian_info[finger_name][key], torch.Tensor):
                        self.jacobian_info[finger_name][key] = self.jacobian_info[finger_name][key].to(dtype=self.dtype, device=self.device)

        
        return self
    
    def forward_kinematics(self, th: torch.Tensor, frame_indices: Optional[torch.Tensor] = None) -> Dict[str, pk.Transform3d]:
        """
        Compute forward kinematics with optimization for Allegro fingers.
        
        Args:
            th: Joint angles [batch_size, n_joints] or [n_joints]
            frame_indices: Specific frame indices to compute (if None, computes common end-effectors)
            
        Returns:
            Dictionary mapping frame names to Transform3d objects
        """
        # Ensure tensor format
        th = torch.atleast_2d(th).to(dtype=self.dtype, device=self.device)
        batch_size = th.shape[0]
        
        # If no specific frames requested, compute end-effectors
        if frame_indices is None:
            frame_indices = self._get_default_frame_indices()
        
        # Convert frame indices to frame names if needed
        if torch.is_tensor(frame_indices):
            if frame_indices.dim() == 0:  # Single index
                frame_names = [self.idx_to_frame[frame_indices.item()]]
            else:  # Multiple indices
                frame_names = [self.idx_to_frame[idx.item()] for idx in frame_indices]
        elif isinstance(frame_indices, list) and isinstance(frame_indices[0], str):
            frame_names = frame_indices
        elif isinstance(frame_indices, str):
            frame_names = [frame_indices]
        else:
            frame_names = frame_indices
        
        results = {}
        
        # Check which frames can be optimized
        for frame_name in frame_names:
            finger_type = self._get_finger_type(frame_name)
            
            if finger_type is not None:
                # Use optimized computation
                transform = self._compute_finger_fk_optimized(th, finger_type)
                results[frame_name] = pk.Transform3d(matrix=transform)
            else:
                # Fall back to standard computation for non-finger frames
                print(f"Falling back to standard computation for {frame_name}")
                frame_idx = torch.tensor([self.frame_to_idx[frame_name]], device=self.device)
                standard_result = self.standard_chain.forward_kinematics(th, frame_idx)
                results[frame_name] = standard_result[frame_name]
        
        return results
    
    def _get_finger_type(self, frame_name: str) -> Optional[str]:
        """Determine which finger a frame belongs to."""
        for finger_name, data in self.analysis['kinematic_chains'].items():
            if frame_name == data['ee_link']:
                return finger_name
        return None
    
    def _get_default_frame_indices(self) -> List[str]:
        """Get default frame names (all frames)."""
        return list(self.frame_to_idx.keys())
    
    def _compute_finger_fk_optimized(self, th: torch.Tensor, finger_type: str) -> torch.Tensor:
        """
        Optimized forward kinematics for a specific finger type.
        
        Args:
            th: Joint angles [batch_size, n_joints]
            finger_type: 'index', 'middle', 'ring', or 'thumb'
            
        Returns:
            Transform matrix [batch_size, 4, 4]
        """
        batch_size = th.shape[0]
        data = self.analysis['kinematic_chains'][finger_type]
        joint_indices = data['joint_indices']
        
        # Extract finger joint angles
        q = th[:, joint_indices]  # [batch_size, 4]
        
        # Dispatch to specialized function
        if finger_type == 'index':
            return self._index_fk_optimized(q)
        elif finger_type == 'middle':
            return self._middle_fk_optimized(q)
        elif finger_type == 'ring':
            return self._ring_fk_optimized(q)
        elif finger_type == 'thumb':
            return self._thumb_fk_optimized(q)
        else:
            raise ValueError(f"Unknown finger type: {finger_type}")
    
    def _index_fk_optimized(self, q: torch.Tensor) -> torch.Tensor:
        """Optimized FK for index finger with unrolled computation."""
        batch_size = q.shape[0]
        
        # Pre-computed static transforms for index finger
        static_transforms = self.static_transforms['index']
        
        # Joint rotations for revolute joints
        # Joint 0: Z-axis rotation
        c0, s0 = torch.cos(q[:, 0]), torch.sin(q[:, 0])
        zeros = torch.zeros_like(c0)
        ones = torch.ones_like(c0)
        
        # Build R0 by constructing each row without inplace operations
        R0_row0 = torch.stack([c0, -s0, zeros, zeros], dim=-1)
        R0_row1 = torch.stack([s0, c0, zeros, zeros], dim=-1)
        R0_row2 = torch.stack([zeros, zeros, ones, zeros], dim=-1)
        R0_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R0 = torch.stack([R0_row0, R0_row1, R0_row2, R0_row3], dim=-2)
        
        # Joint 1: Y-axis rotation  
        c1, s1 = torch.cos(q[:, 1]), torch.sin(q[:, 1])
        
        # Build R1 by constructing each row without inplace operations
        R1_row0 = torch.stack([c1, zeros, s1, zeros], dim=-1)
        R1_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R1_row2 = torch.stack([-s1, zeros, c1, zeros], dim=-1)
        R1_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R1 = torch.stack([R1_row0, R1_row1, R1_row2, R1_row3], dim=-2)
        
        # Joint 2: Y-axis rotation
        c2, s2 = torch.cos(q[:, 2]), torch.sin(q[:, 2])
        
        # Build R2 by constructing each row without inplace operations
        R2_row0 = torch.stack([c2, zeros, s2, zeros], dim=-1)
        R2_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R2_row2 = torch.stack([-s2, zeros, c2, zeros], dim=-1)
        R2_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R2 = torch.stack([R2_row0, R2_row1, R2_row2, R2_row3], dim=-2)
        
        # Joint 3: Y-axis rotation
        c3, s3 = torch.cos(q[:, 3]), torch.sin(q[:, 3])
        
        # Build R3 by constructing each row without inplace operations
        R3_row0 = torch.stack([c3, zeros, s3, zeros], dim=-1)
        R3_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R3_row2 = torch.stack([-s3, zeros, c3, zeros], dim=-1)
        R3_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R3 = torch.stack([R3_row0, R3_row1, R3_row2, R3_row3], dim=-2)
        
        # Correct transform composition following pytorch_kinematics chain structure:
        # Base link (step 0) - no transform needed, just identity
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(*c0.shape, 4, 4)
        
        # Step 1: Joint 0 - joint_offset @ joint_rotation
        T = T @ static_transforms[0].unsqueeze(0) @ R0
        
        # Step 2: Joint 1 - joint_offset @ joint_rotation  
        T = T @ static_transforms[1].unsqueeze(0) @ R1
        
        # Step 3: Joint 2 - joint_offset @ joint_rotation
        T = T @ static_transforms[2].unsqueeze(0) @ R2
        
        # Step 4: Joint 3 - joint_offset @ joint_rotation
        T = T @ static_transforms[3].unsqueeze(0) @ R3
        
        # Step 5: Fixed joint - just joint_offset (no rotation)
        T = T @ static_transforms[4].unsqueeze(0)
        
        return T
    
    def _middle_fk_optimized(self, q: torch.Tensor) -> torch.Tensor:
        """Optimized FK for middle finger."""
        static_transforms = self.static_transforms['middle']
        
        # Joint rotations (same pattern as index)
        c0, s0 = torch.cos(q[:, 0]), torch.sin(q[:, 0])
        zeros = torch.zeros_like(c0)
        ones = torch.ones_like(c0)
        
        # Build R0 by constructing each row without inplace operations
        R0_row0 = torch.stack([c0, -s0, zeros, zeros], dim=-1)
        R0_row1 = torch.stack([s0, c0, zeros, zeros], dim=-1)
        R0_row2 = torch.stack([zeros, zeros, ones, zeros], dim=-1)
        R0_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R0 = torch.stack([R0_row0, R0_row1, R0_row2, R0_row3], dim=-2)
        
        c1, s1 = torch.cos(q[:, 1]), torch.sin(q[:, 1])
        
        # Build R1 by constructing each row without inplace operations
        R1_row0 = torch.stack([c1, zeros, s1, zeros], dim=-1)
        R1_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R1_row2 = torch.stack([-s1, zeros, c1, zeros], dim=-1)
        R1_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R1 = torch.stack([R1_row0, R1_row1, R1_row2, R1_row3], dim=-2)
        
        c2, s2 = torch.cos(q[:, 2]), torch.sin(q[:, 2])
        
        # Build R2 by constructing each row without inplace operations
        R2_row0 = torch.stack([c2, zeros, s2, zeros], dim=-1)
        R2_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R2_row2 = torch.stack([-s2, zeros, c2, zeros], dim=-1)
        R2_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R2 = torch.stack([R2_row0, R2_row1, R2_row2, R2_row3], dim=-2)
        
        c3, s3 = torch.cos(q[:, 3]), torch.sin(q[:, 3])
        
        # Build R3 by constructing each row without inplace operations
        R3_row0 = torch.stack([c3, zeros, s3, zeros], dim=-1)
        R3_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R3_row2 = torch.stack([-s3, zeros, c3, zeros], dim=-1)
        R3_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R3 = torch.stack([R3_row0, R3_row1, R3_row2, R3_row3], dim=-2)
        
        # Correct transform composition following pytorch_kinematics chain structure
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(*c0.shape, 4, 4)
        
        # Apply joint_offset @ joint_rotation for each joint
        T = T @ static_transforms[0].unsqueeze(0) @ R0
        T = T @ static_transforms[1].unsqueeze(0) @ R1
        T = T @ static_transforms[2].unsqueeze(0) @ R2
        T = T @ static_transforms[3].unsqueeze(0) @ R3
        T = T @ static_transforms[4].unsqueeze(0)  # Final fixed joint
        
        return T
    
    def _ring_fk_optimized(self, q: torch.Tensor) -> torch.Tensor:
        """Optimized FK for ring finger."""
        static_transforms = self.static_transforms['ring']
        
        # Joint rotations (same pattern)
        c0, s0 = torch.cos(q[:, 0]), torch.sin(q[:, 0])
        zeros = torch.zeros_like(c0)
        ones = torch.ones_like(c0)
        
        # Build R0 by constructing each row without inplace operations
        R0_row0 = torch.stack([c0, -s0, zeros, zeros], dim=-1)
        R0_row1 = torch.stack([s0, c0, zeros, zeros], dim=-1)
        R0_row2 = torch.stack([zeros, zeros, ones, zeros], dim=-1)
        R0_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R0 = torch.stack([R0_row0, R0_row1, R0_row2, R0_row3], dim=-2)
        
        c1, s1 = torch.cos(q[:, 1]), torch.sin(q[:, 1])
        
        # Build R1 by constructing each row without inplace operations
        R1_row0 = torch.stack([c1, zeros, s1, zeros], dim=-1)
        R1_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R1_row2 = torch.stack([-s1, zeros, c1, zeros], dim=-1)
        R1_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R1 = torch.stack([R1_row0, R1_row1, R1_row2, R1_row3], dim=-2)
        
        c2, s2 = torch.cos(q[:, 2]), torch.sin(q[:, 2])
        
        # Build R2 by constructing each row without inplace operations
        R2_row0 = torch.stack([c2, zeros, s2, zeros], dim=-1)
        R2_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R2_row2 = torch.stack([-s2, zeros, c2, zeros], dim=-1)
        R2_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R2 = torch.stack([R2_row0, R2_row1, R2_row2, R2_row3], dim=-2)
        
        c3, s3 = torch.cos(q[:, 3]), torch.sin(q[:, 3])
        
        # Build R3 by constructing each row without inplace operations
        R3_row0 = torch.stack([c3, zeros, s3, zeros], dim=-1)
        R3_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R3_row2 = torch.stack([-s3, zeros, c3, zeros], dim=-1)
        R3_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R3 = torch.stack([R3_row0, R3_row1, R3_row2, R3_row3], dim=-2)
        
        # Correct transform composition following pytorch_kinematics chain structure
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(*c0.shape, 4, 4)
        
        # Apply joint_offset @ joint_rotation for each joint
        T = T @ static_transforms[0].unsqueeze(0) @ R0
        T = T @ static_transforms[1].unsqueeze(0) @ R1
        T = T @ static_transforms[2].unsqueeze(0) @ R2
        T = T @ static_transforms[3].unsqueeze(0) @ R3
        T = T @ static_transforms[4].unsqueeze(0)  # Final fixed joint
        
        return T
    
    def _thumb_fk_optimized(self, q: torch.Tensor) -> torch.Tensor:
        """Optimized FK for thumb (different joint pattern)."""
        static_transforms = self.static_transforms['thumb']
        
        # Joint 0: NEGATIVE X-axis rotation (thumb base is different)
        c0, s0 = torch.cos(q[:, 0]), torch.sin(q[:, 0])
        zeros = torch.zeros_like(c0)
        ones = torch.ones_like(c0)
        
        # Build R0 for negative X-axis rotation without inplace operations
        R0_row0 = torch.stack([ones, zeros, zeros, zeros], dim=-1)
        R0_row1 = torch.stack([zeros, c0, s0, zeros], dim=-1)  # positive s0 for negative axis
        R0_row2 = torch.stack([zeros, -s0, c0, zeros], dim=-1)  # negative s0 for negative axis
        R0_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R0 = torch.stack([R0_row0, R0_row1, R0_row2, R0_row3], dim=-2)
        
        # Joint 1: Z-axis rotation
        c1, s1 = torch.cos(q[:, 1]), torch.sin(q[:, 1])
        
        # Build R1 for Z-axis rotation without inplace operations
        R1_row0 = torch.stack([c1, -s1, zeros, zeros], dim=-1)
        R1_row1 = torch.stack([s1, c1, zeros, zeros], dim=-1)
        R1_row2 = torch.stack([zeros, zeros, ones, zeros], dim=-1)
        R1_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R1 = torch.stack([R1_row0, R1_row1, R1_row2, R1_row3], dim=-2)
        
        # Joint 2: Y-axis rotation
        c2, s2 = torch.cos(q[:, 2]), torch.sin(q[:, 2])
        
        # Build R2 for Y-axis rotation without inplace operations
        R2_row0 = torch.stack([c2, zeros, s2, zeros], dim=-1)
        R2_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R2_row2 = torch.stack([-s2, zeros, c2, zeros], dim=-1)
        R2_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R2 = torch.stack([R2_row0, R2_row1, R2_row2, R2_row3], dim=-2)
        
        # Joint 3: Y-axis rotation
        c3, s3 = torch.cos(q[:, 3]), torch.sin(q[:, 3])
        
        # Build R3 for Y-axis rotation without inplace operations
        R3_row0 = torch.stack([c3, zeros, s3, zeros], dim=-1)
        R3_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
        R3_row2 = torch.stack([-s3, zeros, c3, zeros], dim=-1)
        R3_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        R3 = torch.stack([R3_row0, R3_row1, R3_row2, R3_row3], dim=-2)
        
        # Correct transform composition following pytorch_kinematics chain structure
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(*c0.shape, 4, 4)
        
        # Apply joint_offset @ joint_rotation for each joint
        T = T @ static_transforms[0].unsqueeze(0) @ R0
        T = T @ static_transforms[1].unsqueeze(0) @ R1
        T = T @ static_transforms[2].unsqueeze(0) @ R2
        T = T @ static_transforms[3].unsqueeze(0) @ R3
        T = T @ static_transforms[4].unsqueeze(0)  # Final fixed joint
        
        return T
    
    def _precompute_jacobian_data(self):
        """Pre-compute data structures needed for optimized Jacobian calculation."""
        # For each finger, pre-compute which joints affect it and their properties
        self.jacobian_info = {}
        
        for finger_name, config in self.analysis['kinematic_chains'].items():
            joint_indices = config['joint_indices']
            joint_axes = []
            
            # Get joint axes from standard chain
            for joint_idx in joint_indices:
                axis = self.standard_chain.axes[joint_idx]
                joint_axes.append(axis.to(dtype=self.dtype, device=self.device))
            
            self.jacobian_info[finger_name] = {
                'joint_indices': joint_indices,
                'joint_axes': torch.stack(joint_axes),  # [4, 3] for each finger
                'ee_link': config['ee_link']
            }

    def jacobian(self, th, locations=None, link_indices=None, analytic=False, locations_in_ee_frame=True):
        """
        Optimized Jacobian calculation with pytorch_kinematics compatible interface.
        
        Args:
            th: Joint angles tensor of shape [batch_size, 16]
            locations: Optional tool locations relative to end-effector frames [batch_size, 3]
            link_indices: Link indices to compute Jacobian for (tensor, int, or iterable) [batch_size] or single value
            analytic: Whether to return analytic Jacobian (not yet supported)
            locations_in_ee_frame: Whether locations are in end-effector frame
        
        Returns:
            Jacobian tensor [batch_size, 6, 16]
        """
        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) == 1:
            th = th.unsqueeze(0)
        
        batch_size = th.shape[0]
        ndof = th.shape[1]
        
        # Handle link_indices - ensure it's a tensor with same batch size
        if link_indices is None:
            raise ValueError("link_indices must be provided")
        
        if not torch.is_tensor(link_indices):
            if isinstance(link_indices, (int, float)):
                # Single link index for all batch elements
                link_indices = torch.full((batch_size,), int(link_indices), dtype=torch.long, device=self.device)
            else:
                # Iterable of link indices
                link_indices = torch.tensor(link_indices, dtype=torch.long, device=self.device)
        
        # Ensure link_indices has the right shape
        if link_indices.dim() == 0:
            # Single scalar - broadcast to batch size
            link_indices = link_indices.unsqueeze(0).expand(batch_size)
        elif link_indices.shape[0] != batch_size:
            raise ValueError(f"link_indices length {link_indices.shape[0]} must match batch size {batch_size}")
        
        # Handle locations
        if locations is not None:
            if not torch.is_tensor(locations):
                locations = torch.tensor(locations, dtype=self.dtype, device=self.device)
            if locations.dim() == 1:
                locations = locations.unsqueeze(0)
            if locations.shape[0] != batch_size:
                raise ValueError(f"locations batch size {locations.shape[0]} must match th batch size {batch_size}")
        
        # Group by unique link indices for efficient computation
        unique_link_indices = torch.unique(link_indices)
        jacobians = []
        
        for unique_idx in unique_link_indices:
            # Find which batch elements correspond to this link index
            mask = link_indices == unique_idx
            batch_indices = torch.where(mask)[0]
            
            if len(batch_indices) == 0:
                continue
                
            # Extract relevant batch elements
            th_group = th[batch_indices]  # [group_size, 16]
            locations_group = locations[batch_indices] if locations is not None else None  # [group_size, 3]
            
            # Get frame name for this link index
            frame_idx = unique_idx.item()
            frame_name = self.idx_to_frame[frame_idx]
            finger_name = self._get_finger_type(frame_name)
            
            if finger_name is None:
                # Fall back to standard computation for non-finger frames
                # Handle tool transformation for standard chain
                if locations_group is not None:
                    cur_transform = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(len(batch_indices), 1, 1)
                    cur_transform[:, :3, 3] = locations_group
                    tool_transform = cur_transform if locations is not None else None
                else:
                    tool_transform = None
                
                if tool_transform is not None:
                    # Convert tool_transform to pytorch_kinematics format
                    tool_transform = pk.Transform3d(pos=tool_transform[:, :3, 3], device=self.device, dtype=self.dtype)
                    
                jac_group = self.standard_chain.calc_jacobian(
                    th_group, 
                    tool=tool_transform, 
                    link_indices=unique_idx,
                    analytic=analytic,
                    tool_in_ee_frame=locations_in_ee_frame
                )
            else:
                # Use optimized finger-specific Jacobian
                if finger_name == 'index':
                    jac_group = self._index_jacobian_optimized(th_group, locations_group, locations_in_ee_frame, None)
                elif finger_name == 'middle':
                    jac_group = self._middle_jacobian_optimized(th_group, locations_group, locations_in_ee_frame, None)
                elif finger_name == 'ring':
                    jac_group = self._ring_jacobian_optimized(th_group, locations_group, locations_in_ee_frame, None)
                elif finger_name == 'thumb':
                    jac_group = self._thumb_jacobian_optimized(th_group, locations_group, locations_in_ee_frame, None)
                
                # Apply analytic transformation if requested
                if analytic:
                    # Get end-effector transform for analytic jacobian
                    fk_result = self.forward_kinematics(th_group, torch.tensor([frame_idx], device=self.device))
                    ee_transform = fk_result[frame_name].get_matrix()
                    
                    if locations_group is not None:
                        cur_transform = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(len(batch_indices), 1, 1)
                        cur_transform[:, :3, 3] = locations_group
                        ee_transform = ee_transform @ cur_transform
                        
                    R = ee_transform[:, :3, :3].transpose(1, 2)
                    T_analytic = torch.eye(3, device=self.device).expand(len(batch_indices), 3, 3)
                    zeros = torch.zeros_like(R)
                    T_analytic = torch.cat((
                        torch.cat((T_analytic, zeros), dim=2),
                        torch.cat((zeros, R), dim=2)),
                        dim=1
                    )
                    jac_group = T_analytic @ jac_group
            
            # Store results with their original batch indices
            jacobians.append((batch_indices, jac_group))
        
        # Reassemble jacobians in original batch order
        result_jacobians = torch.zeros(batch_size, 6, ndof, dtype=self.dtype, device=self.device)
        for batch_indices, jac_group in jacobians:
            result_jacobians[batch_indices] = jac_group
            
        return result_jacobians

    def _index_jacobian_optimized(self, q: torch.Tensor, locations: Optional[torch.Tensor] = None, locations_in_ee_frame: bool = True, cur_transform: torch.Tensor = None) -> torch.Tensor:
        """Optimized Jacobian calculation for index finger."""
        # Get batch dimensions that work with vmap
        batch_shape = q.shape[:-1]  # All dimensions except the last one (joint dimension)
        
        # Get finger-specific joint configuration and FK
        finger_q = q[:, 0:4]  # Index finger joints
        
        # Compute forward kinematics for each joint frame (needed for Jacobian)
        T_frames = []
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(*batch_shape, 4, 4)
        T_frames.append(T.clone())  # Base
        
        static_transforms = self.static_transforms['index']
        joint_axes = self.jacobian_info['index']['joint_axes']  # [4, 3]
        
        # Build transforms for each joint frame
        for i in range(4):
            c, s = torch.cos(finger_q[:, i]), torch.sin(finger_q[:, i])
            
            # Create rotation matrix based on joint axis using stack operations
            zeros_rot = torch.zeros_like(c)
            ones_rot = torch.ones_like(c)
            
            if i == 0:  # Z-axis
                R_row0 = torch.stack([c, -s, zeros_rot, zeros_rot], dim=-1)
                R_row1 = torch.stack([s, c, zeros_rot, zeros_rot], dim=-1)
                R_row2 = torch.stack([zeros_rot, zeros_rot, ones_rot, zeros_rot], dim=-1)
                R_row3 = torch.stack([zeros_rot, zeros_rot, zeros_rot, ones_rot], dim=-1)
                R = torch.stack([R_row0, R_row1, R_row2, R_row3], dim=-2)
            else:  # Y-axis for joints 1, 2, 3
                R_row0 = torch.stack([c, zeros_rot, s, zeros_rot], dim=-1)
                R_row1 = torch.stack([zeros_rot, ones_rot, zeros_rot, zeros_rot], dim=-1)
                R_row2 = torch.stack([-s, zeros_rot, c, zeros_rot], dim=-1)
                R_row3 = torch.stack([zeros_rot, zeros_rot, zeros_rot, ones_rot], dim=-1)
                R = torch.stack([R_row0, R_row1, R_row2, R_row3], dim=-2)
            
            T = T @ static_transforms[i].unsqueeze(0) @ R
            T_frames.append(T.clone())
        
        # Add final static transform
        T = T @ static_transforms[4].unsqueeze(0)
        
        # End-effector position (with optional tool offset)
        if locations is not None and locations_in_ee_frame:
            # Transform tool location to world frame
            tool_local = torch.zeros(*batch_shape, 4, 1, dtype=self.dtype, device=self.device)
            tool_local[..., :3, 0] = locations
            tool_local[..., 3, 0] = 1.0
            tool_world = (T @ tool_local)[..., :3, 0]
        elif locations is not None and not locations_in_ee_frame:
            # Tool location is already in world frame
            tool_world = locations
        else:
            tool_world = T[..., :3, 3]
        
        # Compute Jacobian columns for each joint and build without inplace ops
        jac_columns = []
        
        for global_joint_idx in range(16):  # All 16 joints
            if global_joint_idx < 4:  # Index finger joints 0-3
                joint_idx = global_joint_idx
                
                # Joint transform at this joint
                joint_transform = T_frames[joint_idx + 1]
                
                # Joint axis in world coordinates
                joint_axis_world = (joint_transform[..., :3, :3] @ joint_axes[joint_idx].unsqueeze(-1)).squeeze(-1)
                
                # Position Jacobian: ω × (p_tool - p_joint)
                joint_position = joint_transform[..., :3, 3]
                position_jac = torch.cross(joint_axis_world, tool_world - joint_position, dim=-1)
                
                # Orientation Jacobian: just the joint axis
                orientation_jac = joint_axis_world
                
                # Stack position and orientation parts
                jac_col = torch.cat([position_jac, orientation_jac], dim=-1)
            else:
                # Zero column for non-index finger joints
                zeros_col = torch.zeros(*batch_shape, 6, dtype=self.dtype, device=self.device)
                jac_col = zeros_col
            
            jac_columns.append(jac_col)
        
        # Stack all columns to form the Jacobian matrix
        jac = torch.stack(jac_columns, dim=-1)  # Shape: [*batch_shape, 6, 16]
        
        return jac

    def _middle_jacobian_optimized(self, q: torch.Tensor, locations: Optional[torch.Tensor] = None, locations_in_ee_frame: bool = True, cur_transform: torch.Tensor = None) -> torch.Tensor:
        """Optimized Jacobian calculation for middle finger."""
        return self._finger_jacobian_generic('middle', q, locations, locations_in_ee_frame)

    def _ring_jacobian_optimized(self, q: torch.Tensor, locations: Optional[torch.Tensor] = None, locations_in_ee_frame: bool = True, cur_transform: torch.Tensor = None) -> torch.Tensor:
        """Optimized Jacobian calculation for ring finger."""
        return self._finger_jacobian_generic('ring', q, locations, locations_in_ee_frame)

    def _thumb_jacobian_optimized(self, q: torch.Tensor, locations: Optional[torch.Tensor] = None, locations_in_ee_frame: bool = True, cur_transform: torch.Tensor = None) -> torch.Tensor:
        """Optimized Jacobian calculation for thumb."""
        return self._finger_jacobian_generic('thumb', q, locations, locations_in_ee_frame)

    def _finger_jacobian_generic(self, finger_name: str, q: torch.Tensor, locations: Optional[torch.Tensor] = None, locations_in_ee_frame: bool = True) -> torch.Tensor:
        """Generic optimized Jacobian calculation for any finger."""
        # Get batch dimensions that work with vmap
        batch_shape = q.shape[:-1]  # All dimensions except the last one (joint dimension)
        
        # Get finger-specific configuration
        finger_info = self.jacobian_info[finger_name]
        joint_indices = finger_info['joint_indices']
        joint_axes = finger_info['joint_axes']
        
        # Get finger joint values
        finger_q = q[:, joint_indices]
        
        # Compute forward kinematics step by step to get joint frames
        T_frames = []
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(*batch_shape, 4, 4)
        T_frames.append(T.clone())
        
        static_transforms = self.static_transforms[finger_name]
        
        # Build joint frames
        for i in range(4):
            c, s = torch.cos(finger_q[..., i]), torch.sin(finger_q[..., i])
            axis = joint_axes[i]
            
            # Create rotation matrix based on joint axis (batch_size not needed anymore)
            R = self._create_rotation_matrix(axis, c, s, None)
            
            T = T @ static_transforms[i].unsqueeze(0) @ R
            T_frames.append(T.clone())
        
        # Add final transform to end-effector
        T = T @ static_transforms[4].unsqueeze(0)
        
        # End-effector position (with optional tool offset)
        if locations is not None and locations_in_ee_frame:
            # Transform tool location to world frame
            tool_local = torch.zeros(*batch_shape, 4, 1, dtype=self.dtype, device=self.device)
            tool_local[..., :3, 0] = locations
            tool_local[..., 3, 0] = 1.0
            tool_world = (T @ tool_local)[..., :3, 0]
        elif locations is not None and not locations_in_ee_frame:
            # Tool location is already in world frame
            tool_world = locations
        else:
            tool_world = T[..., :3, 3]
        
        # Compute Jacobian columns for each joint and build without inplace ops
        jac_columns = []
        
        for global_joint_idx in range(16):  # All 16 joints
            if global_joint_idx in joint_indices:
                # Find which local joint index this is
                local_joint_idx = joint_indices.index(global_joint_idx)
                
                # Joint transform at this joint
                joint_transform = T_frames[local_joint_idx + 1]
                
                # Joint axis in world coordinates
                joint_axis_world = (joint_transform[..., :3, :3] @ joint_axes[local_joint_idx].unsqueeze(-1)).squeeze(-1)
                
                # Position Jacobian: ω × (p_tool - p_joint)
                joint_position = joint_transform[..., :3, 3]
                position_jac = torch.cross(joint_axis_world, tool_world - joint_position, dim=-1)
                
                # Orientation Jacobian: just the joint axis
                orientation_jac = joint_axis_world
                
                # Stack position and orientation parts
                jac_col = torch.cat([position_jac, orientation_jac], dim=-1)
            else:
                # Zero column for joints not belonging to this finger
                zeros_col = torch.zeros(*batch_shape, 6, dtype=self.dtype, device=self.device)
                jac_col = zeros_col
            
            jac_columns.append(jac_col)
        
        # Stack all columns to form the Jacobian matrix
        jac = torch.stack(jac_columns, dim=-1)  # Shape: [*batch_shape, 6, 16]
        
        return jac

    def _create_rotation_matrix(self, axis: torch.Tensor, c: torch.Tensor, s: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Create rotation matrix for given axis and angle."""
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        
        # Check axis direction and create appropriate rotation matrix
        if torch.allclose(axis, torch.tensor([0., 0., 1.], dtype=self.dtype, device=self.device)):  # Z-axis
            # Build Z-axis rotation matrix
            R_row0 = torch.stack([c, -s, zeros, zeros], dim=-1)
            R_row1 = torch.stack([s, c, zeros, zeros], dim=-1)
            R_row2 = torch.stack([zeros, zeros, ones, zeros], dim=-1)
            R_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
            R = torch.stack([R_row0, R_row1, R_row2, R_row3], dim=-2)
        elif torch.allclose(axis, torch.tensor([0., 1., 0.], dtype=self.dtype, device=self.device)):  # Y-axis
            # Build Y-axis rotation matrix
            R_row0 = torch.stack([c, zeros, s, zeros], dim=-1)
            R_row1 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
            R_row2 = torch.stack([-s, zeros, c, zeros], dim=-1)
            R_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
            R = torch.stack([R_row0, R_row1, R_row2, R_row3], dim=-2)
        elif torch.allclose(axis, torch.tensor([-1., 0., 0.], dtype=self.dtype, device=self.device)):  # Negative X-axis
            # Build negative X-axis rotation matrix
            R_row0 = torch.stack([ones, zeros, zeros, zeros], dim=-1)
            R_row1 = torch.stack([zeros, c, s, zeros], dim=-1)
            R_row2 = torch.stack([zeros, -s, c, zeros], dim=-1)
            R_row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
            R = torch.stack([R_row0, R_row1, R_row2, R_row3], dim=-2)
        else:
            raise ValueError(f"Unsupported joint axis: {axis}")
        
        return R

    # Interface compatibility methods
    def get_frame_indices(self, *frame_names):
        """Get frame indices for given frame names."""
        return self.standard_chain.get_frame_indices(*frame_names)
    
    def get_joint_parameter_names(self, exclude_fixed=True):
        """Get joint parameter names."""
        return self.standard_chain.get_joint_parameter_names(exclude_fixed)
    
    def get_frame_names(self, exclude_fixed=True):
        """Get frame names."""
        return self.standard_chain.get_frame_names(exclude_fixed)
    
    def calc_jacobian(self, q: torch.Tensor, 
                      tool: Optional[torch.Tensor] = None,
                      link_indices: Optional[torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        """
        Interface-compatible Jacobian calculation.
        
        Args:
            q: Joint angles tensor of shape [batch_size, 16]
            tool: Optional tool transform 
            link_indices: Frame indices to compute Jacobian for
            **kwargs: Additional arguments for compatibility
        
        Returns:
            Jacobian tensor [batch_size, 6, 16]
        """
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=self.dtype, device=self.device)
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        
        # Handle tool locations (simplified for now)
        locations = None
        if tool is not None:
            if hasattr(tool, 'get_matrix'):
                # Extract translation from transform
                locations = tool.get_matrix()[:, :3, 3]
            elif torch.is_tensor(tool):
                locations = tool
        
        # Use optimized jacobian computation directly with link_indices
        results = self.jacobian(q, locations=locations, link_indices=link_indices)
        
        # Handle return types for compatibility
        if isinstance(results, dict):
            # Multiple frames returned as dictionary - return first one for compatibility
            return list(results.values())[0]
        else:
            # Single frame returned as tensor
            return results
        
    def calc_jacobian_and_hessian(self, th, tool=None, link_indices=None, analytic=False,
                                  tool_in_ee_frame=True):
        """
            Calculates robot jacobian and kinematic hessian in the base frame

            Returns:
                J: torch.tensor of shape (N, 6, DOF) representing robot jacobian
                H: torch.tensor of shape (N, 6, DOF, DOF) - kinematic Hessian. The kinematic hessian is the partial
                   derivative of the robot jacobian

        """
        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) <= 1:
            N = 1
            th = th.view(1, -1)
        else:
            N = th.shape[0]
        ndof = th.shape[1]

        # TODO not sure about converting hessian to analytic
        J = self.calc_jacobian(th, tool, link_indices, analytic, tool_in_ee_frame=tool_in_ee_frame)
        H = self.standard_chain.calc_hessian(J)
        return J, H
    
    def __getattr__(self, name):
        """Delegate unknown attributes to standard chain."""
        # Prevent infinite recursion during initialization
        if name == 'standard_chain' or not hasattr(self, 'standard_chain'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.standard_chain, name) 