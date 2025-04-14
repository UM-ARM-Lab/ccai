import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List, Tuple
from .kinematics_jit import forward_kinematics_jit, jacobian_jit
from .metadata import JOINT_COUNTS, END_EFFECTORS, ROOT_LINKS

class HandKinematicsModel(nn.Module):
    """
    Optimized hand kinematics model using JIT compilation.
    
    This model provides a unified interface for forward kinematics and 
    Jacobian calculations for all fingers of the hand.
    """
    
    def __init__(self, use_jit: bool = True):
        """
        Initialize the hand kinematics model.
        
        Args:
            use_jit: Whether to use JIT compilation for faster calculations
        """
        super().__init__()
        self.use_jit = use_jit
        self.finger_names = list(JOINT_COUNTS.keys())
        self._init_model()
    
    def _init_model(self):
        """Initialize model components and precompile if using JIT."""
        if self.use_jit:
            self.precompile_all()
    
    def precompile_all(self):
        """Precompile JIT functions for all fingers and links."""
        from .kinematics_utils import precompile_functions
        link_names = {finger: [END_EFFECTORS[finger]] for finger in self.finger_names}
        precompile_functions(self.finger_names, link_names)
    
    def forward_kinematics(self, finger_name: str, q: torch.Tensor, 
                          device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute forward kinematics for a finger.
        
        Args:
            finger_name: Name of the finger
            q: Joint angles
            device: Device to run the computation on
            
        Returns:
            4x4 transformation matrix/matrices
        """
        if self.use_jit:
            return forward_kinematics_jit(finger_name, q, device)
        
        # Fall back to non-JIT implementation
        if finger_name == "index":
            from .index_fk import index_forward_kinematics
            return index_forward_kinematics(q, device)
        elif finger_name == "middle":
            from .middle_fk import middle_forward_kinematics
            return middle_forward_kinematics(q, device)
        elif finger_name == "ring":
            from .ring_fk import ring_forward_kinematics
            return ring_forward_kinematics(q, device)
        elif finger_name == "thumb":
            from .thumb_fk import thumb_forward_kinematics
            return thumb_forward_kinematics(q, device)
        
        raise ValueError(f"Unknown finger: {finger_name}")
    
    def jacobian(self, finger_name: str, q: torch.Tensor, 
                link_name: Optional[str] = None,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute Jacobian for a finger.
        
        Args:
            finger_name: Name of the finger
            q: Joint angles
            link_name: Name of the link (defaults to end effector)
            device: Device to run the computation on
            
        Returns:
            6xN Jacobian matrix/matrices
        """
        if link_name is None or link_name == '':
            link_name = END_EFFECTORS.get(finger_name)
            
        if self.use_jit:
            return jacobian_jit(finger_name, q, link_name, device)
        
        # Fall back to non-JIT implementation
        if finger_name == "index":
            from .index_jacobian import index_jacobian
            return index_jacobian(q, link_name, device)
        elif finger_name == "middle":
            from .middle_jacobian import middle_jacobian
            return middle_jacobian(q, link_name, device)
        elif finger_name == "ring":
            from .ring_jacobian import ring_jacobian
            return ring_jacobian(q, link_name, device)
        elif finger_name == "thumb":
            from .thumb_jacobian import thumb_jacobian
            return thumb_jacobian(q, link_name, device)
        
        raise ValueError(f"Unknown finger: {finger_name}")
    
    def forward(self, q_dict: Dict[str, torch.Tensor], 
               compute_jacobian: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing kinematics for all fingers.
        
        Args:
            q_dict: Dictionary mapping finger names to joint angles
            compute_jacobian: Whether to also compute Jacobians
            
        Returns:
            Dictionary with forward kinematics (and optionally Jacobians)
        """
        results = {}
        
        for finger, q in q_dict.items():
            # Compute forward kinematics
            fk = self.forward_kinematics(finger, q)
            results[f"{finger}_fk"] = fk
            
            # Optionally compute Jacobians
            if compute_jacobian:
                jac = self.jacobian(finger, q)
                results[f"{finger}_jacobian"] = jac
        
        return results
