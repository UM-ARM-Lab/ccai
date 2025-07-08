"""
Drop-in replacement for pytorch_kinematics with Allegro optimization

This module provides functions that automatically detect Allegro hands and
use optimized kinematics when possible, while falling back to standard
pytorch_kinematics for other robots.

Usage:
    # Replace this:
    # import pytorch_kinematics as pk
    # chain = pk.build_chain_from_urdf(urdf_data)
    
    # With this:
    import pytorch_kinematics as pk
    chain = pk.build_chain_from_urdf(urdf_data)
    
    # Everything else works exactly the same!
    result = chain.forward_kinematics(q, frame_indices)
"""

import torch
import pytorch_kinematics as _pk
from typing import Optional, Union, Dict, Any
from allegro_optimized_kinematics import OptimizedAllegroChain


def is_allegro_urdf(urdf_data: str) -> bool:
    """Check if URDF represents an Allegro hand."""
    allegro_indicators = [
        'allegro_hand',
        'hitosashi_finger',
        'naka_finger', 
        'kusuri_finger',
        'oya_finger'
    ]
    return any(indicator in urdf_data for indicator in allegro_indicators)


def build_chain_from_urdf(data: str, use_optimized_allegro: bool = True):
    """
    Build a Chain object from URDF data with automatic Allegro optimization.
    
    Args:
        data: URDF string data
        use_optimized_allegro: Whether to use optimized kinematics for Allegro hands
        
    Returns:
        Chain object (optimized for Allegro hands when detected)
    """
    if use_optimized_allegro and is_allegro_urdf(data):
        print("🚀 Detected Allegro hand - using optimized kinematics")
        try:
            return OptimizedAllegroChain(urdf_data=data)
        except Exception as e:
            print(f"⚠️ Failed to create optimized chain: {e}")
            print("Falling back to standard kinematics")
            return _pk.build_chain_from_urdf(data, use_optimized_allegro=False)
    else:
        return _pk.build_chain_from_urdf(data, use_optimized_allegro=False)


def build_serial_chain_from_urdf(data: str, end_link_name: str, 
                                root_link_name: str = "", 
                                use_optimized_allegro: bool = True):
    """
    Build a SerialChain object from URDF data.
    
    For Allegro hands, returns an optimized chain wrapped to behave like SerialChain.
    """
    if use_optimized_allegro and is_allegro_urdf(data):
        print("🚀 Detected Allegro hand - using optimized kinematics")
        base_chain = OptimizedAllegroChain(urdf_data=data)
        return AllegroSerialChainWrapper(base_chain, end_link_name, root_link_name)
    else:
        return _pk.build_serial_chain_from_urdf(data, end_link_name, root_link_name, 
                                               use_optimized_allegro=False)


class AllegroSerialChainWrapper:
    """
    Wrapper to make OptimizedAllegroChain behave like SerialChain.
    
    This enables drop-in replacement for code that expects SerialChain interface.
    """
    
    def __init__(self, base_chain: OptimizedAllegroChain, end_link_name: str, 
                 root_link_name: str = ""):
        self.base_chain = base_chain
        self.end_link_name = end_link_name
        self.root_link_name = root_link_name
        
        # Create standard serial chain for interface compatibility
        if hasattr(base_chain, 'standard_chain'):
            self._serial_chain = _pk.SerialChain(
                base_chain.standard_chain, end_link_name, root_link_name
            )
        
        # Delegate common attributes
        for attr in ['n_joints', 'frame_to_idx', 'idx_to_frame', 'dtype', 'device']:
            if hasattr(base_chain, attr):
                setattr(self, attr, getattr(base_chain, attr))
    
    def forward_kinematics(self, th: torch.Tensor, end_only: bool = True):
        """
        Forward kinematics with SerialChain interface.
        
        Args:
            th: Joint angles (can be partial for finger joints only)
            end_only: If True, return only end-effector transform
            
        Returns:
            Transform3d object (if end_only=True) or dict of transforms
        """
        # Handle partial joint angles for finger-specific chains
        if th.shape[-1] < self.base_chain.n_joints:
            # Assume this is finger-specific input, expand to full joint space
            full_th = torch.zeros(th.shape[0], self.base_chain.n_joints, 
                                dtype=th.dtype, device=th.device)
            
            # Determine which finger this chain represents
            finger_type = self._get_finger_type_from_end_link()
            if finger_type is not None:
                finger_data = self.base_chain.analysis['kinematic_chains'][finger_type]
                joint_indices = finger_data['joint_indices']
                full_th[:, joint_indices] = th
                th = full_th
        
        # Use optimized forward kinematics
        if end_only:
            result = self.base_chain.forward_kinematics(th, [self.end_link_name])
            return result[self.end_link_name]
        else:
            return self.base_chain.forward_kinematics(th)
    
    def _get_finger_type_from_end_link(self) -> Optional[str]:
        """Determine finger type from end link name."""
        for finger_name, data in self.base_chain.analysis['kinematic_chains'].items():
            if self.end_link_name == data['ee_link']:
                return finger_name
        return None
    
    def jacobian(self, th: torch.Tensor, locations=None, link_indices=None,
                locations_in_ee_frame: bool = True, ret_eef_pose: bool = False):
        """Compute Jacobian with SerialChain interface."""
        # Handle partial joint angles
        if th.shape[-1] < self.base_chain.n_joints:
            full_th = torch.zeros(th.shape[0], self.base_chain.n_joints,
                                dtype=th.dtype, device=th.device)
            finger_type = self._get_finger_type_from_end_link()
            if finger_type is not None:
                finger_data = self.base_chain.analysis['kinematic_chains'][finger_type]
                joint_indices = finger_data['joint_indices']
                full_th[:, joint_indices] = th
                th = full_th
        
        # Use base chain jacobian (can be optimized later)
        if link_indices is None:
            link_indices = torch.tensor([self.base_chain.frame_to_idx[self.end_link_name]], device=self.base_chain.device)
        
        jac_result = self.base_chain.jacobian(th, locations=locations, 
                                            link_indices=link_indices,
                                            locations_in_ee_frame=locations_in_ee_frame)
        
        # Handle both tensor and dictionary returns from jacobian
        if isinstance(jac_result, dict):
            # Multiple frames returned as dictionary
            result = jac_result[self.end_link_name]
        else:
            # Single frame returned as tensor
            result = jac_result
        
        # Extract relevant joints for this finger if using partial input
        if hasattr(self, '_serial_chain'):
            finger_type = self._get_finger_type_from_end_link()
            if finger_type is not None:
                finger_data = self.base_chain.analysis['kinematic_chains'][finger_type] 
                joint_indices = finger_data['joint_indices']
                result = result[:, :, joint_indices]
        
        if ret_eef_pose:
            fk_result = self.forward_kinematics(th, end_only=True)
            return result, fk_result.get_matrix()
        
        return result
    
    def to(self, dtype=None, device=None):
        """Move to device/dtype."""
        self.base_chain.to(dtype=dtype, device=device)
        if hasattr(self, '_serial_chain'):
            self._serial_chain.to(dtype=dtype, device=device)
        return self
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base chain."""
        return getattr(self.base_chain, name)


# Export pytorch_kinematics symbols for drop-in replacement
from pytorch_kinematics import Transform3d, Chain, SerialChain
import pytorch_kinematics.transforms as transforms

# Override the build functions with optimized versions  
__all__ = ['build_chain_from_urdf', 'build_serial_chain_from_urdf', 'Transform3d', 'Chain', 'SerialChain', 'transforms']


def create_optimized_chain(urdf_data: str, **kwargs):
    """
    Convenience function to create an optimized chain.
    
    Args:
        urdf_data: URDF string data
        **kwargs: Additional arguments
    
    Returns:
        Optimized chain for Allegro hands, standard chain otherwise
    """
    return build_chain_from_urdf(urdf_data, use_optimized_allegro=True, **kwargs)


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, standard_time: float, optimized_time: float, 
                 batch_size: int, num_trials: int):
        self.standard_time = standard_time
        self.optimized_time = optimized_time
        self.batch_size = batch_size
        self.num_trials = num_trials
        self.speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
    
    def __str__(self):
        return (f"Benchmark Results (batch_size={self.batch_size}, trials={self.num_trials}):\n"
                f"  Standard FK: {self.standard_time:.4f}s\n"
                f"  Optimized FK: {self.optimized_time:.4f}s\n"
                f"  Speedup: {self.speedup:.2f}x")


def benchmark_allegro_fk(urdf_data: str, batch_size: int = 100, num_trials: int = 100) -> BenchmarkResults:
    """
    Benchmark optimized vs standard forward kinematics for Allegro hand.
    
    Args:
        urdf_data: Allegro hand URDF data
        batch_size: Number of configurations per trial
        num_trials: Number of timing trials
        
    Returns:
        BenchmarkResults object with timing comparison
    """
    print(f"🏁 Benchmarking Allegro FK: {batch_size} configs × {num_trials} trials")
    
    # Create both chain types
    optimized_chain = OptimizedAllegroChain(urdf_data=urdf_data)
    standard_chain = _pk.build_chain_from_urdf(urdf_data, use_optimized_allegro=False)
    
    # Get end-effector links
    ee_links = [
        'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',  # index
        'allegro_hand_naka_finger_finger_1_aftc_base_link',      # middle
        'allegro_hand_kusuri_finger_finger_2_aftc_base_link',    # ring
        'allegro_hand_oya_finger_3_aftc_base_link'               # thumb
    ]
    
    ee_indices = [standard_chain.frame_to_idx[link] for link in ee_links]
    ee_indices = torch.tensor(ee_indices)
    
    # Generate test configurations
    test_configs = torch.randn(batch_size, 16) * 0.5  # Random joint angles
    
    # Warm up
    for _ in range(5):
        _ = optimized_chain.forward_kinematics(test_configs, ee_links)
        _ = standard_chain.forward_kinematics(test_configs, ee_indices)
    
    # Benchmark standard FK
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(num_trials):
        _ = standard_chain.forward_kinematics(test_configs, ee_indices)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    standard_time = time.perf_counter() - start_time
    
    # Benchmark optimized FK
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(num_trials):
        _ = optimized_chain.forward_kinematics(test_configs, ee_links)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    optimized_time = time.perf_counter() - start_time
    
    results = BenchmarkResults(standard_time, optimized_time, batch_size, num_trials)
    print(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("🤖 Allegro Optimized Wrapper Example")
    
    # Load Allegro URDF
    with open("allegro_hand_right.urdf", 'r') as f:
        urdf_data = f.read()
    
    # Create optimized chain (automatic detection)
    chain = build_chain_from_urdf(urdf_data)
    print(f"✓ Created chain with {chain.n_joints} joints")
    
    # Test forward kinematics
    q = torch.randn(10, 16) * 0.5
    result = chain.forward_kinematics(q)
    print(f"✓ FK result: {len(result)} end-effectors")
    
    # Benchmark performance
    benchmark_results = benchmark_allegro_fk(urdf_data, batch_size=50, num_trials=50)
    print(f"🏆 Achieved {benchmark_results.speedup:.1f}x speedup!") 