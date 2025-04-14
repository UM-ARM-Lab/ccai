import torch
from typing import Tuple, Dict, Optional
from functools import lru_cache

# Dictionary to store compiled functions
_COMPILED_FK_FUNCS = {}
_COMPILED_JAC_FUNCS = {}

@torch.jit.script
def precompute_trig_terms(q: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Precompute trigonometric terms for faster kinematics calculations.
    
    Args:
        q: Joint angles tensor of shape [batch_size, n_joints]
        
    Returns:
        Dictionary with precomputed sin/cos values and common terms
    """
    batch_size, n_joints = q.shape
    
    # Precompute sin and cos for each joint
    sin_q = torch.sin(q)
    cos_q = torch.cos(q)
    
    # Common angle sums for phalanges (for fingers with 4 joints)
    if n_joints >= 3:
        sin_q12 = torch.sin(q[:, 1] + q[:, 2])
        cos_q12 = torch.cos(q[:, 1] + q[:, 2])
    else:
        sin_q12 = torch.zeros(batch_size, device=q.device, dtype=q.dtype)
        cos_q12 = torch.zeros(batch_size, device=q.device, dtype=q.dtype)
        
    if n_joints >= 4:
        sin_q123 = torch.sin(q[:, 1] + q[:, 2] + q[:, 3])
        cos_q123 = torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
    else:
        sin_q123 = torch.zeros(batch_size, device=q.device, dtype=q.dtype)
        cos_q123 = torch.zeros(batch_size, device=q.device, dtype=q.dtype)
    
    # Return all precomputed values
    return {
        "sin_q": sin_q,
        "cos_q": cos_q,
        "sin_q12": sin_q12, 
        "cos_q12": cos_q12,
        "sin_q123": sin_q123,
        "cos_q123": cos_q123
    }

def get_compiled_fk_func(finger_name: str):
    """Get or create JIT-compiled forward kinematics function for a finger."""
    if finger_name in _COMPILED_FK_FUNCS:
        return _COMPILED_FK_FUNCS[finger_name]
    
    if finger_name == "thumb":
        @torch.jit.script
        def thumb_fk_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized thumb FK implementation using the exact equations from thumb_fk.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Create transformation matrix
            T = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)
            
            # Shorthand variable names for better readability
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s123, c123 = sin_q123, cos_q123
            
            # Constants from original implementation
            const1 = 6.12323399573677e-17
            const2 = 6.0999332417281e-17
            const3 = 5.33675006916149e-18
            
            # First row elements
            T[:, 0, 0] = -const1*((const2*s0 + c0)*s1 - const3*c1)*s2*s3 - const1*(s0 - const2*c0)*c2*s3 + \
                         const1*((const2*s0 + c0)*s1 - const3*c1)*c2*c3 + const1*(s0 - const2*c0)*s2*c3 + \
                         (const2*s0 + c0)*c1 + const3*s1
            
            T[:, 0, 1] = const1*((const2*s0 + c0)*s1 - const3*c1)*s2*s3 + \
                         (((const2*s0 + c0)*s1 - const3*c1)*s2 + (s0 - const2*c0)*c2)*c3 + \
                         (((const2*s0 + c0)*s1 - const3*c1)*c2 - (s0 - const2*c0)*s2)*s3
            
            T[:, 0, 2] = -((const2*s0 + c0)*s1 - const3*c1)*s2*s3 + \
                         const1*((const2*s0 + c0)*s1 - const3*c1)*s2*c3 + \
                         const1*((const2*s0 + c0)*s1 - const3*c1)*c2*s3 + \
                         (((const2*s0 + c0)*s1 - const3*c1)*c2 - (s0 - const2*c0)*s2)*c3
            
            T[:, 0, 3] = 0.013*((const2*s0 + c0)*s1 - const3*c1)*s2*s3 + \
                         0.0615*((const2*s0 + c0)*s1 - const3*c1)*s2*c3 + \
                         0.0615*((const2*s0 + c0)*s1 - const3*c1)*c2*s3 + \
                         -0.013*((const2*s0 + c0)*s1 - const3*c1)*c2*c3 + \
                         0.0514*((const2*s0 + c0)*s1 - const3*c1)*s2 + \
                         0.0514*(s0 - const2*c0)*c2 + \
                         0.0576*s0 + 0.005*c0 - 0.0182
            
            # Second row elements
            T[:, 1, 0] = -const1*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2*s3 - \
                         const1*(const1*s0 + 0.996194698091746*c0)*c2*s3 + \
                         const1*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*c2*c3 + \
                         const1*(const1*s0 + 0.996194698091746*c0)*s2*c3 + \
                         ((-0.996194698091746)*s0 + const1*c0)*c1 - 0.0871557427476582*s1
            
            T[:, 1, 1] = const1*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2*s3 + \
                         ((((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2 + \
                         (const1*s0 + 0.996194698091746*c0)*c2)*c3 + \
                         ((((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*c2 - \
                         (const1*s0 + 0.996194698091746*c0)*s2)*s3
            
            T[:, 1, 2] = -(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2*s3 + \
                         const1*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2*c3 + \
                         const1*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*c2*s3 + \
                         ((((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*c2 - \
                         (const1*s0 + 0.996194698091746*c0)*s2)*c3
            
            T[:, 1, 3] = 0.013*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2*s3 + \
                         0.0615*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2*c3 + \
                         0.0615*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*c2*s3 - \
                         0.013*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*c2*c3 + \
                         0.0514*(((-0.996194698091746)*s0 + const1*c0)*s1 + 0.0871557427476582*c1)*s2 + \
                         0.0514*(const1*s0 + 0.996194698091746*c0)*c2 - \
                         0.00498097349045872*s0 + 0.0573808146100845*c0 + 0.0169797949458132
            
            # Third row elements
            T[:, 2, 0] = -const1*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*s3 + \
                         const1*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*c3 + \
                         0.0871557427476582*s0*c1 - 0.996194698091746*s1
            
            T[:, 2, 1] = const1*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*s3 + \
                         ((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*c3 + \
                         ((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*s3 - \
                         const1*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*c3
            
            T[:, 2, 2] = -((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*s3 + \
                         const1*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*c3 + \
                         const1*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*s3 + \
                         ((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*c3
            
            T[:, 2, 3] = 0.013*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*s3 + \
                         0.0615*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 - \
                         0.0871557427476582*c0*c2)*c3 + \
                         0.0615*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*s3 - \
                         0.013*((0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0871557427476582*s2*c0)*c3 + \
                         0.0514*(0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         0.000435778713738291*s0 - 0.00502017078226511*c0 - \
                         0.00447980517722963*c0*c2 - 0.0728842568484771
            
            # Row 4 (unchanged)
            T[:, 3, 0] = 0.0
            T[:, 3, 1] = 0.0
            T[:, 3, 2] = 0.0
            T[:, 3, 3] = 1.0
            
            return T
        
        _COMPILED_FK_FUNCS["thumb"] = thumb_fk_compiled
        return thumb_fk_compiled
        
    elif finger_name == "index":
        @torch.jit.script
        def index_fk_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized index finger FK implementation using the exact equations from index_fk.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Create transformation matrix
            T = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)
            
            # Extract individual sin and cos values for clarity
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s12, c12 = sin_q12, cos_q12
            s123, c123 = sin_q123, cos_q123
            
            # Constants
            const = 6.12323399573677e-17
            
            # First row elements
            T[:, 0, 0] = const*(-s1*s2*c0 + c0*c1*c2)*c3 - const*(s1*c0*c2 + s2*c0*c1)*s3 - s0
            
            T[:, 0, 1] = (-s1*s2*c0 + c0*c1*c2)*s3 - const*(-s1*s2*c0 + c0*c1*c2)*c3 + \
                         const*(s1*c0*c2 + s2*c0*c1)*s3 + (s1*c0*c2 + s2*c0*c1)*c3
            
            T[:, 0, 2] = const*(-s1*s2*c0 + c0*c1*c2)*s3 + (-s1*s2*c0 + c0*c1*c2)*c3 - \
                         (s1*c0*c2 + s2*c0*c1)*s3 + const*(s1*c0*c2 + s2*c0*c1)*c3 + const*s0
            
            T[:, 0, 3] = 0.046*(-s1*s2*c0 + c0*c1*c2)*s3 - 0.013*(-s1*s2*c0 + c0*c1*c2)*c3 + \
                         0.013*(s1*c0*c2 + s2*c0*c1)*s3 + 0.046*(s1*c0*c2 + s2*c0*c1)*c3 + \
                         0.0384*s1*c0*c2 + 0.054*s1*c0 + 0.0384*s2*c0*c1
            
            # Second row elements
            T[:, 1, 0] = const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 - \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + 0.996194698091746*c0
            
            T[:, 1, 1] = (-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 - \
                         const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 + \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                         ((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3
            
            T[:, 1, 2] = const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 + \
                         (-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 - \
                         ((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3 - const*c0
            
            T[:, 1, 3] = 0.046*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 - \
                          0.013*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 + \
                          0.013*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                          0.046*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3 + \
                          0.0384*(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          0.0384*(0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2 + \
                          0.0537945136969543*s0*s1 + 0.00470641010837354*c1 + 0.0452779771520522
            
            # Third row elements
            T[:, 2, 0] = const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 - \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 - 0.0871557427476582*c0
            
            T[:, 2, 1] = (-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 - \
                         const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 + \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         ((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3
            
            T[:, 2, 2] = const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 + \
                         (-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 - \
                         ((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3 + const*c0
            
            T[:, 2, 3] = 0.046*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 - \
                         0.013*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 + \
                         0.013*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         0.046*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3 + \
                         0.0384*(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0384*(-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2 - \
                         0.00470641010837354*s0*s1 + 0.0537945136969543*c1 + 0.0187803718410716
            
            # Row 4 (unchanged)
            T[:, 3, 0] = 0.0
            T[:, 3, 1] = 0.0
            T[:, 3, 2] = 0.0
            T[:, 3, 3] = 1.0
            
            return T
        
        _COMPILED_FK_FUNCS["index"] = index_fk_compiled
        return index_fk_compiled
        
    elif finger_name == "middle":
        @torch.jit.script
        def middle_fk_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized middle finger FK implementation using the exact equations from middle_fk.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Create transformation matrix
            T = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)
            
            # Extract individual sin and cos values for clarity
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s12, c12 = sin_q12, cos_q12
            s123, c123 = sin_q123, cos_q123
            
            # Constant
            const = 6.12323399573677e-17
            
            # For efficiency, compute common terms once
            term1 = -s1*s2*c0 + c0*c1*c2
            term2 = s1*c0*c2 + s2*c0*c1
            
            T[:, 0, 0] = const*term1*c3 - const*term2*s3 - s0
            
            T[:, 0, 3] = 0.046*term1*s3 - 0.013*term1*c3 + \
                        0.013*term2*s3 + 0.046*term2*c3 + \
                        0.0384*s1*c0*c2 + 0.054*s1*c0 + \
                        0.0384*s2*c0*c1
            
            T[:, 1, 0] = const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 - \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + 0.996194698091746*c0
            
            T[:, 1, 1] = (-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 - \
                         const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 + \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                         ((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3
            
            T[:, 1, 2] = const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 + \
                         (-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 - \
                         ((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3 - const*c0
            
            T[:, 1, 3] = 0.046*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 - \
                          0.013*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 + \
                          0.013*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                          0.046*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3 + \
                          0.0384*(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          0.0384*(0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2 + \
                          0.0537945136969543*s0*s1 + 0.00470641010837354*c1 + 0.0452779771520522
            
            T[:, 2, 0] = const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 - \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 - 0.0871557427476582*c0
            
            T[:, 2, 1] = (-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 - \
                         const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 + \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         ((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3
            
            T[:, 2, 2] = const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 + \
                         (-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 - \
                         ((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3 + const*c0
            
            T[:, 2, 3] = 0.046*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 - \
                         0.013*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 + \
                         0.013*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         0.046*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3 + \
                         0.0384*(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0384*(-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2 - \
                         0.00470641010837354*s0*s1 + 0.0537945136969543*c1 + 0.0187803718410716
            
            # Row 4 (unchanged)
            T[:, 3, 0] = 0.0
            T[:, 3, 1] = 0.0
            T[:, 3, 2] = 0.0
            T[:, 3, 3] = 1.0
            
            return T
        
        _COMPILED_FK_FUNCS["middle"] = middle_fk_compiled
        return middle_fk_compiled
        
    elif finger_name == "ring":
        @torch.jit.script
        def ring_fk_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized ring finger FK implementation using the exact equations from ring_fk.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Create transformation matrix
            T = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)
            
            # Extract individual sin and cos values for clarity
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s12, c12 = sin_q12, cos_q12
            s123, c123 = sin_q123, cos_q123
            
            # Constants
            const = 6.12323399573677e-17
            
            # Common terms
            term1 = -s1*s2*c0 + c0*c1*c2  
            term2 = s1*c0*c2 + s2*c0*c1
            
            # Matrix elements from ring_fk.py
            T[:, 0, 0] = const*term1*c3 - const*term2*s3 - s0
            
            T[:, 0, 3] = 0.046*term1*s3 - 0.013*term1*c3 + \
                        0.013*term2*s3 + 0.046*term2*c3 + \
                        0.0384*s1*c0*c2 + 0.054*s1*c0 + \
                        0.0384*s2*c0*c1
            
            T[:, 1, 0] = const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 - \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + 0.996194698091746*c0
            
            T[:, 1, 1] = (-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 - \
                         const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 + \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                         ((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3
            
            T[:, 1, 2] = const*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 + \
                         (-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 - \
                         ((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                         const*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                         (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3 - const*c0
            
            T[:, 1, 3] = 0.046*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*s3 - \
                          0.013*(-(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*s2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*c2)*c3 + \
                          0.013*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*s3 + \
                          0.046*((0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          (0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2)*c3 + \
                          0.0384*(0.996194698091746*s0*s1 + 0.0871557427476582*c1)*c2 + \
                          0.0384*(0.996194698091746*s0*c1 - 0.0871557427476582*s1)*s2 + \
                          0.0537945136969543*s0*s1 + 0.00470641010837354*c1 + 0.0452779771520522
            
            T[:, 2, 0] = const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 - \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 - 0.0871557427476582*c0
            
            T[:, 2, 1] = (-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 - \
                         const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 + \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         ((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3
            
            T[:, 2, 2] = const*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 + \
                         (-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 - \
                         ((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         const*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3 + const*c0
            
            T[:, 2, 3] = 0.046*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*s3 - \
                         0.013*(-(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*s2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*c2)*c3 + \
                         0.013*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*s3 + \
                         0.046*((-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         (-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2)*c3 + \
                         0.0384*(-0.0871557427476582*s0*s1 + 0.996194698091746*c1)*c2 + \
                         0.0384*(-0.0871557427476582*s0*c1 - 0.996194698091746*s1)*s2 - \
                         0.00470641010837354*s0*s1 + 0.0537945136969543*c1 + 0.0187803718410716
            
            # Row 4 (unchanged)
            T[:, 3, 0] = 0.0
            T[:, 3, 1] = 0.0
            T[:, 3, 2] = 0.0
            T[:, 3, 3] = 1.0
            
            return T
        
        _COMPILED_FK_FUNCS["ring"] = ring_fk_compiled
        return ring_fk_compiled
    
    raise ValueError(f"Unknown finger: {finger_name}")

def get_compiled_jacobian_func(finger_name: str, link_name: str):
    """Get or create JIT-compiled Jacobian function for a specific finger and link."""
    key = f"{finger_name}_{link_name}"
    if key in _COMPILED_JAC_FUNCS:
        return _COMPILED_JAC_FUNCS[key]
    
    # Create specialized compiled function for each finger/link combination
    if finger_name == "thumb" and link_name == "oya_ee":
        @torch.jit.script
        def thumb_ee_jacobian_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized thumb Jacobian implementation using the exact equations from thumb_jacobian.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Extract individual sin and cos for clarity
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s123, c123 = sin_q123, cos_q123
            
            # Initialize Jacobian matrix
            J = torch.zeros(batch_size, 6, 4, dtype=dtype, device=device)
            
            # Jacobian elements from thumb_jacobian.py for oya_ee
            J[:, 0, 0] = -0.054*s0*s1 - 0.046*s0*s123 + 0.013*s0*c123 - 1.12481983699639e-34*c0
            J[:, 0, 1] = (0.013*s123 + 0.054*c1 + 0.046*c123)*c0
            J[:, 0, 2] = (0.013*s123 + 0.046*c123)*c0
            J[:, 0, 3] = (0.013*s123 + 0.046*c123)*c0
            
            # Rest of the Jacobian elements for thumb_ee from thumb_jacobian.py
            J[:, 1, 0] = -1.12481983699639e-34*s0 + 0.054*s1*c0 + 0.046*s123*c0 - 0.013*c0*c123
            J[:, 1, 1] = (0.013*s123 + 0.054*c1 + 0.046*c123)*s0
            J[:, 1, 2] = (0.013*s123 + 0.046*c123)*s0
            J[:, 1, 3] = (0.013*s123 + 0.046*c123)*s0
            
            J[:, 2, 1] = -0.054*s1 - 0.046*s123 + 0.013*c123
            J[:, 2, 2] = -0.046*s123 + 0.013*c123
            J[:, 2, 3] = -0.046*s123 + 0.013*c123
            
            J[:, 3, 1] = -s0
            J[:, 3, 2] = -s0
            J[:, 3, 3] = -s0
            
            J[:, 4, 0] = 0.0871557427476582
            J[:, 4, 1] = 0.996194698091746*c0
            J[:, 4, 2] = 0.996194698091746*c0
            J[:, 4, 3] = 0.996194698091746*c0
            
            J[:, 5, 0] = 0.996194698091746
            J[:, 5, 1] = 0.0871557427476582*c0
            J[:, 5, 2] = 0.0871557427476582*c0
            J[:, 5, 3] = 0.0871557427476582*c0
            
            return J
        
        _COMPILED_JAC_FUNCS[key] = thumb_ee_jacobian_compiled
        return thumb_ee_jacobian_compiled
    
    elif finger_name == "index" and link_name == "hitosashi_ee":
        @torch.jit.script
        def index_ee_jacobian_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized index Jacobian implementation using the exact equations from index_jacobian.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Extract individual sin and cos
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s12, c12 = sin_q12, cos_q12
            s123, c123 = sin_q123, cos_q123
            
            # Initialize Jacobian matrix
            J = torch.zeros(batch_size, 6, 4, dtype=dtype, device=device)
            
            # Jacobian elements from index_jacobian.py for hitosashi_ee
            J[:, 0, 0] = -0.054*s0*s1 - 0.0384*s0*s12 - 0.046*s0*s123 + 0.013*s0*c123 - 1.12481983699639e-34*c0
            J[:, 0, 1] = (0.013*s123 + 0.054*c1 + 0.0384*c12 + 0.046*c123)*c0
            J[:, 0, 2] = (0.013*s123 + 0.0384*c12 + 0.046*c123)*c0
            J[:, 0, 3] = (0.013*s123 + 0.046*c123)*c0
            
            # Rest of the elements from index_jacobian.py
            J[:, 1, 0] = -1.12481983699639e-34*s0 + 0.054*s1*c0 + 0.0384*s12*c0 + 0.046*s123*c0 - 0.013*c0*c123
            J[:, 1, 1] = (0.013*s123 + 0.054*c1 + 0.0384*c12 + 0.046*c123)*s0
            J[:, 1, 2] = (0.013*s123 + 0.0384*c12 + 0.046*c123)*s0
            J[:, 1, 3] = (0.013*s123 + 0.046*c123)*s0
            
            J[:, 2, 1] = -0.054*s1 - 0.0384*s12 - 0.046*s123 + 0.013*c123
            J[:, 2, 2] = -0.0384*s12 - 0.046*s123 + 0.013*c123
            J[:, 2, 3] = -0.046*s123 + 0.013*c123
            
            J[:, 3, 1] = -s0
            J[:, 3, 2] = -s0
            J[:, 3, 3] = -s0
            
            J[:, 4, 0] = 0.0871557427476582
            J[:, 4, 1] = 0.996194698091746*c0
            J[:, 4, 2] = 0.996194698091746*c0
            J[:, 4, 3] = 0.996194698091746*c0
            
            J[:, 5, 0] = 0.996194698091746
            J[:, 5, 1] = 0.0871557427476582*c0
            J[:, 5, 2] = 0.0871557427476582*c0
            J[:, 5, 3] = 0.0871557427476582*c0
            
            return J
            
        _COMPILED_JAC_FUNCS[key] = index_ee_jacobian_compiled
        return index_ee_jacobian_compiled
    
    elif finger_name == "middle" and link_name == "naka_ee":
        @torch.jit.script
        def middle_ee_jacobian_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized middle Jacobian implementation using the exact equations from middle_jacobian.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Extract individual sin and cos
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s12, c12 = sin_q12, cos_q12
            s123, c123 = sin_q123, cos_q123
            
            # Initialize Jacobian matrix
            J = torch.zeros(batch_size, 6, 4, dtype=dtype, device=device)
            
            # Middle Jacobian elements from middle_jacobian.py
            J[:, 0, 0] = -0.054*s0*s1 - 0.0384*s0*s12 - 0.046*s0*s123 + 0.013*s0*c123 - 1.12481983699639e-34*c0
            J[:, 0, 1] = (0.013*s123 + 0.054*c1 + 0.0384*c12 + 0.046*c123)*c0
            J[:, 0, 2] = (0.013*s123 + 0.0384*c12 + 0.046*c123)*c0
            J[:, 0, 3] = (0.013*s123 + 0.046*c123)*c0
            
            # Rest of the elements from middle_jacobian.py
            J[:, 1, 0] = -1.12481983699639e-34*s0 + 0.054*s1*c0 + 0.0384*s12*c0 + 0.046*s123*c0 - 0.013*c0*c123
            J[:, 1, 1] = (0.013*s123 + 0.054*c1 + 0.0384*c12 + 0.046*c123)*s0
            J[:, 1, 2] = (0.013*s123 + 0.0384*c12 + 0.046*c123)*s0
            J[:, 1, 3] = (0.013*s123 + 0.046*c123)*s0
            
            J[:, 2, 1] = -0.054*s1 - 0.0384*s12 - 0.046*s123 + 0.013*c123
            J[:, 2, 2] = -0.0384*s12 - 0.046*s123 + 0.013*c123
            J[:, 2, 3] = -0.046*s123 + 0.013*c123
            
            J[:, 3, 1] = -s0
            J[:, 3, 2] = -s0
            J[:, 3, 3] = -s0
            
            J[:, 4, 0] = 0.0871557427476582
            J[:, 4, 1] = 0.996194698091746*c0
            J[:, 4, 2] = 0.996194698091746*c0
            J[:, 4, 3] = 0.996194698091746*c0
            
            J[:, 5, 0] = 0.996194698091746
            J[:, 5, 1] = 0.0871557427476582*c0
            J[:, 5, 2] = 0.0871557427476582*c0
            J[:, 5, 3] = 0.0871557427476582*c0
            
            return J
            
        _COMPILED_JAC_FUNCS[key] = middle_ee_jacobian_compiled
        return middle_ee_jacobian_compiled
    
    elif finger_name == "ring" and link_name == "kusuri_ee":
        @torch.jit.script
        def ring_ee_jacobian_compiled(q: torch.Tensor) -> torch.Tensor:
            """Optimized ring Jacobian implementation using the exact equations from ring_jacobian.py"""
            batch_size = q.shape[0]
            device = q.device
            dtype = q.dtype
            
            # Precompute trig terms
            trig = precompute_trig_terms(q)
            sin_q, cos_q = trig["sin_q"], trig["cos_q"]
            sin_q12, cos_q12 = trig["sin_q12"], trig["cos_q12"]
            sin_q123, cos_q123 = trig["sin_q123"], trig["cos_q123"]
            
            # Extract for clarity
            s0, c0 = sin_q[:, 0], cos_q[:, 0]
            s1, c1 = sin_q[:, 1], cos_q[:, 1]
            s2, c2 = sin_q[:, 2], cos_q[:, 2]
            s3, c3 = sin_q[:, 3], cos_q[:, 3]
            s12, c12 = sin_q12, cos_q12
            s123, c123 = sin_q123, cos_q123
            
            # Initialize Jacobian matrix
            J = torch.zeros(batch_size, 6, 4, dtype=dtype, device=device)
            
            # Ring Jacobian elements from ring_jacobian.py
            J[:, 0, 0] = -0.054*s0*s1 - 0.0384*s0*s12 - 0.046*s0*s123 + 0.013*s0*c123 - 1.12481983699639e-34*c0
            J[:, 0, 1] = (0.013*s123 + 0.054*c1 + 0.0384*c12 + 0.046*c123)*c0
            J[:, 0, 2] = (0.013*s123 + 0.0384*c12 + 0.046*c123)*c0
            J[:, 0, 3] = (0.013*s123 + 0.046*c123)*c0
            
            # Rest of the elements from ring_jacobian.py
            J[:, 1, 0] = -1.12481983699639e-34*s0 + 0.054*s1*c0 + 0.0384*s12*c0 + 0.046*s123*c0 - 0.013*c0*c123
            J[:, 1, 1] = (0.013*s123 + 0.054*c1 + 0.0384*c12 + 0.046*c123)*s0
            J[:, 1, 2] = (0.013*s123 + 0.0384*c12 + 0.046*c123)*s0
            J[:, 1, 3] = (0.013*s123 + 0.046*c123)*s0
            
            J[:, 2, 1] = -0.054*s1 - 0.0384*s12 - 0.046*s123 + 0.013*c123
            J[:, 2, 2] = -0.0384*s12 - 0.046*s123 + 0.013*c123
            J[:, 2, 3] = -0.046*s123 + 0.013*c123
            
            J[:, 3, 1] = -s0
            J[:, 3, 2] = -s0
            J[:, 3, 3] = -s0
            
            J[:, 4, 0] = 0.0871557427476582
            J[:, 4, 1] = 0.996194698091746*c0
            J[:, 4, 2] = 0.996194698091746*c0
            J[:, 4, 3] = 0.996194698091746*c0
            
            J[:, 5, 0] = 0.996194698091746
            J[:, 5, 1] = 0.0871557427476582*c0
            J[:, 5, 2] = 0.0871557427476582*c0
            J[:, 5, 3] = 0.0871557427476582*c0
            
            return J
            
        _COMPILED_JAC_FUNCS[key] = ring_ee_jacobian_compiled
        return ring_ee_jacobian_compiled
    
    # Add implementations for other links as needed
    
    raise ValueError(f"Unknown finger/link combination: {finger_name}/{link_name}")

# Functions to use the compiled implementations
def forward_kinematics_jit(finger_name: str, q: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Fast forward kinematics using JIT compilation.
    
    Args:
        finger_name: Name of the finger ("thumb", "index", "middle", "ring")
        q: Joint angles in radians, shape [batch_size, n_joints] or [n_joints]
        device: Optional device for computation
        
    Returns:
        Transformation matrices
    """
    # Input handling and validation
    if device is None:
        device = q.device if torch.is_tensor(q) else torch.device('cpu')
    
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=torch.float32, device=device)
    
    unbatched = False
    if q.dim() == 1:
        unbatched = True
        q = q.unsqueeze(0)
    
    # Get compiled function and compute result
    fk_func = get_compiled_fk_func(finger_name)
    result = fk_func(q)
    
    return result[0] if unbatched else result

def jacobian_jit(finger_name: str, q: torch.Tensor, link_name: Optional[str] = None,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Fast Jacobian calculation using JIT compilation.
    
    Args:
        finger_name: Name of the finger ("thumb", "index", "middle", "ring")
        q: Joint angles in radians
        link_name: Name of the link (defaults to end effector if None)
        device: Optional device for computation
        
    Returns:
        Jacobian matrix
    """
    # Handle device and input format
    if device is None:
        device = q.device if torch.is_tensor(q) else torch.device('cpu')
    
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=torch.float32, device=device)
    
    from .metadata import END_EFFECTORS
    
    # Default to end effector
    if link_name is None or link_name == '':
        link_name = END_EFFECTORS.get(finger_name, f"{finger_name}_ee")
    
    unbatched = False
    if q.dim() == 1:
        unbatched = True
        q = q.unsqueeze(0)
    
    # Get compiled function and compute result
    jac_func = get_compiled_jacobian_func(finger_name, link_name)
    result = jac_func(q)
    
    return result[0] if unbatched else result
