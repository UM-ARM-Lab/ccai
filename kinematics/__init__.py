# Auto-generated kinematics package
from .index_fk import index_forward_kinematics
from .index_jacobian import index_jacobian
from .middle_fk import middle_forward_kinematics
from .middle_jacobian import middle_jacobian
from .ring_fk import ring_forward_kinematics
from .ring_jacobian import ring_jacobian
from .thumb_fk import thumb_forward_kinematics
from .thumb_jacobian import thumb_jacobian
from .metadata import JOINT_COUNTS, END_EFFECTORS, ROOT_LINKS

# Import optimized versions
from .kinematics_jit import forward_kinematics_jit, jacobian_jit
from .kinematics_utils import timeit, precompile_functions

# Create a convenient mapping for fast lookup
FINGER_FK_FUNCS = {
    'index': index_forward_kinematics,
    'middle': middle_forward_kinematics,
    'ring': ring_forward_kinematics,
    'thumb': thumb_forward_kinematics
}

FINGER_JAC_FUNCS = {
    'index': index_jacobian,
    'middle': middle_jacobian,
    'ring': ring_jacobian,
    'thumb': thumb_jacobian
}

# Fast implementations with JIT compilation
def forward_kinematics_fast(finger_name: str, q, device=None):
    """Unified fast forward kinematics function"""
    try:
        # Try using JIT-compiled version
        return forward_kinematics_jit(finger_name, q, device)
    except (ValueError, NotImplementedError):
        # Fall back to original implementation
        if finger_name in FINGER_FK_FUNCS:
            return FINGER_FK_FUNCS[finger_name](q, device)
        raise ValueError(f"Unknown finger: {forward_kinematics_fast}")

def jacobian_fast(finger_name: str, q, link_name=None, device=None):
    """Unified fast Jacobian calculation function"""
    try:
        # Try using JIT-compiled version
        return jacobian_jit(finger_name, q, link_name, device)
    except (ValueError, NotImplementedError):
        # Fall back to original implementation
        if finger_name in FINGER_JAC_FUNCS:
            return FINGER_JAC_FUNCS[finger_name](q, link_name, device)
        raise ValueError(f"Unknown finger: {jacobian_fast}")

__all__ = [
    'index_forward_kinematics', 'index_jacobian',
    'middle_forward_kinematics', 'middle_jacobian',
    'ring_forward_kinematics', 'ring_jacobian',
    'thumb_forward_kinematics', 'thumb_jacobian',
    'forward_kinematics_fast', 'jacobian_fast',
    'JOINT_COUNTS', 'END_EFFECTORS', 'ROOT_LINKS'
]
