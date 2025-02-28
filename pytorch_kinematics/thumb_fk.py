import torch
from typing import Union, Tuple, List, Optional
from functools import lru_cache

def thumb_forward_kinematics_impl(q: torch.Tensor) -> torch.Tensor:
    """
    Forward kinematics implementation for thumb finger.
    Takes joint angles and returns 4x4 transformation matrices.

    Args:
        q: Joint angles tensor of shape [batch_size, 4]

    Returns:
        Transformation matrices of shape [batch_size, 4, 4]
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype

    # Create transformation matrix
    T = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)

    # Set transformation matrix elements directly
    T[:, 0, 0] = -6.12323399573677e-17*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 1.0*(6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) + 5.33675006916149e-18*torch.sin(q[:, 1])
    T[:, 0, 1] = 6.12323399573677e-17*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 1.0*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 3.74939945665464e-33*(6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) + 2.00096078096157e-50*torch.sin(q[:, 1])
    T[:, 0, 2] = -1.0*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*(6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) - 3.26781694502402e-34*torch.sin(q[:, 1])
    T[:, 0, 3] = 0.013*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 0.0615*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.0615*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) - 0.013*(((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 0.0514*((6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) - 5.33675006916149e-18*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + 1.12481983699639e-34*(6.0999332417281e-17*torch.sin(q[:, 0]) + 1.0*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) + 0.0514*(1.0*torch.sin(q[:, 0]) - 6.0999332417281e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 2]) + 0.0576*torch.sin(q[:, 0]) + 6.00288234288472e-52*torch.sin(q[:, 1]) + 0.005*torch.cos(q[:, 0]) - 0.0182
    T[:, 1, 0] = -6.12323399573677e-17*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 1.0*(-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1])
    T[:, 1, 1] = 6.12323399573677e-17*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 1.0*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 3.74939945665464e-33*(-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) - 3.26781694502402e-34*torch.sin(q[:, 1])
    T[:, 1, 2] = -1.0*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*(-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) + 5.33675006916149e-18*torch.sin(q[:, 1])
    T[:, 1, 3] = 0.013*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 0.0615*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.0615*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) - 0.013*(((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) - (6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 0.0514*((-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + 1.12481983699639e-34*(-0.996194698091746*torch.sin(q[:, 0]) + 6.12323399573677e-17*torch.cos(q[:, 0]))*torch.cos(q[:, 1]) + 0.0514*(6.12323399573677e-17*torch.sin(q[:, 0]) + 0.996194698091746*torch.cos(q[:, 0]))*torch.cos(q[:, 2]) - 0.00498097349045872*torch.sin(q[:, 0]) - 9.80345083507205e-36*torch.sin(q[:, 1]) + 0.0573808146100845*torch.cos(q[:, 0]) + 0.0169797949458132
    T[:, 2, 0] = -6.12323399573677e-17*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.cos(q[:, 3]) + 0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1])
    T[:, 2, 1] = 6.12323399573677e-17*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 1.0*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.cos(q[:, 3]) + 3.26781694502402e-34*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 3.73513185974743e-33*torch.sin(q[:, 1])
    T[:, 2, 2] = -1.0*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.sin(q[:, 3]) + 1.0*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.cos(q[:, 3]) - 5.33675006916149e-18*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 6.0999332417281e-17*torch.sin(q[:, 1])
    T[:, 2, 3] = 0.013*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 0.0615*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) - 0.0871557427476582*torch.cos(q[:, 0])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.0615*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.sin(q[:, 3]) - 0.013*((0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0871557427476582*torch.sin(q[:, 2])*torch.cos(q[:, 0]))*torch.cos(q[:, 3]) + 0.0514*(0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + 9.80345083507205e-36*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.000435778713738291*torch.sin(q[:, 0]) - 1.12053955792423e-34*torch.sin(q[:, 1]) - 0.00447980517722963*torch.cos(q[:, 0])*torch.cos(q[:, 2]) - 0.00502017078226511*torch.cos(q[:, 0]) - 0.0728842568484771

    return T

@lru_cache(maxsize=128)
def thumb_forward_kinematics_single(q_tuple: Tuple[float, ...]) -> torch.Tensor:
    """Cached implementation for single joint configuration."""
    q = torch.tensor([q_tuple], dtype=torch.float32)
    return thumb_forward_kinematics_impl(q)[0]

def thumb_forward_kinematics(q: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Forward kinematics for the thumb finger with PyTorch.

    Args:
        q: Joint angles in radians, shape [batch_size, 4] or [4]
        device: Optional device for computation (CPU/GPU)

    Returns:
        Transformation matrices, shape [batch_size, 4, 4] or [4, 4]

    Notes:
        Expects 4 joints: allegro_hand_oya_finger_joint_12, allegro_hand_oya_finger_joint_13, allegro_hand_oya_finger_joint_14, allegro_hand_oya_finger_joint_15
    """
    # Input validation
    if q.shape[-1] != 4:
        raise ValueError(f"Expected 4 joint values, got {q.shape[-1]}")

    # Handle device placement
    if device is None:
        device = q.device if torch.is_tensor(q) else torch.device('cpu')
    elif torch.is_tensor(q) and q.device != device:
        q = q.to(device)

    # Handle various input formats
    unbatched = False
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=torch.float32, device=device)

    if q.dim() == 1:
        unbatched = True
        q_tuple = tuple(q.cpu().numpy().tolist())
        return thumb_forward_kinematics_single(q_tuple).to(device)

    if q.dim() != 2:
        q = q.reshape(-1, 4)

    # Call optimized implementation
    T = thumb_forward_kinematics_impl(q)

    # Handle unbatched case
    return T.squeeze(0) if unbatched else T
