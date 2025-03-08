import torch
from typing import Union, Tuple, List, Optional
from functools import lru_cache

def middle_forward_kinematics_impl(q: torch.Tensor) -> torch.Tensor:
    """
    Forward kinematics implementation for middle finger.
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
    T[:, 0, 0] = 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) - 1.0*torch.sin(q[:, 0])
    T[:, 0, 1] = 1.0*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 1.0*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) - 3.74939945665464e-33*torch.sin(q[:, 0])
    T[:, 0, 2] = 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 1.0*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*torch.sin(q[:, 0])
    T[:, 0, 3] = 0.046*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 0.013*(-torch.sin(q[:, 1])*torch.sin(q[:, 2])*torch.cos(q[:, 0]) + torch.cos(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.013*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 0.046*(torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) - 1.12481983699639e-34*torch.sin(q[:, 0]) + 0.0384*torch.sin(q[:, 1])*torch.cos(q[:, 0])*torch.cos(q[:, 2]) + 0.054*torch.sin(q[:, 1])*torch.cos(q[:, 0]) + 0.0384*torch.sin(q[:, 2])*torch.cos(q[:, 0])*torch.cos(q[:, 1])
    T[:, 1, 0] = 6.12323399573677e-17*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 1.0*torch.cos(q[:, 0])
    T[:, 1, 1] = 1.0*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 1.0*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) + 3.74939945665464e-33*torch.cos(q[:, 0])
    T[:, 1, 2] = 6.12323399573677e-17*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 1.0*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*torch.cos(q[:, 0])
    T[:, 1, 3] = 0.046*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 0.013*(-torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.sin(q[:, 0])*torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.013*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 0.046*(torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) + 0.0384*torch.sin(q[:, 0])*torch.sin(q[:, 1])*torch.cos(q[:, 2]) + 0.054*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0384*torch.sin(q[:, 0])*torch.sin(q[:, 2])*torch.cos(q[:, 1]) + 1.12481983699639e-34*torch.cos(q[:, 0])
    T[:, 2, 0] = -6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3])
    T[:, 2, 1] = 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 1.0*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3])
    T[:, 2, 2] = -1.0*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) + 1.0*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3])
    T[:, 2, 3] = 0.013*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 0.046*(-torch.sin(q[:, 1])*torch.sin(q[:, 2]) + torch.cos(q[:, 1])*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.046*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.sin(q[:, 3]) - 0.013*(-torch.sin(q[:, 1])*torch.cos(q[:, 2]) - torch.sin(q[:, 2])*torch.cos(q[:, 1]))*torch.cos(q[:, 3]) - 0.0384*torch.sin(q[:, 1])*torch.sin(q[:, 2]) + 0.0384*torch.cos(q[:, 1])*torch.cos(q[:, 2]) + 0.054*torch.cos(q[:, 1]) + 0.0211

    return T

@lru_cache(maxsize=128)
def middle_forward_kinematics_single(q_tuple: Tuple[float, ...]) -> torch.Tensor:
    """Cached implementation for single joint configuration."""
    q = torch.tensor([q_tuple], dtype=torch.float32)
    return middle_forward_kinematics_impl(q)[0]

def middle_forward_kinematics(q: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Forward kinematics for the middle finger with PyTorch.

    Args:
        q: Joint angles in radians, shape [batch_size, 4] or [4]
        device: Optional device for computation (CPU/GPU)

    Returns:
        Transformation matrices, shape [batch_size, 4, 4] or [4, 4]

    Notes:
        Expects 4 joints: allegro_hand_naka_finger_finger_joint_4, allegro_hand_naka_finger_finger_joint_5, allegro_hand_naka_finger_finger_joint_6, allegro_hand_naka_finger_finger_joint_7
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
        return middle_forward_kinematics_single(q_tuple).to(device)

    if q.dim() != 2:
        q = q.reshape(-1, 4)

    # Call optimized implementation
    T = middle_forward_kinematics_impl(q)

    # Handle unbatched case
    return T.squeeze(0) if unbatched else T
