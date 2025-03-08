import torch
from typing import Union, Tuple, List, Optional
from functools import lru_cache

def index_forward_kinematics_impl(q: torch.Tensor) -> torch.Tensor:
    """
    Forward kinematics implementation for index finger.
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
    T[:, 1, 0] = 6.12323399573677e-17*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 0.996194698091746*torch.cos(q[:, 0])
    T[:, 1, 1] = 1.0*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 3.73513185974743e-33*torch.cos(q[:, 0])
    T[:, 1, 2] = 6.12323399573677e-17*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 1.0*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) - 6.0999332417281e-17*torch.cos(q[:, 0])
    T[:, 1, 3] = 0.046*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 0.013*(-(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.013*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 0.046*((0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 0.0384*(0.996194698091746*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.0871557427476582*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0384*(0.996194698091746*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0871557427476582*torch.sin(q[:, 1]))*torch.sin(q[:, 2]) + 0.0537945136969543*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 1.12053955792423e-34*torch.cos(q[:, 0]) + 0.00470641010837354*torch.cos(q[:, 1]) + 0.0452779771520522
    T[:, 2, 0] = 6.12323399573677e-17*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 6.12323399573677e-17*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) - 0.0871557427476582*torch.cos(q[:, 0])
    T[:, 2, 1] = 1.0*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 6.12323399573677e-17*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 6.12323399573677e-17*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) - 3.26781694502401e-34*torch.cos(q[:, 0])
    T[:, 2, 2] = 6.12323399573677e-17*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) + 1.0*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) - 1.0*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 6.12323399573677e-17*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 5.33675006916149e-18*torch.cos(q[:, 0])
    T[:, 2, 3] = 0.046*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.sin(q[:, 3]) - 0.013*(-(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.sin(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.cos(q[:, 2]))*torch.cos(q[:, 3]) + 0.013*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.sin(q[:, 3]) + 0.046*((-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + (-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]))*torch.cos(q[:, 3]) + 0.0384*(-0.0871557427476582*torch.sin(q[:, 0])*torch.sin(q[:, 1]) + 0.996194698091746*torch.cos(q[:, 1]))*torch.cos(q[:, 2]) + 0.0384*(-0.0871557427476582*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.996194698091746*torch.sin(q[:, 1]))*torch.sin(q[:, 2]) - 0.00470641010837354*torch.sin(q[:, 0])*torch.sin(q[:, 1]) - 9.80345083507204e-36*torch.cos(q[:, 0]) + 0.0537945136969543*torch.cos(q[:, 1]) + 0.0187803718410716

    return T

@lru_cache(maxsize=128)
def index_forward_kinematics_single(q_tuple: Tuple[float, ...]) -> torch.Tensor:
    """Cached implementation for single joint configuration."""
    q = torch.tensor([q_tuple], dtype=torch.float32)
    return index_forward_kinematics_impl(q)[0]

def index_forward_kinematics(q: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Forward kinematics for the index finger with PyTorch.

    Args:
        q: Joint angles in radians, shape [batch_size, 4] or [4]
        device: Optional device for computation (CPU/GPU)

    Returns:
        Transformation matrices, shape [batch_size, 4, 4] or [4, 4]

    Notes:
        Expects 4 joints: allegro_hand_hitosashi_finger_finger_joint_0, allegro_hand_hitosashi_finger_finger_joint_1, allegro_hand_hitosashi_finger_finger_joint_2, allegro_hand_hitosashi_finger_finger_joint_3
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
        return index_forward_kinematics_single(q_tuple).to(device)

    if q.dim() != 2:
        q = q.reshape(-1, 4)

    # Call optimized implementation
    T = index_forward_kinematics_impl(q)

    # Handle unbatched case
    return T.squeeze(0) if unbatched else T
