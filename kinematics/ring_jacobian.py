import torch
from typing import Union, Tuple, List, Dict, Optional
from functools import lru_cache

def ring_jacobian_impl(q: torch.Tensor, link_name: str) -> torch.Tensor:
    """
    Jacobian calculation for ring finger.
    Takes joint angles and link name, returns 6xN Jacobian matrix.

    Args:
        q: Joint angles tensor of shape [batch_size, 4]
        link_name: Name of the link to compute Jacobian for

    Returns:
        Jacobian matrix of shape [batch_size, 6, 4]
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype

    # Initialize Jacobian matrix
    J = torch.zeros(batch_size, 6, 4, dtype=dtype, device=device)

    if link_name == 'allegro_hand_kusuri_finger_finger_link_9':
        J[:, 0, 0] = -2.16840434497101e-18
        J[:, 4, 0] = -0.0871557427476582
        J[:, 5, 0] = 0.996194698091746
    elif link_name == 'allegro_hand_kusuri_finger_finger_link_10':
        J[:, 0, 0] = -0.054*torch.sin(q[:, 0])*torch.sin(q[:, 1]) - 2.16840434497101e-18
        J[:, 0, 1] = 0.054*torch.cos(q[:, 0])*torch.cos(q[:, 1])
        J[:, 1, 0] = 0.0537945136969543*torch.sin(q[:, 1])*torch.cos(q[:, 0])
        J[:, 1, 1] = 0.0537945136969543*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.00470641010837354*torch.sin(q[:, 1])
        J[:, 2, 0] = 0.00470641010837354*torch.sin(q[:, 1])*torch.cos(q[:, 0])
        J[:, 2, 1] = 0.00470641010837354*torch.sin(q[:, 0])*torch.cos(q[:, 1]) - 0.0537945136969543*torch.sin(q[:, 1])
        J[:, 3, 1] = -1.0*torch.sin(q[:, 0])
        J[:, 4, 0] = -0.0871557427476582
        J[:, 4, 1] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 5, 0] = 0.996194698091746
        J[:, 5, 1] = 0.0871557427476582*torch.cos(q[:, 0])
    elif link_name == 'allegro_hand_kusuri_finger_finger_link_11':
        J[:, 0, 0] = -0.054*torch.sin(q[:, 0])*torch.sin(q[:, 1]) - 0.0384*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2]) - 2.16840434497101e-18
        J[:, 0, 1] = (0.054*torch.cos(q[:, 1]) + 0.0384*torch.cos(q[:, 1] + q[:, 2]))*torch.cos(q[:, 0])
        J[:, 0, 2] = 0.0384*torch.cos(q[:, 0])*torch.cos(q[:, 1] + q[:, 2])
        J[:, 1, 0] = (0.0537945136969543*torch.sin(q[:, 1]) + 0.038253876406723*torch.sin(q[:, 1] + q[:, 2]))*torch.cos(q[:, 0])
        J[:, 1, 1] = -8.67361737988404e-19*torch.sin(q[:, 0])**2*torch.sin(q[:, 1] + q[:, 2]) + 0.0537945136969543*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.038253876406723*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.00470641010837354*torch.sin(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2])
        J[:, 1, 2] = 0.038253876406723*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2])
        J[:, 2, 0] = (0.00470641010837354*torch.sin(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2]))*torch.cos(q[:, 0])
        J[:, 2, 1] = 0.00470641010837354*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) - 0.0537945136969543*torch.sin(q[:, 1]) - 0.038253876406723*torch.sin(q[:, 1] + q[:, 2])
        J[:, 2, 2] = 0.00334678052151007*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) - 0.038253876406723*torch.sin(q[:, 1] + q[:, 2])
        J[:, 3, 1] = -1.0*torch.sin(q[:, 0])
        J[:, 3, 2] = -1.0*torch.sin(q[:, 0])
        J[:, 4, 0] = -0.0871557427476582
        J[:, 4, 1] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 4, 2] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 5, 0] = 0.996194698091746
        J[:, 5, 1] = 0.0871557427476582*torch.cos(q[:, 0])
        J[:, 5, 2] = 0.0871557427476582*torch.cos(q[:, 0])
    elif link_name == 'allegro_hand_kusuri_finger_finger_2_aftc_base_link':
        J[:, 0, 0] = -0.054*torch.sin(q[:, 0])*torch.sin(q[:, 1]) - 0.0384*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2]) - 0.016*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.013*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 2.16840434497101e-19*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 2.16840434497101e-18
        J[:, 0, 1] = (2.16840434497101e-19*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.013*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.054*torch.cos(q[:, 1]) + 0.0384*torch.cos(q[:, 1] + q[:, 2]) + 0.016*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 0, 2] = (2.16840434497101e-19*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.013*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0384*torch.cos(q[:, 1] + q[:, 2]) + 0.016*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 0, 3] = (2.16840434497101e-19*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.013*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.016*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 1, 0] = (0.0537945136969543*torch.sin(q[:, 1]) + 0.038253876406723*torch.sin(q[:, 1] + q[:, 2]) + 0.0159391151694679*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 1, 1] = -8.67361737988404e-19*torch.sin(q[:, 0])**2*torch.sin(q[:, 1] + q[:, 2]) + 0.0129505310751927*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0537945136969543*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.038253876406723*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.0159391151694679*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00470641010837354*torch.sin(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2]) + 0.00139449188396253*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 1, 2] = 0.0129505310751927*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.038253876406723*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.0159391151694679*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2]) + 0.00139449188396253*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 1, 3] = 0.0129505310751927*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0159391151694679*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00139449188396253*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 0] = (0.00470641010837354*torch.sin(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2]) + 0.00139449188396253*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 2, 1] = 0.00113302465571956*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00470641010837354*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.00139449188396253*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 0.0537945136969543*torch.sin(q[:, 1]) - 0.038253876406723*torch.sin(q[:, 1] + q[:, 2]) - 0.0159391151694679*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 2] = 0.00113302465571956*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00334678052151007*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.00139449188396253*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 0.038253876406723*torch.sin(q[:, 1] + q[:, 2]) - 0.0159391151694679*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 3] = 0.00113302465571956*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00139449188396253*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 0.0159391151694679*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 3, 1] = -1.0*torch.sin(q[:, 0])
        J[:, 3, 2] = -1.0*torch.sin(q[:, 0])
        J[:, 3, 3] = -1.0*torch.sin(q[:, 0])
        J[:, 4, 0] = -0.0871557427476582
        J[:, 4, 1] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 4, 2] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 4, 3] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 5, 0] = 0.996194698091746
        J[:, 5, 1] = 0.0871557427476582*torch.cos(q[:, 0])
        J[:, 5, 2] = 0.0871557427476582*torch.cos(q[:, 0])
        J[:, 5, 3] = 0.0871557427476582*torch.cos(q[:, 0])
    elif link_name == 'kusuri_ee':
        J[:, 0, 0] = -0.054*torch.sin(q[:, 0])*torch.sin(q[:, 1]) - 0.0384*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2]) - 0.046*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.013*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 1.12481983699639e-34*torch.cos(q[:, 0]) - 2.16840434497101e-18
        J[:, 0, 1] = (0.013*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 1.3363823550461e-51*torch.cos(q[:, 0]) + 0.054*torch.cos(q[:, 1]) + 0.0384*torch.cos(q[:, 1] + q[:, 2]) + 0.046*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 0, 2] = (0.013*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 1.3363823550461e-51*torch.cos(q[:, 0]) + 0.0384*torch.cos(q[:, 1] + q[:, 2]) + 0.046*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 0, 3] = (0.013*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 1.3363823550461e-51*torch.cos(q[:, 0]) + 0.046*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]))*torch.cos(q[:, 0])
        J[:, 1, 0] = -1.12053955792423e-34*torch.sin(q[:, 0]) + 0.0537945136969543*torch.sin(q[:, 1])*torch.cos(q[:, 0]) + 0.038253876406723*torch.sin(q[:, 1] + q[:, 2])*torch.cos(q[:, 0]) + 0.0458249561122203*torch.sin(q[:, 1] + q[:, 2] + q[:, 3])*torch.cos(q[:, 0]) - 0.0129505310751927*torch.cos(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 1, 1] = 6.68191177523049e-52*torch.sin(2*q[:, 0]) + 0.00470641010837354*torch.sin(q[:, 1]) + 0.0268972568484771*torch.sin(q[:, 0] - q[:, 1]) + 0.0268972568484771*torch.sin(q[:, 0] + q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2]) + 2.16840434497101e-19*torch.sin(-2*q[:, 0] + q[:, 1] + q[:, 2]) - 0.0191269382033615*torch.sin(-q[:, 0] + q[:, 1] + q[:, 2]) + 0.0191269382033615*torch.sin(q[:, 0] + q[:, 1] + q[:, 2]) + 2.16840434497101e-19*torch.sin(2*q[:, 0] + q[:, 1] + q[:, 2]) + 0.00400916416639228*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.0229124780561101*torch.sin(-q[:, 0] + q[:, 1] + q[:, 2] + q[:, 3]) + 0.0229124780561101*torch.sin(q[:, 0] + q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00647526553759635*torch.cos(-q[:, 0] + q[:, 1] + q[:, 2] + q[:, 3]) - 0.00647526553759635*torch.cos(q[:, 0] + q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 1, 2] = 4.33680868994202e-19*torch.sin(q[:, 0])**2*torch.sin(q[:, 1] + q[:, 2]) + 0.0129505310751927*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.038253876406723*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.0458249561122203*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2]) + 0.00400916416639228*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 1, 3] = 0.0129505310751927*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0458249561122203*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00400916416639228*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) - 0.00113302465571956*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 0] = -9.80345083507204e-36*torch.sin(q[:, 0]) + 0.00470641010837354*torch.sin(q[:, 1])*torch.cos(q[:, 0]) + 0.00334678052151007*torch.sin(q[:, 1] + q[:, 2])*torch.cos(q[:, 0]) + 0.00400916416639228*torch.sin(q[:, 1] + q[:, 2] + q[:, 3])*torch.cos(q[:, 0]) - 0.00113302465571956*torch.cos(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 1] = 3.46944695195361e-18*torch.sin(q[:, 0])**2*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00113302465571956*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00470641010837354*torch.sin(q[:, 0])*torch.cos(q[:, 1]) + 0.00334678052151007*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.00400916416639228*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 0.0537945136969543*torch.sin(q[:, 1]) - 0.038253876406723*torch.sin(q[:, 1] + q[:, 2]) - 0.0458249561122203*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 2] = 1.73472347597681e-18*torch.sin(q[:, 0])**2*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00113302465571956*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00334678052151007*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2]) + 0.00400916416639228*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 0.038253876406723*torch.sin(q[:, 1] + q[:, 2]) - 0.0458249561122203*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 2, 3] = 1.73472347597681e-18*torch.sin(q[:, 0])**2*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00113302465571956*torch.sin(q[:, 0])*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.00400916416639228*torch.sin(q[:, 0])*torch.cos(q[:, 1] + q[:, 2] + q[:, 3]) - 0.0458249561122203*torch.sin(q[:, 1] + q[:, 2] + q[:, 3]) + 0.0129505310751927*torch.cos(q[:, 1] + q[:, 2] + q[:, 3])
        J[:, 3, 1] = -1.0*torch.sin(q[:, 0])
        J[:, 3, 2] = -1.0*torch.sin(q[:, 0])
        J[:, 3, 3] = -1.0*torch.sin(q[:, 0])
        J[:, 4, 0] = -0.0871557427476582
        J[:, 4, 1] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 4, 2] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 4, 3] = 0.996194698091746*torch.cos(q[:, 0])
        J[:, 5, 0] = 0.996194698091746
        J[:, 5, 1] = 0.0871557427476582*torch.cos(q[:, 0])
        J[:, 5, 2] = 0.0871557427476582*torch.cos(q[:, 0])
        J[:, 5, 3] = 0.0871557427476582*torch.cos(q[:, 0])
    elif link_name == 'kusuri_ee':
        pass  # End effector with empty block
    else:
        raise ValueError(f"Unknown link name {link_name} for ring finger")

    # Return the Jacobian matrix
    return J

@lru_cache(maxsize=128)
def ring_jacobian_single(q_tuple: Tuple[float, ...], link_name: str) -> torch.Tensor:
    """Cached Jacobian implementation for single joint configuration."""
    q = torch.tensor([q_tuple], dtype=torch.float32)
    return ring_jacobian_impl(q, link_name)[0]

def ring_jacobian(q: torch.Tensor, link_name: Optional[str] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Jacobian calculation for the ring finger with PyTorch.

    Args:
        q: Joint angles in radians, shape [batch_size, 4] or [4]
        link_name: Optional name of the link to compute Jacobian for
        device: Optional device for computation (CPU/GPU)

    Returns:
        Jacobian matrix of shape [batch_size, 6, 4] or [6, 4]

    Notes:
        Defaults to end effector link if link_name is not specified.
    """
    # Input validation
    if q.shape[-1] != 4:
        raise ValueError(f"Expected 4 joint values, got {q.shape[-1]}")

    # Handle device placement
    if device is None:
        device = q.device if torch.is_tensor(q) else torch.device('cpu')
    elif torch.is_tensor(q) and q.device != device:
        q = q.to(device)

    # Default to end effector if not specified
    if link_name is None or link_name == '':
        link_name = 'kusuri_ee'

    # Handle various input formats
    unbatched = False
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=torch.float32, device=device)

    if q.dim() == 1:
        unbatched = True
        q_tuple = tuple(q.cpu().numpy().tolist())
        return ring_jacobian_single(q_tuple, link_name).to(device)

    if q.dim() != 2:
        q = q.reshape(-1, 4)

    # Call implementation function
    J = ring_jacobian_impl(q, link_name)

    # Handle unbatched case
    return J.squeeze(0) if unbatched else J
