import torch
from typing import Optional, Union, Tuple, List

def use_best_device(tensor: Optional[torch.Tensor] = None) -> torch.device:
    """
    Determine the best available device (CUDA if available, otherwise CPU).
    
    Args:
        tensor: Optional tensor to check device compatibility
        
    Returns:
        The best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # If tensor is provided, check if it's already on CUDA
        if tensor is not None and tensor.device.type != 'cuda':
            # But don't force moving tensors that might be on a different device
            return tensor.device
        return device
    return torch.device('cpu')

def ensure_tensor_on_device(data: Union[torch.Tensor, List, Tuple, float], 
                           device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Ensure data is a tensor on the specified device.
    
    Args:
        data: Input data (tensor, list, tuple or scalar)
        device: Target device (uses best available if None)
        
    Returns:
        Tensor on the specified device
    """
    if device is None:
        device = use_best_device()
        
    if torch.is_tensor(data):
        return data.to(device)
    
    return torch.tensor(data, device=device, dtype=torch.float32)

def enable_cuda_optimization() -> bool:
    """
    Configure PyTorch for optimal CUDA performance.
    
    Returns:
        True if CUDA optimizations were applied, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    # Set benchmark mode for improved performance when input sizes don't change
    torch.backends.cudnn.benchmark = True
    
    # Set deterministic algorithms for reproducibility (comment this out if speed is critical)
    # torch.backends.cudnn.deterministic = True
    
    return True
