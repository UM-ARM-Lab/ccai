import torch
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps

def timeit(func):
    """Decorator for timing function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper

def optimize_jit_compilation(module: torch.nn.Module) -> torch.jit.ScriptModule:
    """Convert a PyTorch module to TorchScript for faster execution."""
    return torch.jit.script(module)

def precompile_functions(finger_names: list, link_names: Dict[str, list]) -> None:
    """
    Precompile JIT functions for all fingers and links.
    
    Args:
        finger_names: List of finger names
        link_names: Dictionary mapping finger names to list of links
    """
    from .kinematics_jit import get_compiled_fk_func, get_compiled_jacobian_func
    
    # Precompile FK functions
    for finger in finger_names:
        try:
            get_compiled_fk_func(finger)
            print(f"Precompiled FK for {finger}")
        except Exception as e:
            print(f"Failed to precompile FK for {finger}: {e}")
    
    # Precompile Jacobian functions
    for finger, links in link_names.items():
        for link in links:
            try:
                get_compiled_jacobian_func(finger, link)
                print(f"Precompiled Jacobian for {finger}/{link}")
            except Exception as e:
                print(f"Failed to precompile Jacobian for {finger}/{link}: {e}")

def use_fused_ops(func: Callable) -> Callable:
    """
    Decorator to convert a function to use fused operations where possible.
    This requires PyTorch 1.7.0+ and works best on GPU.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to use AMP (automatic mixed precision) if on GPU
        device = kwargs.get('device')
        if device is None and args and torch.is_tensor(args[0]):
            device = args[0].device
        
        if device is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return func(*args, **kwargs)
        
        return func(*args, **kwargs)
    
    return wrapper

def cached_tensor_computation(maxsize: int = 128):
    """
    Advanced caching decorator for tensor computations.
    Handles tensors by caching based on tensor values rather than identity.
    """
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert tensor args to hashable representation (tuple of values)
            cache_key = []
            for arg in args:
                if torch.is_tensor(arg):
                    # Use tensor data for caching instead of the tensor object
                    cache_key.append(tuple(arg.detach().cpu().numpy().flatten()))
                else:
                    cache_key.append(arg)
            
            # Add kwargs to cache key
            for k, v in sorted(kwargs.items()):
                if torch.is_tensor(v):
                    cache_key.append((k, tuple(v.detach().cpu().numpy().flatten())))
                else:
                    cache_key.append((k, v))
            
            # Make final cache key hashable
            cache_key = tuple(cache_key)
            
            if cache_key in cache:
                return cache[cache_key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Simple FIFO eviction strategy
                cache.pop(next(iter(cache)))
            
            cache[cache_key] = result
            return result
        
        return wrapper
    
    return decorator
