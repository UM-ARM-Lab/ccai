import torch 
from functools import wraps

def partial_to_full_state(partial):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, thumb = torch.chunk(partial, chunks=2, dim=-1)
    full = torch.cat((
        index,
        torch.zeros_like(index),
        torch.zeros_like(index),
        thumb
    ), dim=-1)
    return full

def full_to_partial_state(full):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, mid, ring, thumb = torch.chunk(full, chunks=4, dim=-1)
    partial = torch.cat((
        index,
        thumb
    ), dim=-1)
    return partial

def combine_finger_constraints(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get values from dict
        xu = kwargs.pop('xu', None)
        if xu is None:
            xu = args[1]

        compute_grads = kwargs.pop('compute_grads', True)
        compute_hess = kwargs.pop('compute_hess', False)

        # compute contact constraints for index finger
        g_i, grad_g_i, hess_g_i = func(args[0], xu, finger_name='index',
                                       compute_grads=compute_grads, compute_hess=compute_hess, **kwargs)
        g_t, grad_g_t, hess_g_t = func(args[0], xu, finger_name='thumb',
                                       compute_grads=compute_grads, compute_hess=compute_hess, **kwargs)

        g = torch.cat((g_i, g_t), dim=1)
        if compute_grads:
            grad_g = torch.cat((grad_g_i, grad_g_t), dim=1)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.cat((hess_g_i, hess_g_t), dim=1)
            return g, grad_g, hess_g

        return g, grad_g, None

    return wrapper

def state2ee_pos(state, finger_name, chain, frame_indices, world_trans):
    """
    :params state: B x 8 joint configuration for full hand
    :return ee_pos: B x 3 position of ee

    """
    fk_dict = chain.forward_kinematics(partial_to_full_state(state), frame_indices=frame_indices)
    m = world_trans.compose(fk_dict[finger_name])
    points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=m.device).unsqueeze(0)
    ee_p = m.transform_points(points_finger_frame).squeeze(-2)
    return ee_p