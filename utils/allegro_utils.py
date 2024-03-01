import torch 
from functools import wraps

full_finger_list = ['index', 'middle', 'ring', 'thumb']
def partial_to_full_state(partial, fingers):
    """
    fingers: which fingers are in the partial state
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    num_fingers = len(fingers)
    partial_fingers = torch.chunk(partial, chunks=num_fingers, dim=-1)
    partial_dict = dict(zip(fingers, partial_fingers))
    full = []
    for i, finger in enumerate(full_finger_list):
        if finger in fingers:
            full.append(partial_dict[finger])
        if finger not in fingers:
            full.append(torch.zeros_like(partial_fingers[0]))
    full = torch.cat(full, dim=-1)
    return full

def full_to_partial_state(full, fingers):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, mid, ring, thumb = torch.chunk(full, chunks=4, dim=-1)
    full_dict = dict(zip(full_finger_list, [index, mid, ring, thumb]))
    partial = []
    for finger in fingers:
        partial.append(full_dict[finger])
    partial = torch.cat(partial, dim=-1)
    return partial

def combine_finger_constraints(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        fingers = self.fingers
        # Get values from dict
        xu = kwargs.pop('xu', None)
        if xu is None:
            xu = args[1]

        compute_grads = kwargs.pop('compute_grads', True)
        compute_hess = kwargs.pop('compute_hess', False)

        # compute contact constraints for index finger
        g_list, grad_g_list, hess_g_list = [], [], []
        for finger in fingers:
            g, grad_g, hess_g = func(self, xu, finger_name=finger,
                                    compute_grads=compute_grads, compute_hess=compute_hess, **kwargs)
            g_list.append(g)
            grad_g_list.append(grad_g)
            hess_g_list.append(hess_g)
        g = torch.cat(g_list, dim=1)
        if compute_grads:
            grad_g = torch.cat(grad_g_list, dim=1)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.cat(hess_g_list, dim=1)
            return g, grad_g, hess_g

        return g, grad_g, None

    return wrapper

def state2ee_pos(state, finger_name, fingers, chain, frame_indices, world_trans):
    """
    :params state: B x 8 joint configuration for full hand
    :return ee_pos: B x 3 position of ee

    """
    fk_dict = chain.forward_kinematics(partial_to_full_state(state, fingers), frame_indices=frame_indices)
    m = world_trans.compose(fk_dict[finger_name])
    points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=m.device).unsqueeze(0)
    ee_p = m.transform_points(points_finger_frame).squeeze(-2)
    return ee_p