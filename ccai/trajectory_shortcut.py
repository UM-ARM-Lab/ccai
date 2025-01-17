import torch

def shortcut_trajectory(trajectory, dx_rob, dx_obj, window=5, epsilon=1e-3):
    """
   
    Finds redundant parts of trajectories and removes them

    Loops over T. Loops over states in the window. If the state in the window is close enough (less than epsilon away) to state T, the states are removed.

    """
    T = trajectory.shape[0]

    new_trajectory = []

    for t in range(0, T):
        if t + window > T:
            window = T - t - 1
        
        state = trajectory[t, :dx_rob+dx_obj]
        state_rob = trajectory[t, :dx_rob]
        action = trajectory[t, dx_rob+dx_obj:]

        skip_inds = []
        num_skips_this_t = 0
        for i in range(1, window):
            if t+i in skip_inds:
                continue
            if torch.norm(trajectory[t+i, :dx_rob] - state_rob) < epsilon:
                action += trajectory[t+i, dx_rob+dx_obj:]
                skip_inds.append(t+i)
                num_skips_this_t += 1
            else:
                print(torch.norm(trajectory[t+i, :dx_rob] - state_rob))
                break
        if num_skips_this_t > 0:
            print('skipped', num_skips_this_t, 'states')
            action /= (num_skips_this_t + 1)
        new_trajectory.append(torch.cat((state, action)).unsqueeze(0))
    
    return torch.cat(new_trajectory, dim=0)