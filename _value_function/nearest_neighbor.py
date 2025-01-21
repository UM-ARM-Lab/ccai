import numpy as np
import pickle as pkl
import pathlib
import shutil
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
# from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, convert_partial_to_full_config, delete_imgs
import torch
from scipy.spatial import KDTree
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def save_neighbors():
    dataset = pkl.load(open(f'{fpath.resolve()}/regrasp_to_turn_datasets/regrasp_to_turn_succ_dataset.pkl', 'rb'))
    neighbors = np.array([tup[0][0] for tup in dataset])
    actions = np.array([tup[1][-1] for tup in dataset])

    neighbors[:,-2:] = 0.0   
    actions[:,-2:] = 0.0 

    neighbors = torch.tensor(neighbors)
    actions = torch.tensor(actions)

    pkl.dump(neighbors, open(f'{fpath.resolve()}/regrasp_to_turn_datasets/neighbors.pkl', 'wb'))
    pkl.dump(actions, open(f'{fpath.resolve()}/regrasp_to_turn_datasets/actions.pkl', 'wb'))

    neighbor_trajs = np.array([tup[0] for tup in dataset])
    action_trajs = np.array([tup[1] for tup in dataset])

    neighbor_trajs[:,:,-2] = 0.0
    action_trajs[:,:,-2] = 0.0

    neighbor_trajs = torch.tensor(neighbor_trajs)
    action_trajs = torch.tensor(action_trajs)

    pkl.dump(neighbor_trajs, open(f'{fpath.resolve()}/regrasp_to_turn_datasets/neighbor_trajs.pkl', 'wb'))
    pkl.dump(action_trajs, open(f'{fpath.resolve()}/regrasp_to_turn_datasets/action_trajs.pkl', 'wb'))

    print(f'saved {neighbors.shape[0]} succesful neighbors')

def find_nn_0(state):

    state=state.cpu().flatten()
    if state.shape[-1] == 16:
        state_full = torch.concat((
                    state[:8], 
                    torch.tensor([0., 0.5, 0.65, 0.65], dtype=torch.float32), 
                    state[8:]
                    ))
    elif state.shape[-1] == 20:
        state_full = state

    state_full[-2:] = 0.0

    neighbors = pkl.load(open(f'{fpath.resolve()}/regrasp_to_turn_datasets/neighbors.pkl', 'rb'))
    actions = pkl.load(open(f'{fpath.resolve()}/regrasp_to_turn_datasets/actions.pkl', 'rb'))

    distances = torch.norm(neighbors - state_full, dim=1)
    min_distance_index = torch.argmin(distances).reshape(1)

    neighbor = neighbors[min_distance_index]
    action = actions[min_distance_index]

    return (neighbor, action)

def find_nn_1(state):

    state=state.cpu().reshape(13, -1)
    if state.shape[-1] == 16:

        insertion_vector = torch.tensor([0., 0.5, 0.65, 0.65], dtype=torch.float32)
        state_full = torch.cat((
            state[:, :8],  
            insertion_vector.expand(state.shape[0], -1),  
            state[:, 8:]
        ), dim=-1)
        
    elif state.shape[-1] == 20:
        state_full = state

    state_full[:,-2:] = 0.0
    neighbors = pkl.load(open(f'{fpath.resolve()}/regrasp_to_turn_datasets/neighbor_trajs.pkl', 'rb'))
    actions = pkl.load(open(f'{fpath.resolve()}/regrasp_to_turn_datasets/action_trajs.pkl', 'rb'))

    distances = torch.sum(torch.norm(neighbors - state_full, dim=1), dim=1)
    min_distance_index = torch.argmin(distances).reshape(1)

    neighbor = neighbors[min_distance_index]
    action = actions[min_distance_index]

    return (neighbor, action)


if __name__ == '__main__':
    save_neighbors()

    neighbors = pkl.load(open(f'{fpath.resolve()}/regrasp_to_turn_datasets/neighbor_trajs.pkl', 'rb'))
    test = neighbors[12]
    find_nn_1(test)

    