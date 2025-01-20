import numpy as np
import pickle as pkl
import pathlib
import shutil
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
# from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, emailer, convert_partial_to_full_config, delete_imgs
import torch
from scipy.spatial import KDTree
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def save_neighbors():
    dataset = pkl.load(open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_succ_dataset.pkl', 'rb'))
    neighbors = np.array([tup[0].flatten() for tup in dataset])
    neighbor_actions = np.array([tup[1].flatten() for tup in dataset])
    neighbors[:,-2:] = 0.0   
    neighbor_actions[:,-2:] = 0.0 
    neighbors = torch.tensor(neighbors)
    neighbor_actions = torch.tensor(neighbor_actions)

    pkl.dump(neighbors, open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_neighbors.pkl', 'wb'))
    pkl.dump(neighbor_actions, open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_neighbor_actions.pkl', 'wb'))

    print(f'saved {neighbors.shape[0]} succesful neighbors')
    # tree = KDTree(initials)
    # pkl.dump(tree, open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_tree.pkl', 'wb'))

def find_nn(state):

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
    
    # dataset = pkl.load(open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_succ_dataset.pkl', 'rb'))
    # tree = pkl.load(open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_tree.pkl', 'rb'))
    # distance, idx = tree.query(state_full, k=1)  

    neighbors = pkl.load(open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_neighbors.pkl', 'rb'))
    neighbor_actions = pkl.load(open(f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_neighbor_actions.pkl', 'rb'))

    distances = torch.norm(neighbors - state_full, dim=1)
    min_distance_index = torch.argmin(distances).reshape(1)

    neighbor = neighbors[min_distance_index]
    action = neighbor_actions[min_distance_index]

    return (neighbor, action)


if __name__ == '__main__':
    save_neighbors()

    