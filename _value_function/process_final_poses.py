import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import wandb
import matplotlib.pyplot as plt

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')


filenames = []
#for i in range(10000/200):
for i in [0,1,4,5]:
    filename = f'value_dataset_odin_{i*200}.pkl'
    filenames.append(filename)

def calculate_cost(final_pose):

    goal = np.array([0.0,0.0,-1.5708])
    obj_dof = 3

    # final_pose: (N, 20)
    #state = np.concatenate((final_pose[:, :8], final_pose[:, 12:19]), axis=1)
    # we're only actually using the screwdriver values
    state = final_pose[:, -4:-1]

    upright_cost = 20 * np.sum((state[:, -obj_dof:-1]) ** 2, axis = 1) # the screwdriver should only rotate in z direction
    goal_cost = np.sum((1 * (state[:, -obj_dof:] - goal) ** 2), axis = 1).reshape(-1)
    # add a running cost
    #goal_cost += torch.sum((1 * (state[:, -obj_dof:] - goal.unsqueeze(0)) ** 2))

    return list(goal_cost + upright_cost)

if __name__ == "__main__":

    combined_initial_poses = np.empty((0, 20))
    combined_final_poses = np.empty((0, 20))
    combined_costs = []
    for filename in filenames:
        with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
            pose_tuples = pkl.load(file)
            initial_poses, final_poses = zip(*pose_tuples)
            initial_poses = np.array([t.numpy() for t in initial_poses]).reshape(-1, 20)
            final_poses = np.array(final_poses).reshape(-1, 20)
            # Use np.concatenate to append arrays
            combined_initial_poses = np.concatenate((combined_initial_poses, initial_poses), axis=0)
            combined_final_poses = np.concatenate((combined_final_poses, final_poses), axis=0)

            costs = calculate_cost(final_poses)
            combined_costs.extend(costs)
    
    combined_costs = np.array(combined_costs)
    print(np.mean(combined_costs))

    pose_cost_dataset = zip(combined_initial_poses, combined_costs)
    pose_cost_savepath = f'{fpath.resolve()}/combined_value_dataset.pkl'
    with open(pose_cost_savepath, 'wb') as f:
        pkl.dump(pose_cost_dataset, f)

    final_poses_savepath = f'{fpath.resolve()}/combined_final_poses.pkl'
    final_poses_dataset = combined_final_poses
    with open(final_poses_savepath, 'wb') as f:
        pkl.dump(final_poses_dataset, f)