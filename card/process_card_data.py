from _value_function.screwdriver_problem import init_env
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import wandb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics.transforms as tf
import time
from pathlib import Path
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
torch.serialization.default_restore_location = lambda storage, loc: storage.cpu()


def calculate_cost(initial_pose, final_pose):

    card_pose = initial_pose.flatten()[-5]
    desired_delta = -0.02*3
    goal = desired_delta + card_pose

    state = final_pose.flatten()

    goal_cost = ((state[-5] - goal) ** 2)
    # goal_cost = sum(goal_cost)

    return goal_cost

if __name__ == "__main__":

    filenames = []

    for file in Path(f'{fpath.resolve()}/card_datasets').glob("card_dataset*.pkl"):
        filenames.append(file)

    combined_index_trajs = np.empty((0, 9, 22))
    combined_middle_trajs = np.empty((0, 9, 22))
    # combined_start_yaws = []
    combined_costs = []

    vis = False
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=vis)

    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            initializations, traj_index1s, traj_middles, traj_index2s, pose_index2s, *extra = zip(*pose_tuples)
            
            start_poses = np.array([t.numpy() for t in initializations]).reshape(-1, 22)
            final_poses = np.array([t.numpy() for t in pose_index2s]).reshape(-1, 22)

            traj_index1s = [t.reshape(9, 22) for t in traj_index1s]
            traj_middles = [t.reshape(9, 22) for t in traj_middles]
            traj_index2s = [t.reshape(9, 22) for t in traj_index2s]

            original_length = len(traj_index1s)

            for i in range(original_length - 1, -1, -1):
                if traj_index1s[i].shape[0] != 9 or traj_middles[i].shape[0] != 9 or traj_index2s[i].shape[0] != 9:
                    print("Broken trajectory removed")

                    start_poses = np.delete(start_poses, i, axis=0)
                    final_poses = np.delete(final_poses, i, axis=0)

                    traj_index1s.pop(i)
                    traj_middles.pop(i)
                    traj_index2s.pop(i)
            
            combined_index_trajs = np.concatenate((combined_index_trajs, np.stack(traj_index1s, axis=0), np.stack(traj_index2s, axis=0)), axis=0)
            combined_middle_trajs = np.concatenate((combined_middle_trajs, np.stack(traj_middles, axis=0)), axis=0)
            
            costs = []
            
            for i in range(len(start_poses)):
                cost = calculate_cost(start_poses[i], final_poses[i])
                # combined_start_yaws.append(final_poses[i][-2])

                costs.append(cost)
            
            combined_costs.extend(costs)

    combined_costs = np.array(combined_costs)
    # combined_start_yaws = np.array(combined_start_yaws)

    if combined_index_trajs.shape[0] > 20000:
        combined_index_trajs = combined_index_trajs[:20000]
        combined_middle_trajs = combined_middle_trajs[:10000]

    index_dataset = zip(combined_index_trajs, combined_costs)
    middle_dataset = zip(combined_middle_trajs, combined_costs)
    
    index_savepath = f'{fpath.resolve()}/card_datasets/combined_index_dataset.pkl'
    middle_savepath = f'{fpath.resolve()}/card_datasets/combined_middle_dataset.pkl'
   
    pkl.dump(index_dataset, open(index_savepath, 'wb'))
    pkl.dump(middle_dataset, open(middle_savepath, 'wb'))

    print("num samples: ", len(combined_costs))
    
    plt.figure(figsize=(10, 6))
    plt.hist(combined_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 0 Costs')
    plt.grid(True)
    plt.show()