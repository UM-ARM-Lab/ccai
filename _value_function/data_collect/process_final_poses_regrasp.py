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
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def calculate_turn_cost(initial_pose, final_pose):
    turn_angle = np.pi/2

    screwdriver_pose = initial_pose.flatten()[-4:-1]
    screwdriver_goal = np.array([0, 0, -turn_angle]) + screwdriver_pose
    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()

    screwdriver_state = final_pose.flatten()[-4:-1]
    screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()

    # make both matrices 3D (batch_size, 3, 3)
    screwdriver_mat = torch.tensor(screwdriver_mat).unsqueeze(0)
    screwdriver_goal_mat = torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0], 1, 1)

    distance2goal = tf.so3_relative_angle(screwdriver_mat, screwdriver_goal_mat, cos_angle=False).detach().cpu()

    final_distance_to_goal = torch.min(distance2goal.abs())
    # if final_distance_to_goal < 30 / 180 * np.pi:
    if final_distance_to_goal < 45 / 180 * np.pi:
        succ = True
    else:
        succ = False

    # we're only actually using the screwdriver values
    state = final_pose.flatten()[-4:-1]
    # upright_cost = 20 * np.sum((state[-3:-1]) ** 2) # the screwdriver should only rotate in z direction
    goal_cost = np.sum((1 * (state[-3:] - screwdriver_goal) ** 2)).reshape(-1)
    total_cost = np.minimum(goal_cost, 5.0)

    return total_cost, succ

def calculate_regrasp_cost(q):
    delta_q = q[1:] - q[:-1]
    smoothness_cost = np.sum((q[1:] - q[-1]) ** 2)
    action_cost = np.sum(delta_q ** 2)

    total_cost = np.minimum(smoothness_cost + action_cost, 5.0)

    return total_cost

if __name__ == "__main__":

    noisy = False
    filenames = []
    if noisy:
        for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("noisy_regrasp_to_turn_dataset*.pkl"):
            filenames.append(file)
    else:
        for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("regrasp_to_turn_dataset*.pkl"):
            filenames.append(file)

    combined_regrasp_poses = np.empty((0, 20))
    combined_regrasp_trajs = np.empty((0, 13, 20))
    combined_turn_poses = np.empty((0, 20))
    combined_turn_trajs = np.empty((0, 13, 20))
    combined_succ_regrasp_tuples = []
    combined_turn_costs = []
    combined_regrasp_costs = []

    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            pregrasp_poses, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs = zip(*pose_tuples)
            
            regrasp_poses = np.array([t.numpy() for t in regrasp_poses]).reshape(-1, 20)
            turn_poses = np.array(turn_poses).reshape(-1, 20)
            combined_turn_poses = np.concatenate((combined_turn_poses, turn_poses), axis=0)
            
            # pad because some trajectories are shorter than 13
            processed_turn_trajectories = []
            for traj in turn_trajs:
                if traj.shape[0] < 13:
                    rows_to_add = 13 - traj.shape[0]
                    last_row = traj[-1]
                    padding = np.tile(last_row, (rows_to_add, 1))
                    padded_traj = np.vstack((traj, padding))
                    processed_turn_trajectories.append(padded_traj)
                else:
                    processed_turn_trajectories.append(traj)

            turn_trajs= np.array(processed_turn_trajectories).reshape(turn_poses.shape[0], -1, 20)
            
            combined_turn_trajs = np.concatenate((combined_turn_trajs, turn_trajs), axis=0)
            combined_regrasp_trajs = np.concatenate((combined_regrasp_trajs, regrasp_trajs), axis=0)
            
            turn_costs = []
            regrasp_costs = []
            for i in range(len(regrasp_poses)):
                cost, succ = calculate_turn_cost(regrasp_poses[i], turn_poses[i])
                regrasp_cost = calculate_regrasp_cost(regrasp_trajs[i])
                turn_costs.append(cost)
                regrasp_costs.append(regrasp_cost)
                if succ:
                    combined_succ_regrasp_tuples.append((regrasp_trajs[i].reshape(13, 20), turn_trajs[i].reshape(13, 20)))
            
            combined_turn_costs.extend(turn_costs)
            combined_regrasp_costs.extend(regrasp_costs)

    combined_turn_costs = np.array(combined_turn_costs)
    combined_regrasp_costs = np.array(combined_regrasp_costs)

    regrasp_cost_dataset = zip(combined_regrasp_trajs, combined_regrasp_costs, combined_turn_trajs, combined_turn_costs)
    
    if noisy:
        regrasp_cost_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/noisy_combined_regrasp_to_turn_dataset.pkl'
        succ_regrasp_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/noisy_regrasp_to_turn_succ_dataset.pkl'
    else:
        regrasp_cost_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset.pkl'
        succ_regrasp_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/regrasp_to_turn_succ_dataset.pkl'
   
    pkl.dump(regrasp_cost_dataset, open(regrasp_cost_savepath, 'wb'))
    pkl.dump(combined_succ_regrasp_tuples, open(succ_regrasp_savepath, 'wb'))

    print("num successes: ", len(combined_succ_regrasp_tuples), "num total: ", len(combined_turn_costs))
    
    plt.figure(figsize=(10, 6))
    plt.hist(combined_turn_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 0 Costs')
    plt.grid(True)
    plt.show()