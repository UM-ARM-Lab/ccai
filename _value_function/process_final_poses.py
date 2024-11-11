from screwdriver_problem import init_env
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


filenames = []
for file in Path(f'{fpath.resolve()}/value_datasets').glob("value_dataset_*.pkl"):
    filenames.append(file)

def calculate_cost(initial_pose, final_pose):

    screwdriver_pose = initial_pose[-4:-1]
    screwdriver_goal = np.array([0, 0, -np.pi/2]) + screwdriver_pose
    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()

    screwdriver_state = final_pose[-4:-1]
    screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()

    # make both matrices 3D (batch_size, 3, 3)
    screwdriver_mat = torch.tensor(screwdriver_mat).unsqueeze(0)
    screwdriver_goal_mat = torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0], 1, 1)

    distance2goal = tf.so3_relative_angle(screwdriver_mat, screwdriver_goal_mat, cos_angle=False).detach().cpu()

    final_distance_to_goal = torch.min(distance2goal.abs())
    if final_distance_to_goal < 30 / 180 * np.pi:
        succ = True
    else:
        succ = False

    # final_pose: (N, 20)
    #state = np.concatenate((final_pose[:, :8], final_pose[:, 12:19]), axis=1)
    # we're only actually using the screwdriver values
    state = final_pose[-4:-1]
    
    # upright_cost = 20 * np.sum((state[-3:-1]) ** 2) # the screwdriver should only rotate in z direction
    goal_cost = np.sum((1 * (state[-3:] - screwdriver_goal) ** 2)).reshape(-1)
    # total_cost = np.minimum(goal_cost + upright_cost, 10.0)
    total_cost = np.minimum(goal_cost, 5.0)

    return total_cost, succ

if __name__ == "__main__":

    combined_initial_poses = np.empty((0, 20))
    combined_final_poses = np.empty((0, 20))
    combined_costs = []
    succs = []
    fails = []
    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)
            initial_poses, final_poses = zip(*pose_tuples)
            initial_poses = np.array([t.numpy() for t in initial_poses]).reshape(-1, 20)
            final_poses = np.array(final_poses).reshape(-1, 20)
            # Use np.concatenate to append arrays
            combined_initial_poses = np.concatenate((combined_initial_poses, initial_poses), axis=0)
            combined_final_poses = np.concatenate((combined_final_poses, final_poses), axis=0)

            costs = []
            for i in range(len(final_poses)):
                cost, succ = calculate_cost(initial_poses[i], final_poses[i])
                costs.append(cost)
                if succ:
                    succs.append(i)
                else:
                    fails.append(i)

            combined_costs.extend(costs)

    combined_costs = np.array(combined_costs)
    print("combined costs shape: ", combined_costs.shape)
    # plot costs
    # plt.figure(figsize=(10, 6))
    # plt.hist(combined_costs.flatten(), bins=50, color='blue', label='Costs')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Cost Value')
    # plt.title('Costs')
    # plt.grid(True)
    # plt.show()

    vis = False
    if vis:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for idx in fails:
            if combined_costs[idx] > 5.0:
                env.reset(torch.from_numpy(combined_initial_poses[idx]).reshape(1,20).float())
                time.sleep(0.5)
                print("cost: ", combined_costs[idx])
                env.reset(torch.from_numpy(combined_final_poses[idx]).reshape(1,20).float())
                time.sleep(1.0)

    pose_cost_dataset = zip(combined_initial_poses, combined_costs)
    pose_cost_savepath = f'{fpath.resolve()}/value_datasets/combined_value_dataset.pkl'
    with open(pose_cost_savepath, 'wb') as f:
        pkl.dump(pose_cost_dataset, f)

    final_poses_savepath = f'{fpath.resolve()}/value_datasets/combined_final_poses.pkl'
    final_poses_dataset = combined_final_poses
    with open(final_poses_savepath, 'wb') as f:
        pkl.dump(final_poses_dataset, f)

    succesful_poses_savepath = f'{fpath.resolve()}/initial_poses/successful_initial_poses.pkl'
    succesful_poses = combined_initial_poses[succs]
    with open(succesful_poses_savepath, 'wb') as f:
        pkl.dump(succesful_poses, f)
        