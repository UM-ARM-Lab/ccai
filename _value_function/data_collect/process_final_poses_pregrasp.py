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

noisy = False
filenames = []
if noisy:
    for file in Path(f'{fpath.resolve()}/pregrasp_to_turn_datasets').glob("noisy_pregrasp_to_turn_dataset*.pkl"):
        filenames.append(file)
else:
    for file in Path(f'{fpath.resolve()}/pregrasp_to_turn_datasets').glob("pregrasp_to_turn_dataset*.pkl"):
        filenames.append(file)

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
    if final_distance_to_goal < 30 / 180 * np.pi:
        succ = True
    else:
        succ = False

    # we're only actually using the screwdriver values
    state = final_pose.flatten()[-4:-1]
    # upright_cost = 20 * np.sum((state[-3:-1]) ** 2) # the screwdriver should only rotate in z direction
    goal_cost = np.sum((1 * (state[-3:] - screwdriver_goal) ** 2)).reshape(-1)
    total_cost = np.minimum(goal_cost, 5.0)

    return total_cost, succ

if __name__ == "__main__":

    combined_pregrasp_poses = np.empty((0, 20))
    combined_turn_final_poses = np.empty((0, 20))
    combined_turn_full_trajectories = np.empty((0, 13, 20))
    combined_succ_pregrasp_tuples = []
    combined_turn_costs = []

    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)
            pregrasp_poses, turn_final_poses, turn_full_trajectories = zip(*pose_tuples)
            
            pregrasp_poses = np.array([t.numpy() for t in pregrasp_poses]).reshape(-1, 20)
            turn_final_poses = np.array(turn_final_poses).reshape(-1, 20)
            
            # pad because some trajectories are shorter than 13
            processed_trajectories = []
            for traj in turn_full_trajectories:
                if traj.shape[0] < 13:
                    rows_to_add = 13 - traj.shape[0]
                    last_row = traj[-1]
                    padding = np.tile(last_row, (rows_to_add, 1))
                    padded_traj = np.vstack((traj, padding))
                    processed_trajectories.append(padded_traj)
                else:
                    processed_trajectories.append(traj)

            turn_full_trajectories= np.array(processed_trajectories).reshape(turn_final_poses.shape[0], -1, 20)
            # turn_full_trajectories = np.array(turn_full_trajectories).reshape(turn_final_poses.shape[0], -1, 20)
            
            combined_pregrasp_poses = np.concatenate((combined_pregrasp_poses, pregrasp_poses), axis=0)
            combined_turn_final_poses = np.concatenate((combined_turn_final_poses, turn_final_poses), axis=0)
            combined_turn_full_trajectories = np.concatenate((combined_turn_full_trajectories, turn_full_trajectories), axis=0)
            
            turn_costs = []
            for i in range(len(pregrasp_poses)):
                cost, succ = calculate_turn_cost(pregrasp_poses[i], turn_final_poses[i])
                turn_costs.append(cost)
                if succ:
                    combined_succ_pregrasp_tuples.append((pregrasp_poses[i].reshape(1, 20), turn_final_poses[i].reshape(1, 20)))
            
            combined_turn_costs.extend(turn_costs)

    combined_turn_costs = np.array(combined_turn_costs)

    pregrasp_cost_dataset = zip(combined_pregrasp_poses, combined_turn_costs, combined_turn_full_trajectories)
    
    if noisy:
        pregrasp_cost_savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/noisy_combined_pregrasp_to_turn_dataset.pkl'
        succ_pregrasp_savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/noisy_pregrasp_to_turn_succ_dataset.pkl'
    else:
        pregrasp_cost_savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/combined_pregrasp_to_turn_dataset.pkl'
        succ_pregrasp_savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_succ_dataset.pkl'
   
    pkl.dump(pregrasp_cost_dataset, open(pregrasp_cost_savepath, 'wb'))
    pkl.dump(combined_succ_pregrasp_tuples, open(succ_pregrasp_savepath, 'wb'))

    print("num successes: ", len(combined_succ_pregrasp_tuples), "num total: ", len(combined_turn_costs))
    # exit()

    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    # for i in range(combined_turn_full_trajectories.shape[0]):
    #     trial = combined_turn_full_trajectories[i]
    #     if not calculate_turn_cost(combined_pregrasp_poses[i], combined_turn_final_poses[i])[1]:
    #         for traj in trial:
    #             env.reset(dof_pos = torch.tensor(traj.reshape(1,20)).to(device=config['sim_device']).float())
    #             time.sleep(0.01)
    #         time.sleep(0.2)
    # gym.destroy_viewer(viewer)
    # gym.destroy_sim(sim)
    
    plt.figure(figsize=(10, 6))
    plt.hist(combined_turn_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 0 Costs')
    plt.grid(True)
    plt.show()