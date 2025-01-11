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

filenames = []
for file in Path(f'{fpath.resolve()}/value_datasets').glob("full_task_dataset_*.pkl"):
    filenames.append(file)

def calculate_turn_cost(initial_pose, final_pose):
    turn_angle = np.pi/2

    screwdriver_pose = initial_pose[-4:-1]
    screwdriver_goal = np.array([0, 0, -turn_angle]) + screwdriver_pose
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

    # we're only actually using the screwdriver values
    state = final_pose[-4:-1]
    # upright_cost = 20 * np.sum((state[-3:-1]) ** 2) # the screwdriver should only rotate in z direction
    goal_cost = np.sum((1 * (state[-3:] - screwdriver_goal) ** 2)).reshape(-1)
    total_cost = np.minimum(goal_cost, 5.0)

    return total_cost, succ

def calculate_regrasp_cost(initial_pose, final_pose):

    state = final_pose[-4:-1]
    upright_cost = np.sum((state[-3:-1]) ** 2) # the screwdriver should only rotate in z direction

    return upright_cost

if __name__ == "__main__":

    # _, turn0_final_poses, _, _, turn1_final_poses, _ = zip(*pkl.load(filenames[0].open('rb')))
    combined_pregrasp_poses = np.empty((0, 20))
    combined_turn0_final_poses = np.empty((0, 20))
    combined_turn0_full_trajectories = np.empty((0, 13, 20))
    combined_regrasp_poses = np.empty((0, 20))
    combined_turn1_final_poses = np.empty((0, 20))
    combined_turn1_full_trajectories = np.empty((0, 13, 20))
    both_turn_full_trajectories = np.empty((0, 26, 20))

    combined_turn0_costs = []
    combined_regrasp_costs = []
    combined_turn1_costs = []

    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)
            pregrasp_poses, turn0_final_poses, turn0_full_trajectories, regrasp_poses, turn1_final_poses, turn1_full_trajectories = zip(*pose_tuples)
            
            pregrasp_poses = np.array([t.numpy() for t in pregrasp_poses]).reshape(-1, 20)
            turn0_final_poses = np.array(turn0_final_poses).reshape(-1, 20)
            turn0_full_trajectories = np.array(turn0_full_trajectories).reshape(turn0_final_poses.shape[0], -1, 20)
            regrasp_poses = np.array([t.numpy() for t in regrasp_poses]).reshape(-1, 20)
            turn1_final_poses = np.array(turn1_final_poses).reshape(-1, 20)
            turn1_full_trajectories = np.array(turn1_full_trajectories).reshape(turn1_final_poses.shape[0], -1, 20)
            both_turn_full_trajectories = np.concatenate((turn0_full_trajectories, turn1_full_trajectories), axis=1)
        
            combined_pregrasp_poses = np.concatenate((combined_pregrasp_poses, pregrasp_poses), axis=0)
            combined_turn0_final_poses = np.concatenate((combined_turn0_final_poses, turn0_final_poses), axis=0)
            combined_turn0_full_trajectories = np.concatenate((combined_turn0_full_trajectories, turn0_full_trajectories), axis=0)
            combined_regrasp_poses = np.concatenate((combined_regrasp_poses, regrasp_poses), axis=0)
            combined_turn1_final_poses = np.concatenate((combined_turn1_final_poses, turn1_final_poses), axis=0)
            combined_turn1_full_trajectories = np.concatenate((combined_turn1_full_trajectories, turn1_full_trajectories), axis=0)

            turn0_costs = []
            for i in range(len(pregrasp_poses)):
                cost, succ = calculate_turn_cost(pregrasp_poses[i], turn0_final_poses[i])
                turn0_costs.append(cost)
            
            regrasp_costs = []
            for i in range(len(turn0_final_poses)):
                cost = calculate_regrasp_cost(turn0_final_poses[i], regrasp_poses[i])
                regrasp_costs.append(cost)

            turn1_costs = []
            for i in range(len(regrasp_poses)):
                cost, succ = calculate_turn_cost(regrasp_poses[i], turn1_final_poses[i])
                turn1_costs.append(cost)

            combined_turn0_costs.extend(turn0_costs)
            combined_regrasp_costs.extend(regrasp_costs)
            combined_turn1_costs.extend(turn1_costs)

    combined_turn0_costs = np.array(combined_turn0_costs)
    combined_regrasp_costs = np.array(combined_regrasp_costs)
    combined_turn1_costs = np.array(combined_turn1_costs)

    pregrasp_cost_dataset = zip(combined_pregrasp_poses, combined_turn0_costs, combined_turn0_full_trajectories)
    turn_cost_dataset = zip(combined_turn0_final_poses, combined_regrasp_costs, combined_regrasp_poses)
    regrasp_cost_dataset = zip(combined_regrasp_poses, combined_turn1_costs, combined_turn1_full_trajectories)
    
    pregrasp_cost_savepath = f'{fpath.resolve()}/value_datasets/combined_pregrasp_dataset.pkl'
    turn_cost_savepath = f'{fpath.resolve()}/value_datasets/combined_turn_dataset.pkl'
    regrasp_cost_savepath = f'{fpath.resolve()}/value_datasets/combined_regrasp_dataset.pkl'
    
    
    pkl.dump(pregrasp_cost_dataset, open(pregrasp_cost_savepath, 'wb'))
    pkl.dump(turn_cost_dataset, open(turn_cost_savepath, 'wb'))
    pkl.dump(regrasp_cost_dataset, open(regrasp_cost_savepath, 'wb'))


    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    for trial in both_turn_full_trajectories:
        for traj in trial:
            env.reset(dof_pos = torch.tensor(traj.reshape(1,20)).to(device=config['sim_device']).float())
            time.sleep(1)
        time.sleep(1)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
    plt.figure(figsize=(10, 6))
    plt.hist(combined_turn0_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 0 Costs')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(combined_regrasp_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Regrasp Costs')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(combined_turn1_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 1 Costs')
    plt.grid(True)
    plt.show()


   
