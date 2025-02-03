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

falls = 0

def calculate_turn_cost(initial_pose, final_pose):
    global falls
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
    # upright_cost = 50 * np.sum((state[-3:-1] - np.array([0, 0])) ** 2) 

    ######################### REWARD SHAPING
    # if np.any(state[-3:-1] > 0.3):
    #     upright_cost = 100.0
    #     falls += 1
    # else:
    #     upright_cost = 0.0
    # goal_cost = np.sum(((state[-1] - screwdriver_goal[-1]) ** 2)).reshape(-1)
    # total_cost = np.minimum(goal_cost+upright_cost, 5.0)
    ###########################################################################

    goal_cost = np.sum(((state[-3:] - screwdriver_goal) ** 2)).reshape(-1)
    total_cost = np.minimum(goal_cost, 5.0)

    return total_cost

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

    combined_regrasp_trajs = np.empty((0, 13, 20))
    combined_turn_trajs = np.empty((0, 13, 20))
    combined_turn_costs = []
    combined_regrasp_costs = []

    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    # filenames = filenames[:1]
    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            pregrasp_poses, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs = zip(*pose_tuples)
            
            regrasp_poses = np.array([t.numpy() for t in regrasp_poses]).reshape(-1, 20)
            turn_poses = np.array(turn_poses).reshape(-1, 20)

            if turn_trajs[0].shape[0] != 13:
                print("broken trajectory")
                continue
            
            combined_turn_trajs = np.concatenate((combined_turn_trajs, turn_trajs), axis=0)
            combined_regrasp_trajs = np.concatenate((combined_regrasp_trajs, regrasp_trajs), axis=0)
            
            turn_costs = []
            regrasp_costs = []

            for i in range(len(regrasp_poses)):
                cost = calculate_turn_cost(regrasp_poses[i], turn_poses[i])

                # if cost > 90:
                #     env.reset(torch.from_numpy(turn_poses[i]).reshape(1,20).float())
                #     time.sleep(2.0)

                regrasp_cost = calculate_regrasp_cost(regrasp_trajs[i])
                turn_costs.append(cost)
                regrasp_costs.append(regrasp_cost)
            
            combined_turn_costs.extend(turn_costs)
            combined_regrasp_costs.extend(regrasp_costs)

    combined_turn_costs = np.array(combined_turn_costs)
    combined_regrasp_costs = np.array(combined_regrasp_costs)

    regrasp_cost_dataset = zip(combined_regrasp_trajs, combined_regrasp_costs, combined_turn_trajs, combined_turn_costs)
    
    if noisy:
        regrasp_cost_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/noisy_combined_regrasp_to_turn_dataset.pkl'
    else:
        regrasp_cost_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset.pkl'
   
    pkl.dump(regrasp_cost_dataset, open(regrasp_cost_savepath, 'wb'))

    print("num samples: ", len(combined_regrasp_costs))
    print("num falls: ", falls)
    
    plt.figure(figsize=(10, 6))
    plt.hist(combined_turn_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 0 Costs')
    plt.grid(True)
    plt.show()