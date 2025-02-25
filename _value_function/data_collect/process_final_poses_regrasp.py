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
torch.serialization.default_restore_location = lambda storage, loc: storage.cpu()

falls = 0

def calculate_turn_cost(initial_pose, final_pose):
    global falls
    turn_angle = np.pi/2

    screwdriver_pose = initial_pose.flatten()[-4:-1]

    # account for weird angle wrapping
    if screwdriver_pose[2] < -np.pi:
        screwdriver_pose[2] += 4 * np.pi

    screwdriver_goal = np.array([0, 0, -turn_angle]) + screwdriver_pose
    
    # we're only actually using the screwdriver values
    state = final_pose.flatten()[-4:-1]

    if state[2] < -np.pi:
        state[2] += 4 * np.pi

    ######################### REWARD SHAPING
    # if np.any(state[-3:-1] > 0.3):
    #     upright_cost = 100.0
    #     falls += 1
    # else:
    #     upright_cost = 0.0
    # goal_cost = np.sum(((state[-1] - screwdriver_goal[-1]) ** 2)).reshape(-1)
    # total_cost = np.minimum(goal_cost+upright_cost, 5.0)
    ###########################################################################

    goal_cost = ((state[-3:] - screwdriver_goal) ** 2).flatten()
    goal_cost = sum(goal_cost)

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

    name = ""

    if noisy:
        for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("noisy_regrasp_to_turn_dataset*.pkl"):
            filenames.append(file)
    else:
        for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("regrasp_to_turn_dataset*.pkl"):
            # filenames.append(file)
            if file.name.startswith("regrasp_to_turn_dataset_narrow"):
                filenames.append(file)
            # else:   
            #     print("Skipping narrow dataset")

    combined_regrasp_trajs = np.empty((0, 13, 20))
    combined_turn_trajs = np.empty((0, 13, 20))
    combined_start_yaws = []
    
    combined_turn_costs = []

    vis = False
    if vis:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    # filenames = filenames[:1]
    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            _, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs, *extra = zip(*pose_tuples)
            
            regrasp_poses = np.array([t.numpy() for t in regrasp_poses]).reshape(-1, 20)
            turn_poses = np.array(turn_poses).reshape(-1, 20)

            regrasp_trajs = list(regrasp_trajs)
            turn_trajs = list(turn_trajs)

            # for i in range(turn_trajs[0].shape[0]):
            #     env.reset(torch.from_numpy(turn_trajs[0][i]).reshape(1,20).float())

            original_length = len(turn_trajs)
            # turn_trajs = [traj for traj in turn_trajs if traj.shape[0] == 13]

            for i in range(original_length - 1, -1, -1):
                if turn_trajs[i].shape[0] != 13 or regrasp_trajs[i].shape[0] != 13:
                    print("Broken trajectory removed")

                    regrasp_poses = np.delete(regrasp_poses, i, axis=0)
                    turn_poses = np.delete(turn_poses, i, axis=0)

                    regrasp_trajs.pop(i)
                    turn_trajs.pop(i)
            
            combined_turn_trajs = np.concatenate((combined_turn_trajs, turn_trajs), axis=0)
            combined_regrasp_trajs = np.concatenate((combined_regrasp_trajs, regrasp_trajs), axis=0)
            
            turn_costs = []
            
            for i in range(len(regrasp_poses)):
                cost = calculate_turn_cost(regrasp_poses[i], turn_poses[i])
                combined_start_yaws.append(regrasp_poses[i][-2])

                if vis:
                    if cost > 4.0 and cost < 5.0:
                        for j in range(13):
                            env.reset(torch.from_numpy(turn_trajs[i][j]).reshape(1,20).float())
                            time.sleep(.10)
                        time.sleep(2.0)

                turn_costs.append(cost)
            
            combined_turn_costs.extend(turn_costs)

    combined_turn_costs = np.array(combined_turn_costs)
    combined_start_yaws = np.array(combined_start_yaws)

    if combined_regrasp_trajs.shape[0] > 10000:
        combined_regrasp_trajs = combined_regrasp_trajs[:10000]
        combined_turn_trajs = combined_turn_trajs[:10000]
        combined_turn_costs = combined_turn_costs[:10000]
        combined_start_yaws = combined_start_yaws[:10000]

    regrasp_to_turn_dataset = zip(combined_regrasp_trajs, combined_turn_trajs, combined_turn_costs)
    turn_to_turn_dataset = zip(combined_turn_trajs, combined_turn_costs, combined_start_yaws)
    
    regrasp_to_turn_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset{name}.pkl'
    turn_to_turn_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/combined_turn_to_turn_dataset{name}.pkl'
   
    pkl.dump(regrasp_to_turn_dataset, open(regrasp_to_turn_savepath, 'wb'))
    pkl.dump(turn_to_turn_dataset, open(turn_to_turn_savepath, 'wb'))

    print("num samples: ", len(combined_turn_costs))
    
    plt.figure(figsize=(10, 6))
    plt.hist(combined_turn_costs.flatten(), bins=20, color='blue', label='Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Turn 0 Costs')
    plt.grid(True)
    plt.show()