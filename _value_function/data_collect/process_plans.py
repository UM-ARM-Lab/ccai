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
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost 


if __name__ == "__main__":

    noisy = False
    filenames = []

    for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("regrasp_to_turn_dataset_narrow_plan*.pkl"):
        filenames.append(file)

    combined_regrasp_trajs = np.empty((0, 13, 20))
    combined_turn_trajs = np.empty((0, 13, 20))
    
    combined_regrasp_plans = np.empty((0, 12, 13, 20))
    combined_turn_plans = np.empty((0, 12, 13, 20))
    
    combined_turn_costs = []

    vis = False
    if vis:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    # filenames = filenames[:1]
    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            _, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs, *extra = zip(*pose_tuples)

            if len(extra) > 0:
                regrasp_plan = extra[0]
                turn_plan = extra[1]
                regrasp_plan = np.empty((0, 12, 13, 20))
                turn_plan = np.empty((0, 12, 13, 20))
                # turn these into the right shape by padding with 0s? or just repeat the last step 13-N times?
            else:
                regrasp_plan = np.empty((0, 12, 13, 20))
                turn_plan = np.empty((0, 12, 13, 20))
            
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

            combined_turn_plans = np.concatenate((combined_turn_plans, turn_plan), axis=0)
            combined_regrasp_plans = np.concatenate((combined_regrasp_plans, regrasp_plan), axis=0)
            
            turn_costs = []
            
            for i in range(len(regrasp_poses)):
                cost = calculate_turn_cost(regrasp_poses[i], turn_poses[i])

                if vis:
                    if cost > 4.0 and cost < 5.0:
                        for j in range(13):
                            env.reset(torch.from_numpy(turn_trajs[i][j]).reshape(1,20).float())
                            time.sleep(.10)
                        time.sleep(2.0)

                turn_costs.append(cost)
            
            combined_turn_costs.extend(turn_costs)

    combined_turn_costs = np.array(combined_turn_costs)

    regrasp_to_turn_dataset = zip(combined_regrasp_trajs, combined_turn_trajs, combined_turn_costs)
    turn_to_turn_dataset = zip(combined_turn_trajs, combined_turn_costs)

    # regrasp_plan_dataset = combined_regrasp_plans
    # turn_plan_dataset = combined_turn_plans
    
    regrasp_to_turn_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset.pkl'
    turn_to_turn_savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/combined_turn_to_turn_dataset.pkl'
   
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