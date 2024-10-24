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

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')


filenames = []
#for i in range(10000/200):
for i in [1,2,3,4,5,7,8,9,10,11]:
    filename = f'/value_datasets/value_dataset_{i*500}.pkl'
    filenames.append(filename)

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
    state = screwdriver_pose
    upright_cost = 20 * np.sum((state[-3:-1]) ** 2) # the screwdriver should only rotate in z direction
    goal_cost = np.sum((1 * (state[-3:] - screwdriver_goal) ** 2)).reshape(-1)

    return goal_cost + upright_cost, succ

if __name__ == "__main__":

    combined_initial_poses = np.empty((0, 20))
    combined_final_poses = np.empty((0, 20))
    combined_costs = []
    succs = []
    fails = []
    for filename in filenames:
        with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
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
    #print(np.mean(combined_costs))

    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    # for idx in fails:#succs:
    #     env.reset(torch.from_numpy(combined_initial_poses[idx]).reshape(1,20).float(), deterministic=True)
    #     time.sleep(0.5)
    #     env.reset(torch.from_numpy(combined_final_poses[idx]).reshape(1,20).float(), deterministic=True)
    #     time.sleep(1.0)

    pose_cost_dataset = zip(combined_initial_poses, combined_costs)
    pose_cost_savepath = f'{fpath.resolve()}//value_datasetscombined_value_dataset.pkl'
    with open(pose_cost_savepath, 'wb') as f:
        pkl.dump(pose_cost_dataset, f)

    final_poses_savepath = f'{fpath.resolve()}//value_datasetscombined_final_poses.pkl'
    final_poses_dataset = combined_final_poses
    with open(final_poses_savepath, 'wb') as f:
        pkl.dump(final_poses_dataset, f)

    succesful_poses_savepath = f'{fpath.resolve()}/initial_poses/successful_initial_poses.pkl'
    succesful_poses = combined_initial_poses[succs]
    with open(succesful_poses_savepath, 'wb') as f:
        pkl.dump(succesful_poses, f)
        