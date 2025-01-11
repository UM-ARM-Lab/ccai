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



if __name__ == "__main__":

    combined_initial_poses = np.empty((0, 20))
    combined_final_poses = np.empty((0, 20))
    combined_costs = []
    succs = []
    fails = []

    filename = f'{fpath.resolve()}/pregrasp_to_turn_datasets/_old_data'
    with open(filename, 'rb') as file:
        pose_tuples = pkl.load(file)
        initial_poses, final_poses = zip(*pose_tuples)
        initial_poses = np.array([t.numpy() for t in initial_poses]).reshape(-1, 20)
        final_poses = np.array(final_poses).reshape(-1, 20)
        
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

    pose_cost_dataset = zip(combined_initial_poses, combined_costs)
    pose_cost_savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/_old_data_dataset.pkl'
    with open(pose_cost_savepath, 'wb') as f:
        pkl.dump(pose_cost_dataset, f)