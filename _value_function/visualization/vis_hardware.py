from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
from _value_function.train_value_function_regrasp import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA
import time

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

data = {}
methods = ['vf']#, 'novf']
params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

for method in methods:
    data[method] = {
            'costs': [],
            'tilt_angles': [],
            'yaw_deltas': [],
        }
    for i in range(2):
        file = f'{fpath.resolve()}/test/hardware_results/{method}_trial_{i}.pkl'
        with open(file, 'rb') as file:
            pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj, *initial_samples = pkl.load(file)
            # results[method_name][(pregrasp_index, repeat_index)]\
            cost = calculate_turn_cost(regrasp_pose.numpy(), turn_pose)
            data[method]['costs'].append(cost)

        for j in range(13):
            env.reset(torch.from_numpy(turn_traj[j]).reshape(1, 20).float())
            time.sleep(0.1)

    print(f"{method} costs: {data[method]['costs']}")
    




