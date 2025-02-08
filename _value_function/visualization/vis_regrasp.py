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

file = f'{fpath.resolve()}/test/regrasp.pkl'
with open(file, 'rb') as file:
    regrasp_trajs = pkl.load(file)

file = f'{fpath.resolve()}/regrasp_to_turn_datasets/regrasp_to_turn_dataset_a_0.pkl'
with open(file, 'rb') as file:
    pregrasp_poses, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs = zip(*pkl.load(file))

params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

for i in range(len(regrasp_trajs)):
    for j in range(12):
        print(j)
        env.reset(torch.tensor(regrasp_trajs[i][j]).reshape(1,20))
        time.sleep(0.1)
        if j == 11 or j==0:
            time.sleep(1.5)
