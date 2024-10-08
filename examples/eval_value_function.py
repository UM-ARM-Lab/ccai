import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from train_value_function import Net

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    
    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    with open(f'{fpath.resolve()}/eval_dataset.pkl', 'rb') as file:
        pose_cost_tuples  = pkl.load(file)
    
    checkpoint = torch.load(f'{fpath.resolve()}/value_function.pkl')
    cost_mean = checkpoint['cost_mean']
    cost_std = checkpoint['cost_std']
    poses_mean = checkpoint['poses_mean']
    poses_std = checkpoint['poses_std']

    poses, costs = zip(*pose_cost_tuples)

    poses = np.array([t.numpy() for t in poses]).reshape(-1,20)
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)
    poses_tensor = torch.from_numpy(poses_norm).float()

    costs = np.array(costs).reshape(-1,1)
    costs_norm = (costs - cost_mean) / cost_std
    costs_tensor = torch.from_numpy(costs_norm).float()

    model = Net(poses_tensor.shape[1], 1)
    model.load_state_dict(checkpoint['model_state'])
