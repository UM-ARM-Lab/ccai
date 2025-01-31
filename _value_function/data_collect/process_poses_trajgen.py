from _value_function.screwdriver_problem import init_env
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
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

if __name__ == "__main__":

    noisy = False
    filenames = []
    if noisy:
        for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("noisy_regrasp_to_turn_dataset*.pkl"):
            filenames.append(file)
    else:
        for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("regrasp_to_turn_dataset*.pkl"):
            filenames.append(file)

    combined_pregrasp_poses = np.empty((0, 20))
    combined_regrasp_poses = np.empty((0, 20))
    combined_regrasp_trajs = np.empty((0, 13, 20))
    combined_turn_trajs = np.empty((0, 13, 20))

    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            pregrasp_poses, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs = zip(*pose_tuples)
            
            # create empty arrays to hold the poses
            succ_pregrasp_poses = np.empty((0, 20))
            succ_regrasp_poses = np.empty((0, 20))
            succ_regrasp_trajs = np.empty((0, 13, 20))
            succ_turn_trajs = np.empty((0, 13, 20))
            
            for i in range(len(pregrasp_poses)):
                cost = calculate_turn_cost(regrasp_poses[i].numpy(), turn_poses[i])
                if cost < 1.0:
                    succ_pregrasp_poses = np.concatenate((succ_pregrasp_poses, pregrasp_poses[i].numpy().reshape(1,20)), axis=0)
                    succ_regrasp_poses = np.concatenate((succ_regrasp_poses, regrasp_poses[i].reshape(1,20)), axis=0)
                    succ_regrasp_trajs = np.concatenate((succ_regrasp_trajs, regrasp_trajs[i].reshape(1,13,20)), axis=0)
                    succ_turn_trajs = np.concatenate((succ_turn_trajs, turn_trajs[i].reshape(1,13,20)), axis=0)

            combined_pregrasp_poses = np.concatenate((combined_pregrasp_poses, succ_pregrasp_poses), axis=0)
            combined_regrasp_poses = np.concatenate((combined_regrasp_poses, succ_regrasp_poses), axis=0)
            combined_regrasp_trajs = np.concatenate((combined_regrasp_trajs, succ_regrasp_trajs), axis=0)
            combined_turn_trajs = np.concatenate((combined_turn_trajs, succ_turn_trajs), axis=0)

    trajgen_dataset = zip(combined_pregrasp_poses, combined_regrasp_poses, combined_regrasp_trajs, combined_turn_trajs)
    
    if noisy:
        trajgen_savepath = f'{fpath.resolve()}/trajgen_datasets/noisy_combined_trajgen_dataset.pkl'
    else:
        trajgen_savepath = f'{fpath.resolve()}/trajgen_datasets/combined_trajgen_dataset.pkl'
   
    pkl.dump(trajgen_dataset, open(trajgen_savepath, 'wb'))

    print("num samples: ", len(combined_pregrasp_poses))