from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
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
torch.serialization.default_restore_location = lambda storage, loc: storage.cpu()

if __name__ == "__main__":

    filenames = []
    for file in Path(f'{fpath.resolve()}/regrasp_to_turn_datasets').glob("regrasp_to_turn_dataset_narrow_plan*.pkl"):
        filenames.append(file)

    regrasp_pose_inputs = []
    turn_pose_inputs = []
    regrasp_plan_outputs = []
    turn_plan_outputs = []

    for filename in filenames:
        with open(filename, 'rb') as file:
            pose_tuples = pkl.load(file)

            _, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs, regrasp_plan, turn_plan = zip(*pose_tuples)

            regrasp_trajs = list(regrasp_trajs)
            turn_trajs = list(turn_trajs)
            regrasp_plan = list(regrasp_plan)
            turn_plan = list(turn_plan)

            ########### Remove corrupted data ##########

            original_length = len(turn_trajs)
            for i in range(original_length - 1, -1, -1):
                if turn_trajs[i].shape[0] != 13 or regrasp_trajs[i].shape[0] != 13:
                    print("Broken trajectory removed")

                    regrasp_trajs.pop(i)
                    turn_trajs.pop(i)

                    regrasp_plan.pop(i)
                    turn_plan.pop(i)
            
            turn_trajs = np.array(turn_trajs)
            regrasp_trajs = np.array(regrasp_trajs)
            
            ###### Calculate turn costs to filter out bad trials ######

            regrasp_poses = np.array([t.numpy() for t in regrasp_poses]).reshape(-1, 20)
            turn_poses = np.array(turn_poses).reshape(-1, 20)

            good_turn_trajs = np.empty((0, 13, 20))
            good_regrasp_trajs = np.empty((0, 13, 20))
            good_regrasp_plans = []
            good_turn_plans = []


            for i in range(len(regrasp_poses)):
                cost = calculate_turn_cost(regrasp_poses[i], turn_poses[i])

                if cost < 1.0:
                    good_turn_trajs = np.vstack([good_turn_trajs, turn_trajs[i].reshape(1,13,20)])
                    good_regrasp_trajs = np.vstack([good_regrasp_trajs, regrasp_trajs[i].reshape(1,13,20)])

                    good_regrasp_plans.extend(regrasp_plan[i])
                    good_turn_plans.extend(turn_plan[i])
            ###########################################################

            # Convert to 15 dim and then add time dimension

            if good_regrasp_trajs.shape[0] == 0:
                continue
            
            else:
                rg_inputs = good_regrasp_trajs[:,:-1,:].reshape(-1, 20)
                rg_inputs = convert_full_to_partial_config(rg_inputs)
                assert(rg_inputs.shape[0]%12 == 0)
                T_array = np.tile(np.arange(12), rg_inputs.shape[0]//12).reshape(-1, 1)
                rg_inputs = np.hstack([rg_inputs, T_array])
                rg_inputs = list(rg_inputs)
                regrasp_pose_inputs.extend(rg_inputs)

                t_inputs = good_turn_trajs[:,:-1,:].reshape(-1, 20)
                t_inputs = convert_full_to_partial_config(t_inputs)
                assert(t_inputs.shape[0]%12 == 0)
                t_inputs = np.hstack([t_inputs, T_array])
                t_inputs = list(t_inputs)
                turn_pose_inputs.extend(t_inputs)
                
                regrasp_plan_outputs.extend(good_regrasp_plans)
                turn_plan_outputs.extend(good_turn_plans)

    regrasp_plan_dataset = zip(regrasp_pose_inputs, regrasp_plan_outputs)
    turn_plan_dataset = zip(turn_pose_inputs, turn_plan_outputs)

    ###############################################################################################
    
    # The inputs are lists of length equal to the number of low cost trials. 
    # Each element is a pose tensor of shape (16,), where the last index is the time dimension.
    
    # The outputs are lists of length equal to the number of low cost trials. 
    # Each element is a list of length 12, where each element is a tensor of shape (X, L), 
    # where X goes from 12 to 1 as the horizon receds, and L is 30 for regrasp, 36 for turning.

    ###############################################################################################

    regrasp_plan_savepath = f'{fpath.resolve()}/diffusion_datasets/regrasp_diffusion_dataset.pkl'
    turn_plan_savepath = f'{fpath.resolve()}/diffusion_datasets/turn_diffusion_dataset.pkl'
   
    pkl.dump(regrasp_plan_dataset, open(regrasp_plan_savepath, 'wb'))
    pkl.dump(turn_plan_dataset, open(turn_plan_savepath, 'wb'))

    print("num good samples: ", len(regrasp_pose_inputs)//12)