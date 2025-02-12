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

    dataset_trajs_regrasp = np.empty((0, 12, 30))
    dataset_trajs_turn = np.empty((0, 12, 36))
    total_samples = 0

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
                    # print("Good trial")
                    good_turn_trajs = np.vstack([good_turn_trajs, turn_trajs[i].reshape(1,13,20)])
                    good_regrasp_trajs = np.vstack([good_regrasp_trajs, regrasp_trajs[i].reshape(1,13,20)])

                    good_regrasp_plans.append(regrasp_plan[i])
                    good_turn_plans.append(turn_plan[i])
            ###########################################################

            # Convert to 15 dim and then add time dimension
            n_good_trials = good_regrasp_trajs.shape[0]
            total_samples += n_good_trials

            if n_good_trials == 0:
                continue
            
            else:
                
                actual_regrasp_state_trajs = good_regrasp_trajs[:,1:,:].reshape(-1,20) 
                actual_regrasp_state_trajs = convert_full_to_partial_config(actual_regrasp_state_trajs).reshape(-1,12,15)
                actual_regrasp_trajs = np.empty((n_good_trials, 12, 30))
                

                actual_turn_state_trajs = good_turn_trajs[:,1:,:].reshape(-1,20) 
                actual_turn_state_trajs = convert_full_to_partial_config(actual_turn_state_trajs).reshape(-1,12,15)
                actual_turn_trajs = np.empty((n_good_trials, 12, 36))
                

                for trial in range(len(good_regrasp_plans)):

                    # regrasp

                    regrasp_plans = good_regrasp_plans[trial]
                    actual_action_traj_regrasp = np.empty((12, 30-15))
                    for step in range(12):
                        action = regrasp_plans[step][0,15:]
                        actual_action_traj_regrasp[step] = action
                    actual_traj = np.hstack((actual_regrasp_state_trajs[trial], actual_action_traj_regrasp)).reshape(12, 30)
                    actual_regrasp_trajs[trial] = actual_traj

                    # turn

                    turn_plans = good_turn_plans[trial]
                    actual_action_traj_turn = np.empty((12, 36-15))
                    for step in range(12):
                        action = turn_plans[step][0,15:]
                        actual_action_traj_turn[step] = action
                    actual_traj = np.hstack((actual_turn_state_trajs[trial], actual_action_traj_turn)).reshape(12, 36)
                    actual_turn_trajs[trial] = actual_traj

                    for idx in range(12):

                        # regrasp

                        traj = np.empty((12, 30))
                        traj[:idx] = actual_regrasp_trajs[trial,:idx,:]
                        traj[idx:] = regrasp_plans[idx]
                        dataset_trajs_regrasp = np.vstack((dataset_trajs_regrasp, traj.reshape(1,12,30)))

                        # turn

                        traj = np.empty((12, 36))
                        traj[:idx] = actual_turn_trajs[trial,:idx,:]
                        traj[idx:] = turn_plans[idx]
                        dataset_trajs_turn = np.vstack((dataset_trajs_turn, traj.reshape(1,12,36)))

    ###############################################################################################
    
    # 

    ###############################################################################################

    # pad regrasp trajs with 0s
    padding = np.zeros((dataset_trajs_regrasp.shape[0],12,6))
    dataset_trajs_regrasp = np.concatenate((dataset_trajs_regrasp, padding), axis=2)

    regrasp_plan_savepath = f'{fpath.resolve()}/diffusion_datasets/regrasp_diffusion_dataset.pkl'
    turn_plan_savepath = f'{fpath.resolve()}/diffusion_datasets/turn_diffusion_dataset.pkl'
   
    pkl.dump(dataset_trajs_regrasp, open(regrasp_plan_savepath, 'wb'))
    pkl.dump(dataset_trajs_turn, open(turn_plan_savepath, 'wb'))

    print("num good samples: ", total_samples)