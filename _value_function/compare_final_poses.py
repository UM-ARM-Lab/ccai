from screwdriver_problem import init_env, convert_full_to_partial_config
from process_final_poses import calculate_cost
from train_value_function import Net, query_ensemble
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

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

# experiment_name = '_single_SGD_1k_iters'
# experiment_name = '_single_SGD_10k_iters'
# experiment_name = '_single_Adam_1k_iters'

# experiment_name = '_ensemble_SGD_10k_iters'
# experiment_name = '_ensemble_SGD_1k_iters'
# experiment_name = '_ensemble_Adam_1k_iters'
# experiment_name = '_ensemble_Adam_100k_iters'
experiment_name = '_ensemble_Adam_500_iters_optimal'

filename = f'final_pose_comparisons{experiment_name}.pkl'
with open(f'{fpath.resolve()}/eval/{filename}', 'rb') as file:
    tuples = pkl.load(file)
    # initial_poses_full, optimized_poses_full, initial_final_poses, optimized_final_poses = zip(*[(t[0], t[1], t[2], t[3]) for t in tuples])
    initial_poses_full, optimized_poses_full,\
    initial_final_poses, optimized_final_poses,\
    initial_trajectories, optimized_trajectories = zip(*tuples)

    initial_final_poses = np.array(initial_final_poses).reshape(-1, 20)
    optimized_final_poses = np.array(optimized_final_poses).reshape(-1, 20)

with open(f'{fpath.resolve()}/initial_poses/initial_poses_10k.pkl', 'rb') as file:
    tenk_poses = pkl.load(file)
    tenk_poses = np.array([tensor.numpy() for tensor in tenk_poses]).reshape(-1, 20)


if __name__ == "__main__":

    ############################################################
    # Get real costs before and after

    initial_costs = []
    optimized_costs = []
    for i in range(len(initial_poses_full)):
        before, _ = calculate_cost(initial_poses_full[i].numpy(), initial_final_poses[i])
        initial_costs.append(before)
        after, _ = calculate_cost(optimized_poses_full[i].numpy(), optimized_final_poses[i])
        optimized_costs.append(after)

    # print(initial_costs)
    # print(optimized_costs)
    # exit()
    ############################################################
    # Get predicted costs before and after
    checkpoints = torch.load(open(f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl', 'rb'))
    models = []
    shape = (15, 1)
    for checkpoint in checkpoints:
        model = Net(shape[0], shape[1])
        model.load_state_dict(checkpoint['model_state'])
        models.append(model)
    poses_mean = checkpoints[0]['poses_mean']
    poses_std = checkpoints[0]['poses_std']
    cost_mean = checkpoints[0]['cost_mean']
    cost_std = checkpoints[0]['cost_std']
    min_std_threshold = 1e-5
    poses_std = np.where(poses_std < min_std_threshold, min_std_threshold, poses_std)

    predicted_costs_before = []
    predicted_costs_after = []
    predicted_stds_before = []
    predicted_stds_after = []
    
    for initial_pose_full, optimized_pose_full in zip(initial_poses_full, optimized_poses_full):

        initial_pose = convert_full_to_partial_config(initial_pose_full.reshape(1,20))
        optimized_pose = convert_full_to_partial_config(optimized_pose_full.reshape(1,20))
        initial_pose_norm = (initial_pose - poses_mean) / poses_std
        initial_pose_norm = initial_pose_norm.float()
        optimized_pose_norm = (optimized_pose - poses_mean) / poses_std
        optimized_pose_norm = optimized_pose_norm.float()


        before = query_ensemble(initial_pose_norm, models)
        after = query_ensemble(optimized_pose_norm, models)

        prediction_before_norm = before.mean(dim=0)
        stds_before_norm = before.std(dim=0)  
        prediction_after_norm = after.mean(dim=0)
        stds_after_norm = after.std(dim=0)

        prediction_before = prediction_before_norm * cost_std + cost_mean
        prediction_after = prediction_after_norm * cost_std + cost_mean
        stds_before = stds_before_norm * cost_std
        stds_after = stds_after_norm * cost_std

        predicted_costs_before.append(prediction_before.detach().numpy())
        predicted_stds_before.append(stds_before.detach().numpy())
        predicted_costs_after.append(prediction_after.detach().numpy())
        predicted_stds_after.append(stds_after.detach().numpy())

    ############################################################

    print("Average decrease in real cost:", np.mean(np.array(initial_costs) - np.array(optimized_costs)))
    print("Average decrease in predicted cost:", np.mean(np.array(predicted_costs_before) - np.array(predicted_costs_after)))
    
    plt.figure(figsize=(12, 6))

    # Define offsets for spacing
    real_offset = 0.1  
    predicted_offset = - real_offset  

    # Limit the values for plotting
    n_plot = 20
    predicted_costs_before_plot = predicted_costs_before[:n_plot]
    predicted_costs_after_plot = predicted_costs_after[:n_plot]
    predicted_stds_before_plot = predicted_stds_before[:n_plot]
    predicted_stds_after_plot = predicted_stds_after[:n_plot]
    initial_costs_plot = initial_costs[:n_plot]
    optimized_costs_plot = optimized_costs[:n_plot]

    # Scatter initial and optimized costs with error bars and offsets
    plt.errorbar(
        np.arange(len(predicted_costs_before_plot)) + predicted_offset, 
        predicted_costs_before_plot, 
        yerr=predicted_stds_before_plot, 
        fmt='o', 
        color='blue', 
        capsize=3,
        label='Predicted Initial Costs'
    )
    plt.errorbar(
        np.arange(len(predicted_costs_after_plot)) + predicted_offset, 
        predicted_costs_after_plot, 
        yerr=predicted_stds_after_plot, 
        fmt='o', 
        color='red', 
        capsize=3,
        label='Predicted Optimized Costs'
    )
    plt.scatter(np.arange(len(initial_costs_plot)) + real_offset, initial_costs_plot, color='blue', marker='x', label='Real Initial Costs')
    plt.scatter(np.arange(len(optimized_costs_plot)) + real_offset, optimized_costs_plot, color='red', marker='x', label='Real Optimized Costs')

    # Draw lines between initial and optimized costs for both real and predicted
    for i in range(len(initial_costs_plot)):
        # Real cost connection (black dotted line)
        plt.plot([i + real_offset, i + real_offset], [initial_costs_plot[i], optimized_costs_plot[i]], 'k--', linewidth=1)  
        # Predicted cost connection (gray dotted line)
        plt.plot([i + predicted_offset, i + predicted_offset], [predicted_costs_before_plot[i], predicted_costs_after_plot[i]], 'k--', linewidth=1)

    # Final plot adjustments
    plt.xlabel('Sample Index')
    plt.ylabel('Cost Value')
    plt.title('Initial and Optimized Costs: Real vs Predicted with Error Bars')
    plt.legend()

    xticks = np.arange(-0.5, n_plot, 1.0)

    # Set the gridlines at the specified positions
    plt.gca().set_xticks(xticks, minor=False)
    plt.gca().set_xticks([], minor=True)  # Disable minor ticks if any
    plt.grid(True, which='major', axis='x')

    plt.show()

    failures = []
    vis = True
    if vis:
        params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for i in range(len(initial_trajectories)):
        # for i in range(5):

            if optimized_costs[i] > 3:
                failure = torch.from_numpy(convert_full_to_partial_config(initial_trajectories[i][0].reshape(1,20)))
                failures.append(failure)

                env.reset(torch.from_numpy(optimized_trajectories[i][0]).reshape(1,20).float())
                time.sleep(2.0)

                # print("initial")
                # for j in range(len(initial_trajectories[i])):
                #     env.reset(torch.from_numpy(initial_trajectories[i][j]).reshape(1,20).float())
                #     time.sleep(0.1)
                # time.sleep(1.0)
                # print("optimized")
                # for j in range(len(optimized_trajectories[i])):
                #     env.reset(torch.from_numpy(optimized_trajectories[i][j]).reshape(1,20).float())
                #     time.sleep(0.1)
                # time.sleep(1.0)

        pkl.dump(failures, open(f'{fpath}/eval/failures.pkl', 'wb'))