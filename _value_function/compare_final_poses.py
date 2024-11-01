from screwdriver_problem import init_env
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
experiment_name = '_ensemble_SGD_100k_iters'

# experiment_name = '_mse_50samples'

filename = f'final_pose_comparisons{experiment_name}.pkl'
with open(f'{fpath.resolve()}/eval/{filename}', 'rb') as file:
    tuples = pkl.load(file)
    initial_poses, optimized_poses, initial_final_poses, optimized_final_poses = zip(*tuples)
    initial_final_poses = np.array(initial_final_poses).reshape(-1, 20)
    optimized_final_poses = np.array(optimized_final_poses).reshape(-1, 20)

with open(f'{fpath.resolve()}/initial_poses/initial_poses_10k.pkl', 'rb') as file:
    tenk_poses = pkl.load(file)
    tenk_poses = np.array([tensor.numpy() for tensor in tenk_poses]).reshape(-1, 20)


if __name__ == "__main__":

    initial_costs = []
    optimized_costs = []
    for i in range(len(initial_poses)):
        before, _ = calculate_cost(initial_poses[i].numpy(), initial_final_poses[i])
        initial_costs.append(before)
        after, _ = calculate_cost(optimized_poses[i].numpy(), optimized_final_poses[i])
        optimized_costs.append(after)

    ############################################################
    # Get predicted costs before and after

    checkpoints = torch.load(open(f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl', 'rb'))
    models = []
    shape = (20, 1)
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
    
    for initial_pose, optimized_pose in zip(initial_poses, optimized_poses):
        initial_pose = (initial_pose - poses_mean) / (poses_std + 0.000001)
        initial_pose = initial_pose.float()
        optimized_pose = (optimized_pose - poses_mean) / (poses_std + 0.000001)
        optimized_pose = optimized_pose.float()

        before = query_ensemble(initial_pose, models)
        after = query_ensemble(optimized_pose, models)

        prediction_before = before.mean(dim=0)
        stds_before = before.std(dim=0)  
        prediction_after = after.mean(dim=0)
        stds_after = after.std(dim=0)

        prediction_before = prediction_before * cost_std + cost_mean
        prediction_after = prediction_after * cost_std + cost_mean
        stds_before = stds_before * cost_std
        stds_after = stds_after * cost_std

        predicted_costs_before.append(prediction_before.detach().numpy())
        predicted_stds_before.append(stds_before.detach().numpy())
        predicted_costs_after.append(prediction_after.detach().numpy())
        predicted_stds_after.append(stds_after.detach().numpy())

    ############################################################

    print("Average decrease in real cost:", np.mean(np.array(initial_costs) - np.array(optimized_costs)))
    print("Average decrease in predicted cost:", np.mean(np.array(predicted_costs_before) - np.array(predicted_costs_after)))
    
    plt.figure(figsize=(12, 6))

    # Define offsets for spacing
    real_offset = 0.1  # Slightly shift real costs to the left
    predicted_offset = -real_offset  # Slightly shift predicted costs to the right

    # Limit the values for plotting
    predicted_costs_before = predicted_costs_before[:20]
    predicted_costs_after = predicted_costs_after[:20]
    predicted_stds_before = predicted_stds_before[:20]
    predicted_stds_after = predicted_stds_after[:20]
    initial_costs = initial_costs[:20]
    optimized_costs = optimized_costs[:20]

    # Scatter initial and optimized costs with error bars and offsets
    plt.errorbar(
        np.arange(len(predicted_costs_before)) + predicted_offset, 
        predicted_costs_before, 
        yerr=predicted_stds_before, 
        fmt='o', 
        color='blue', 
        capsize=3,
        label='Predicted Initial Costs'
    )
    plt.errorbar(
        np.arange(len(predicted_costs_after)) + predicted_offset, 
        predicted_costs_after, 
        yerr=predicted_stds_after, 
        fmt='o', 
        color='red', 
        capsize=3,
        label='Predicted Optimized Costs'
    )
    plt.scatter(np.arange(len(initial_costs)) + real_offset, initial_costs, color='blue', marker='x', label='Real Initial Costs')
    plt.scatter(np.arange(len(optimized_costs)) + real_offset, optimized_costs, color='red', marker='x', label='Real Optimized Costs')

    # Draw lines between initial and optimized costs for both real and predicted
    for i in range(len(initial_costs)):
        # Real cost connection (black dotted line)
        plt.plot([i + real_offset, i + real_offset], [initial_costs[i], optimized_costs[i]], 'k--', linewidth=1)  
        # Predicted cost connection (gray dotted line)
        plt.plot([i + predicted_offset, i + predicted_offset], [predicted_costs_before[i], predicted_costs_after[i]], 'k--', linewidth=1)

    # Final plot adjustments
    plt.xlabel('Sample Index')
    plt.ylabel('Cost Value')
    plt.title('Initial and Optimized Costs: Real vs Predicted with Error Bars')
    plt.legend()
    plt.grid(True)
    plt.show()


    vis = False
    if vis:
        params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for i in range(len(initial_poses)):
            print("original turn")
            print("cost: ", initial_costs[i])

            # env.reset(initial_poses[i].reshape(1,20).float(), deterministic=True)
            # time.sleep(0.5)
            # env.reset(torch.from_numpy(initial_final_poses[i]).reshape(1,20).float(), deterministic=True)
            # time.sleep(1.0)

            print("optimized turn")
            print("cost: ", optimized_costs[i])

            # if optimized_costs[i] > 3:

            env.reset(optimized_poses[i].reshape(1,20).float(), deterministic=True)
            time.sleep(.5)
            # env.reset(torch.from_numpy(optimized_final_poses[i]).reshape(1,20).float(), deterministic=True)
            # time.sleep(1.0)



                # print("original pose")
                # env.reset(initial_poses[i].reshape(1,20).float(), deterministic=True)
                # time.sleep(0.5)
                # print("optimized pose")
                # env.reset(optimized_poses[i].reshape(1,20).float(), deterministic=True)
                # time.sleep(1.0)