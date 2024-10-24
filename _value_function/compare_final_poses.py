from process_final_poses import calculate_cost
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

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')


with open(f'{fpath.resolve()}/eval/final_pose_comparisons_mse2.pkl', 'rb') as file:
    tuples = pkl.load(file)
    initial_poses, initial_final_poses, optimized_final_poses = zip(*tuples)
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
        after, _ = calculate_cost(initial_poses[i].numpy(), optimized_final_poses[i])
        optimized_costs.append(after)
    
    plt.figure(figsize=(10, 6))
    # Scatter costs
    plt.scatter(range(len(initial_costs)), initial_costs, color='blue', label='Initial Costs')
    plt.scatter(range(len(optimized_costs)), optimized_costs, color='red', label='Optimized Costs')
    # Draw dotted lines
    for i in range(len(initial_costs)):
        plt.plot([i, i], [initial_costs[i], optimized_costs[i]], 'k--')  # Black dotted line

    plt.xlabel('Sample Index')
    plt.ylabel('Cost Value')
    plt.title('Initial Costs vs Optimized Costs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # pca = PCA(n_components=3)
    # pca.fit(tenk_poses)

    # #initial_final_poses_pca = pca.transform(initial_final_poses)
    # initial_final_poses_pca = pca.transform(tenk_poses)

    # optimized_final_poses_pca = pca.transform(optimized_final_poses)

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # Initial poses
    # ax.scatter(initial_final_poses_pca[:, 0], initial_final_poses_pca[:, 1], initial_final_poses_pca[:, 2], color='blue', label='Initial Poses')
    # # Optimized poses
    # #ax.scatter(optimized_final_poses_pca[:, 0], optimized_final_poses_pca[:, 1], optimized_final_poses_pca[:, 2], color='red', label='Optimized Poses')

    # # Labels and legend
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # ax.set_title('PCA of Initial and Optimized Final Poses')
    # ax.legend()

    # plt.show()
