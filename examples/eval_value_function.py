import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from train_value_function import Net
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

if __name__ == "__main__":

    # Example 20-dimensional dataset (replace with your actual data)
    # Assume 'data' is a NumPy array of shape (n_samples, 20)
    filenames = [
        'value_dataset_100_random.pkl',
        'value_dataset_9500.pkl'
    ]
    validation_proportion = 0.1
    poses = []
    costs = []
    for filename in filenames:
        with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
            pose_cost_tuples  = pkl.load(file)
            new_poses, new_costs = zip(*pose_cost_tuples)
            poses.extend(new_poses)
            costs.extend(new_costs)
    poses = np.array([t.numpy() for t in poses]).reshape(-1,20)
    costs = np.array(costs).reshape(-1,1)

    # Step 1: Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(poses)
    # Step 2: Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)
    # Step 3: Visualize the reduced data in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='b', marker='o')

    # Label axes
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA of Initial States')

    plt.show()

    # Plot KDE
    plt.figure(figsize=(8, 6))
    sns.kdeplot(costs, color='b')
    plt.title('Cost KDE Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()







    # fpath = pathlib.Path(f'{CCAI_PATH}/data')
    # with open(f'{fpath.resolve()}/eval_dataset.pkl', 'rb') as file:
    #     pose_cost_tuples  = pkl.load(file)
    
    # checkpoint = torch.load(f'{fpath.resolve()}/value_function.pkl')
    # cost_mean = checkpoint['cost_mean']
    # cost_std = checkpoint['cost_std']
    # poses_mean = checkpoint['poses_mean']
    # poses_std = checkpoint['poses_std']

    # poses, costs = zip(*pose_cost_tuples)

    # poses = np.array([t.numpy() for t in poses]).reshape(-1,20)
    # poses_norm = (poses - poses_mean) / (poses_std + 0.000001)
    # poses_tensor = torch.from_numpy(poses_norm).float()

    # costs = np.array(costs).reshape(-1,1)
    # costs_norm = (costs - cost_mean) / cost_std
    # costs_tensor = torch.from_numpy(costs_norm).float()

    # model = Net(poses_tensor.shape[1], 1)
    # model.load_state_dict(checkpoint['model_state'])
