from screwdriver_problem import init_env
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from train_value_function import Net
import time

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
shape = (20, 1)

def get_data():
    filename = '/initial_poses/initial_poses_10k.pkl'
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        poses  = pkl.load(file)
    inputs = np.array([t.numpy() for t in poses]).reshape(-1, 20)
    inputs = torch.from_numpy(inputs)[300:305]
    
    succ_filename = '/initial_poses/successful_initial_poses.pkl'

    with open(f'{fpath.resolve()}/{succ_filename}', 'rb') as file:
        succ_poses = pkl.load(file)

    succs = np.array(succ_poses).reshape(-1,20)
    succ_mean = np.mean(succs, axis=0)
    succ_mean = torch.tensor(succ_mean, dtype=torch.float32)

    return inputs, succ_mean

def grad_descent():
    torch.manual_seed(42)
    model_name = "2"
    model = Net(shape[0], shape[1])
    checkpoint = torch.load(f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
    model.load_state_dict(checkpoint['model_state'])
    poses_mean = checkpoint['poses_mean']
    poses_std = checkpoint['poses_std']
    cost_mean = checkpoint['cost_mean']
    cost_std = checkpoint['cost_std']
    
    poses, target_mean = get_data()
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)
    poses_norm = poses_norm.float()
    original_costs_norm = model(poses_norm).detach().cpu().numpy()
    original_costs = original_costs_norm * cost_std + cost_mean

    # Enable gradient computation
    poses_norm.requires_grad_(True)

    optimizer = optim.Adam([poses_norm], lr=1e-2)
    mse_loss = nn.MSELoss()
    model.eval()

    # gradient descent
    num_iterations = 500
    target_value = torch.tensor([0.0], dtype=torch.float32)

    for i in range(num_iterations):
        optimizer.zero_grad()  
        predictions = model(poses_norm)
        predictions = predictions * cost_std + cost_mean

        mse = mse_loss(predictions, target_value.expand_as(predictions))
        mean_loss = torch.mean((poses_norm - target_mean.unsqueeze(0)) ** 2, dim=1)  
        mean_loss = torch.mean(mean_loss)  # maybe this should relate to variance
        #print("mean: ", mean_loss)
        #print("mse: ", mse)

        loss = mse #+ 0.1 * mean_loss
        loss.backward()

        # Set gradients of the last four values of each pose to 0
        poses_norm.grad[:, -4:] = 0

        optimizer.step()

        if i % 10 == 0:
        #    print(f"Iteration {i}: Loss = {loss.item()}")
            pass

    optimized_poses_norm = poses_norm.detach().cpu().numpy()
    optimized_costs_norm = model(poses_norm).detach().cpu().numpy()
    optimized_poses = optimized_poses_norm * poses_std + poses_mean
    optimized_costs = optimized_costs_norm * cost_std + cost_mean

    initial_pose_tuples = [(initial, optimized) for initial, optimized in zip(poses.numpy(), optimized_poses)]
    predicted_cost_tuples = [(initial, optimized) for initial, optimized in zip(original_costs, optimized_costs)]

    print(original_costs)
    print(optimized_costs)

    vis = False
    if vis:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for initial, optimized in initial_pose_tuples:
            env.reset(torch.from_numpy(initial).reshape(1,20), deterministic=True)
            time.sleep(0.5)
            env.reset(torch.from_numpy(optimized).reshape(1,20).float(), deterministic=True)
            time.sleep(1.0)


    #print(predicted_cost_tuples)
    print(np.mean(np.abs((poses.numpy() -  optimized_poses))))

    output_filename = f'{fpath.resolve()}/eval/initial_and_optimized_poses.pkl'
    with open(output_filename, 'wb') as f:
        pkl.dump(initial_pose_tuples, f)

    return poses.numpy(), optimized_poses


if __name__ == "__main__":
    poses, optimized_poses = grad_descent()
    poses = poses[:,:-4]
    optimized_poses = optimized_poses[:,:-4]

    # Fit PCA to reduce dimensions to 3D
    pca = PCA(n_components=3)
    poses_pca = pca.fit_transform(poses)
    optimized_poses_pca = pca.transform(optimized_poses)  # Apply the same PCA to optimized_poses

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot poses
    ax.scatter(poses_pca[:, 0], poses_pca[:, 1], poses_pca[:, 2], c='b', label='Poses')
    # Plot optimized poses
    ax.scatter(optimized_poses_pca[:, 0], optimized_poses_pca[:, 1], optimized_poses_pca[:, 2], c='r', label='Optimized Poses')

    # Draw dotted lines connecting corresponding points
    for i in range(len(poses_pca)):
        ax.plot([poses_pca[i, 0], optimized_poses_pca[i, 0]],
                [poses_pca[i, 1], optimized_poses_pca[i, 1]],
                [poses_pca[i, 2], optimized_poses_pca[i, 2]],
                'k--')  # Dotted line

    # Add labels and legend
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()

    plt.show()