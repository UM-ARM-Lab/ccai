from card.card_problem import init_env
from card.train_vf_card_index import Net, query_ensemble, load_ensemble, stack_trajs
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
# shape = (15, 1)

filename = 'card_datasets/combined_index_dataset.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    tuples = pkl.load(file)
    trajs1, trajs2, costs, start_ys = zip(*tuples)

start_ys = np.array(start_ys).flatten()
index_poses = stack_trajs(trajs1)
start_ys = start_ys.repeat(9, axis=0)
index_poses = np.concatenate([index_poses, np.array(start_ys).reshape(-1, 1)], axis=1)

x = 30#10
which_one = 18*x + 4 
index_poses = index_poses[which_one,:]

def grad_descent(lr = 0.01, iters = 1000):
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="index_vf")
    
    poses = torch.from_numpy(index_poses)

    poses_norm = (poses - poses_mean) / poses_std
    poses_norm = poses_norm.float()

    # column_to_add = torch.full((poses_norm.size(0), 1), which_one)
    column_to_add = torch.tensor([which_one % 9], dtype=torch.float32).reshape(1)
    poses_norm = torch.cat((poses_norm, column_to_add), dim=0).reshape(1,-1)

    original_costs_norm = torch.mean(query_ensemble(poses_norm, models), dim=0).detach().cpu().numpy()
    original_costs = original_costs_norm * cost_std + cost_mean

    poses_norm.requires_grad_(True)

    optimizer = optim.SGD([poses_norm], lr=lr)
    iterations = iters

    for model in models:
        model.eval()

    # gradient descent
    num_iterations = iterations
    target_value = torch.tensor([0.0], dtype=torch.float32)

    pose_optimization_trajectory = []
    
    for i in range(num_iterations):
        optimizer.zero_grad()  
        ensemble_predictions = query_ensemble(poses_norm, models)
        ensemble_predictions = ensemble_predictions * cost_std + cost_mean
        predictions = torch.mean(ensemble_predictions, dim=0)

        mse = torch.mean((predictions - target_value) ** 2)
        mean_squared_variance = torch.mean((ensemble_predictions - predictions) ** 2)
        variance_weight = 5
        loss = mse + mean_squared_variance * variance_weight
        loss.backward()

        # Set gradients of the last 5 values of each pose to 0
        poses_norm.grad[:, -5:] = 0
        
        optimizer.step()

        if i % 20 == 0:
            print(f"Iteration {i}: mse = {mse.item()}, variance_loss = {mean_squared_variance.item()}")
            #print(f"Iteration {i}: Loss = {loss.item()}")
            pass
        n_setpoints = 10
        if i % int(num_iterations/n_setpoints) == 0:
            pose_optimization_trajectory.append(poses_norm.detach().cpu().numpy()[:,:-1] * poses_std + poses_mean)

    optimized_poses_norm = poses_norm.detach().cpu().numpy()[:,:-1]
    optimized_costs_norm = torch.mean(query_ensemble(poses_norm, models), dim=0).detach().cpu().numpy()
    optimized_poses = optimized_poses_norm * poses_std + poses_mean
    optimized_costs = optimized_costs_norm * cost_std + cost_mean

    print("mean of new predicted costs: ", optimized_costs.mean())
    print("original costs: ", original_costs)
    print("optimized costs: ", optimized_costs)

    return poses.numpy(), optimized_poses, pose_optimization_trajectory, optimized_costs.mean()

def convert_partial_to_full_config(partial_config):

    partial_config = partial_config.clone().float().reshape(-1)[:-1]
    full_config = torch.cat((
    partial_config[:8].reshape(1,-1),
    torch.tensor([[0, 0.2, 0.3, 0.2]]),
    torch.tensor([[1.2, 0.3, 0.0, 0.8]]),
    torch.tensor([[partial_config[8], partial_config[9], 0.0, 0.0, 0.0, partial_config[10]]])
    ), dim=1)

    return full_config.reshape(-1,22)

if __name__ == "__main__":

    poses, optimized_poses, semi_optimized_poses, _ = grad_descent()
    poses = torch.from_numpy(poses)
    optimized_poses = torch.from_numpy(optimized_poses)
    semi_optimized_poses = [torch.from_numpy(poses).reshape(-1) for poses in semi_optimized_poses]

    full_poses = convert_partial_to_full_config(poses)
    full_optimized_poses = convert_partial_to_full_config(optimized_poses)
    full_semi_optimized_poses = [convert_partial_to_full_config(poses) for poses in semi_optimized_poses]

    full_semi_optimized_poses = [poses.detach().cpu().numpy() for poses in full_semi_optimized_poses]
    semi_optimized_poses = [poses.detach().cpu().numpy() for poses in semi_optimized_poses]
    full_semi_optimized_poses = np.array(full_semi_optimized_poses)
    semi_optimized_poses = np.array(semi_optimized_poses)
    
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer = init_env(visualize=True)

    while True:
        print("before")
        for i in range(full_optimized_poses.shape[0]):
            env.reset(full_poses[i].reshape(1,22).float())
            time.sleep(0.5)
        print("after")
        for i in range(full_optimized_poses.shape[0]):
            env.reset(full_optimized_poses[i].reshape(1,22).float())
            time.sleep(0.5)


    for j in range(full_semi_optimized_poses.shape[1]):
        for i in range(full_semi_optimized_poses.shape[0]):
            env.reset(torch.from_numpy(full_semi_optimized_poses[i,j,:]).reshape(1,20).float())
            if i == 0 or i == full_semi_optimized_poses.shape[0]-1:
                time.sleep(1.0)
            print(f'Setpoint {i}')
            time.sleep(0.03)

    pca = False
    if pca:
        # Fit PCA on the 10k_poses data to get axes
        pca = PCA(n_components=3)
        pca.fit(regrasp_poses)

        # Transform poses and optimized_poses based on the PCA axes from 10k_poses
        poses_pca = pca.transform(poses)
        optimized_poses_pca = pca.transform(optimized_poses)

        # Transform semi_optimized_poses based on the PCA axes
        semi_optimized_poses_pca = [pca.transform(np.array(soposes)) for soposes in semi_optimized_poses]

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plot_intermediate = True
        for i in range(len(semi_optimized_poses_pca)):
            if i == 0:
                color = 'r'
                label = 'Initial Pose'
            elif i == len(semi_optimized_poses_pca) - 1:
                color = 'b'
                label = 'Optimized Pose'
            else:
                color = 'g'
                label = None
                if not plot_intermediate:
                    continue
            # Add label only once for each type
            ax.scatter(semi_optimized_poses_pca[i][:, 0], 
                    semi_optimized_poses_pca[i][:, 1], 
                    semi_optimized_poses_pca[i][:, 2], 
                    c=color, label=label if i in [0, len(semi_optimized_poses_pca) - 1] or plot_intermediate else "")

        # Draw dotted lines connecting initial -> semi_optimized -> optimized for each point
        for i in range(len(semi_optimized_poses_pca[0])):
            
            # Start with the initial pose
            line_points = [poses_pca[i]]
            # Append each semi-optimized pose
            for j in range(len(semi_optimized_poses_pca)):
                if j != 0 and j != len(semi_optimized_poses_pca) - 1 and not plot_intermediate:
                    continue
                line_points.append(semi_optimized_poses_pca[j][i])
            # Add the optimized pose at the end
            line_points.append(optimized_poses_pca[i])
            
            # Convert line points to x, y, z for plotting
            line_points = np.array(line_points)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'k--', label='Path' if i == 0 else "")

        # Add labels and legend
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        ax.legend(loc='best')

        plt.show()