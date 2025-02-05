from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config, convert_partial_to_full_config
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
# shape = (15, 1)

filename = '/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    pose_cost_tuples = pkl.load(file)
    regrasp_trajs, regrasp_costs, turn_trajs, turn_costs = zip(*pose_cost_tuples)

# regrasp_stacked = np.stack(regrasp_trajs, axis=0)
# regrasp_poses = regrasp_stacked[:,-1,:]
# regrasp_poses = regrasp_poses.reshape(-1, 20)[:100,:]

# config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
# for i in range(regrasp_poses.shape[0]):
#     env.reset(torch.from_numpy(regrasp_poses[i]).reshape(1,20).float())
#     # if i == 0 or i == regrasp_poses.shape[0]-1:
#     #     time.sleep(1.0)
#     print(f'Setpoint {i}')
#     time.sleep(0.3)

# exit()

regrasp_stacked = np.stack(regrasp_trajs, axis=0)

regrasp_poses = regrasp_stacked[:,-1,:]
regrasp_poses = regrasp_poses.reshape(-1, 20)[0:10,:]
regrasp_poses = convert_full_to_partial_config(regrasp_poses)

regrasp_traj = regrasp_stacked[0,:,:].reshape(13,20)


def grad_descent(lr = 0.3, iters = 1000):
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")
    
    poses = torch.from_numpy(regrasp_poses)

    poses_norm = (poses - poses_mean) / poses_std
    poses_norm = poses_norm.float()

    column_to_add = torch.full((poses_norm.size(0), 1), 12)
    poses_norm = torch.cat((poses_norm, column_to_add), dim=1)

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
        variance_weight = 5.0
        loss = mse + mean_squared_variance * variance_weight
        loss.backward()

        # Set gradients of the last four values of each pose to 0
        poses_norm.grad[:, -4:] = 0

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


if __name__ == "__main__":

    poses, optimized_poses, semi_optimized_poses, _ = grad_descent()
    full_poses = convert_partial_to_full_config(poses)
    full_optimized_poses = convert_partial_to_full_config(optimized_poses)
    full_semi_optimized_poses = [convert_partial_to_full_config(poses) for poses in semi_optimized_poses]

    full_semi_optimized_poses = np.array(full_semi_optimized_poses)
    semi_optimized_poses = np.array(semi_optimized_poses)
    

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
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