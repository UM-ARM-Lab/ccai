from screwdriver_problem import init_env
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from train_value_function import Net, query_ensemble
import time

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
shape = (20, 1)

filename = '/initial_poses/initial_poses_10k.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    total_poses  = pkl.load(file)
    total_poses = np.array([tensor.numpy() for tensor in total_poses]).reshape(-1, 20)

def get_data():

    inputs = torch.from_numpy(total_poses)[100:120]
    return inputs

def grad_descent():
    checkpoints = torch.load(open(f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl', 'rb'))
    models = []
    for checkpoint in checkpoints:
        model = Net(shape[0], shape[1])
        model.load_state_dict(checkpoint['model_state'])

        # checkpoint = torch.load(f'{fpath.resolve()}/value_functions/value_function_{2}.pkl')
        # model.load_state_dict(checkpoint['model_state'])
    
        models.append(model)
        # break
    
    poses_mean = checkpoints[0]['poses_mean']
    poses_std = checkpoints[0]['poses_std']
    cost_mean = checkpoints[0]['cost_mean']
    cost_std = checkpoints[0]['cost_std']
    
    poses = get_data()
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)
    poses_norm = poses_norm.float()
    original_costs_norm = torch.mean(query_ensemble(poses_norm, models), dim=0).detach().cpu().numpy()
    original_costs = original_costs_norm * cost_std + cost_mean

    # Enable gradient computation
    poses_norm.requires_grad_(True)

    optimizer = optim.SGD([poses_norm], lr=0.1)
    # optimizer = optim.Adam([poses_norm], lr=1e-2)

    model.eval()

    # gradient descent
    num_iterations = 1000
    target_value = torch.tensor([0.0], dtype=torch.float32)

    pose_optimization_trajectory = []
    
    for i in range(num_iterations):
        optimizer.zero_grad()  
        ensemble_predictions = query_ensemble(poses_norm, models)
        ensemble_predictions = ensemble_predictions * cost_std + cost_mean
        predictions = torch.mean(ensemble_predictions, dim=0)

        mse = torch.mean((predictions - target_value) ** 2)
        mean_squared_variance = torch.mean((ensemble_predictions - predictions) ** 2)
        loss = mse + mean_squared_variance
        loss.backward()

        # Set gradients of the last four values of each pose to 0
        poses_norm.grad[:, -4:] = 0

        optimizer.step()

        if i % 100 == 0:
            
            print(f"Iteration {i}: mse = {mse.item()}, variance_loss = {mean_squared_variance.item()}")
            #print(f"Iteration {i}: Loss = {loss.item()}")
            pass
        n_setpoints = 20
        if i % int(num_iterations/n_setpoints) == 0:
            pose_optimization_trajectory.append(poses_norm.detach().cpu().numpy() * poses_std + poses_mean)

    optimized_poses_norm = poses_norm.detach().cpu().numpy()
    optimized_costs_norm = torch.mean(query_ensemble(poses_norm, models), dim=0).detach().cpu().numpy()
    optimized_poses = optimized_poses_norm * poses_std + poses_mean
    optimized_costs = optimized_costs_norm * cost_std + cost_mean

    initial_pose_tuples = [(initial, optimized) for initial, optimized in zip(poses.numpy(), optimized_poses)]
    predicted_cost_tuples = [(initial, optimized) for initial, optimized in zip(original_costs, optimized_costs)]

    # print(original_costs)
    print("mean of new predicted costs: ", optimized_costs.mean())

    vis = False
    if vis:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for initial, optimized in initial_pose_tuples:
            env.reset(torch.from_numpy(initial).reshape(1,20), deterministic=True)
            time.sleep(0.5)
            # input("Press Enter to continue...")
            env.reset(torch.from_numpy(optimized).reshape(1,20).float(), deterministic=True)
            time.sleep(1.0)
            # input("Press Enter to continue...")


    #print(predicted_cost_tuples)
    #print(np.mean(np.abs((poses.numpy() -  optimized_poses))))

    # experiment_name = '_ensemble_SGD_100k_iters'
    # experiment_name = '_ensemble_SGD_10k_iters'
    experiment_name = '_ensemble_Adam_10k_iters'
    output_filename = f'{fpath.resolve()}/eval/initial_and_optimized_poses{experiment_name}.pkl'
    with open(output_filename, 'wb') as f:
        pkl.dump(initial_pose_tuples, f)

    return poses.numpy(), optimized_poses, pose_optimization_trajectory


if __name__ == "__main__":

    # Assuming grad_descent() is defined elsewhere and returns two arrays
    poses, optimized_poses,semi_optimized_poses = grad_descent()


    semi_optimized_poses = np.array(semi_optimized_poses)
    vis_so_poses = True
    if vis_so_poses:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for j in range(semi_optimized_poses.shape[1]):
            for i in range(semi_optimized_poses.shape[0]):
                env.reset(torch.from_numpy(semi_optimized_poses[i,j,:]).reshape(1,20).float(), deterministic=True)
                print(f'Setpoint {i}')
                time.sleep(0.01)


    
    poses = poses[:, :-4]
    optimized_poses = optimized_poses[:, :-4]

    # Fit PCA on the 10k_poses data to get axes
    pca = PCA(n_components=3)
    pca.fit(total_poses[:, :-4])

    # Transform poses and optimized_poses based on the PCA axes from 10k_poses
    poses_pca = pca.transform(poses)
    optimized_poses_pca = pca.transform(optimized_poses)

    # Transform semi_optimized_poses based on the PCA axes
    semi_optimized_poses_pca = [pca.transform(np.array(poses)[:, :-4]) for poses in semi_optimized_poses]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_intermediate = True
    for i in range(len(semi_optimized_poses_pca)):
        if i == 0:
            color = 'r'
        elif i == len(semi_optimized_poses_pca)-1:
            color = 'b'
        else:
            color = 'g'
            if not plot_intermediate:
                continue
        ax.scatter(semi_optimized_poses_pca[i][:, 0], 
                   semi_optimized_poses_pca[i][:, 1], 
                   semi_optimized_poses_pca[i][:, 2], 
                   c=color)

    # Draw dotted lines connecting initial -> semi_optimized -> optimized for each point
    for i in range(len(semi_optimized_poses_pca[0])):
        
        # Start with the initial pose
        line_points = [poses_pca[i]]
        # Append each semi-optimized pose
        for j in range(len(semi_optimized_poses_pca)):
            if j != 0 and j != len(semi_optimized_poses_pca)-1 and not plot_intermediate:
                continue
            line_points.append(semi_optimized_poses_pca[j][i])
        # Add the optimized pose at the end
        line_points.append(optimized_poses_pca[i])
        
        # Convert line points to x, y, z for plotting
        line_points = np.array(line_points)
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'k--')  # Dotted line

    # Add labels and legend
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()

    plt.show()




