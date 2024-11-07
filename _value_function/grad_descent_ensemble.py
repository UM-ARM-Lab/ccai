from screwdriver_problem import init_env, convert_full_to_partial_config, convert_partial_to_full_config
from train_value_function import Net, query_ensemble, load_ensemble
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
shape = (15, 1)

filename = '/initial_poses/initial_poses_10k.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    total_poses_full  = pkl.load(file)
    total_poses_full = np.array([tensor.numpy() for tensor in total_poses_full]).reshape(-1, 20)
total_poses = convert_full_to_partial_config(total_poses_full)

def get_data():

    inputs = torch.from_numpy(total_poses)[100:150]
    return inputs

def get_vf_gradients(poses, models, poses_mean, poses_std, cost_mean, cost_std):

    poses_norm = (poses - poses_mean) / poses_std
    poses_norm = poses_norm.float()

    # Enable gradient computation
    poses_norm.requires_grad_(True)
    for model in models:
        model.eval()

    target_value = torch.tensor([0.0], dtype=torch.float32)

    ensemble_predictions_norm = query_ensemble(poses_norm, models)
    ensemble_predictions = ensemble_predictions_norm * cost_std + cost_mean
    predictions = torch.mean(ensemble_predictions, dim=0)

    mse = torch.mean((predictions - target_value) ** 2)
    mean_squared_variance = torch.mean((ensemble_predictions - predictions) ** 2)
    loss = mse + mean_squared_variance
    loss.backward()

    grads_norm = poses_norm.grad
    grads = grads_norm * poses_std
    print(grads)
    return grads

def grad_descent(lr = 0.2358):
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble()
    
    poses = get_data()
    poses_norm = (poses - poses_mean) / poses_std
    poses_norm = poses_norm.float()
    original_costs_norm = torch.mean(query_ensemble(poses_norm, models), dim=0).detach().cpu().numpy()
    original_costs = original_costs_norm * cost_std + cost_mean
    # print(original_costs)

    # Enable gradient computation
    poses_norm.requires_grad_(True)

    exp = 0
    dump = True

    if exp == 0:
        experiment_name = '_ensemble_Adam_500_iters'
        optimizer = optim.Adam([poses_norm], lr=0.2358)
        iterations = 500
    elif exp == 1:
        experiment_name = '_ensemble_Adam_5000_iters'
        optimizer = optim.Adam([poses_norm], lr=1e-1)
        iterations = 5000
    elif exp == 2 and lr is not None:
        dump = False
        optimizer = optim.Adam([poses_norm], lr=lr)
        iterations = 500

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
        loss = mse + mean_squared_variance
        loss.backward()

        # Set gradients of the last four values of each pose to 0
        # poses_norm.grad[:, -4:] = 0

        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: mse = {mse.item()}, variance_loss = {mean_squared_variance.item()}")
            #print(f"Iteration {i}: Loss = {loss.item()}")
            pass
        n_setpoints = 10
        if i % int(num_iterations/n_setpoints) == 0:
            pose_optimization_trajectory.append(poses_norm.detach().cpu().numpy() * poses_std + poses_mean)

    optimized_poses_norm = poses_norm.detach().cpu().numpy()
    optimized_costs_norm = torch.mean(query_ensemble(poses_norm, models), dim=0).detach().cpu().numpy()
    optimized_poses = optimized_poses_norm * poses_std + poses_mean
    optimized_costs = optimized_costs_norm * cost_std + cost_mean

    full_initial = convert_partial_to_full_config(poses.numpy())
    full_optimized = convert_partial_to_full_config(optimized_poses)

    initial_pose_tuples = [(initial, optimized) for initial, optimized in zip(full_initial, full_optimized)]
    # predicted_cost_tuples = [(initial, optimized) for initial, optimized in zip(original_costs, optimized_costs)]

    print("mean of new predicted costs: ", optimized_costs.mean())
    print("original costs: ", original_costs)
    print("optimized costs: ", optimized_costs)

    vis = False
    if vis:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for initial, optimized in initial_pose_tuples:

            env.reset(torch.from_numpy(initial).reshape(1,20).float(), deterministic=True)
            time.sleep(0.5)
            # input("Press Enter to continue...")
            env.reset(torch.from_numpy(optimized).reshape(1,20).float(), deterministic=True)
            time.sleep(2.0)
            # input("Press Enter to continue...")

    if dump:
        output_filename = f'{fpath.resolve()}/eval/initial_and_optimized_poses{experiment_name}.pkl'
        with open(output_filename, 'wb') as f:
            pkl.dump(initial_pose_tuples, f)

    return poses.numpy(), optimized_poses, pose_optimization_trajectory, optimized_costs.mean()

def lr_sweep():
    lr = 0.2358
    _, _, _, mean_cost = grad_descent(lr=lr)
    min_cost = mean_cost
    best_lr = lr
    results = []
    costs = [mean_cost]
    results.append(f'lr: {lr}, cost: {mean_cost}')
    
    lr_multiplier = 1.01
    direction = 1
    patience = 5  # Number of steps to wait before considering reversing direction
    non_improving_steps = 0  # Track steps without improvement

    for _ in range(100):
        # Adjust the learning rate based on previous cost trends
        if direction == 1:
            lr *= lr_multiplier
        else:
            lr /= lr_multiplier

        # Run gradient descent with the updated learning rate
        _, _, _, mean_cost = grad_descent(lr=lr)

        # Check if the new cost improved
        if mean_cost < min_cost:
            min_cost = mean_cost
            best_lr = lr
            non_improving_steps = 0  # Reset non-improving counter
        else:
            non_improving_steps += 1

        # Reverse direction if too many steps without improvement
        if non_improving_steps >= patience:
            direction *= -1
            non_improving_steps = 0  # Reset after changing direction

        # Log the cost and learning rate
        costs.append(mean_cost)
        results.append(f'lr: {lr:.4f}, cost: {mean_cost:.4f}')

    # Print all results and best learning rate
    for result in results:
        print(result)
    print(f'Best lr: {best_lr:.4f}, min cost: {min_cost:.4f}')



if __name__ == "__main__":
    # lr_sweep()
    # exit()
    poses = get_data()
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble()
    get_vf_gradients(poses, models, poses_mean, poses_std, cost_mean, cost_std)
    exit()

    poses, optimized_poses,semi_optimized_poses, _ = grad_descent()
    full_poses = convert_partial_to_full_config(poses)
    full_optimized_poses = convert_partial_to_full_config(optimized_poses)
    full_semi_optimized_poses = [convert_partial_to_full_config(poses) for poses in semi_optimized_poses]

    full_semi_optimized_poses = np.array(full_semi_optimized_poses)
    semi_optimized_poses = np.array(semi_optimized_poses)
    
    vis_so_poses = False
    if vis_so_poses:
        config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for j in range(full_semi_optimized_poses.shape[1]):
            for i in range(full_semi_optimized_poses.shape[0]):
                env.reset(torch.from_numpy(full_semi_optimized_poses[i,j,:]).reshape(1,20).float(), deterministic=True)
                if i == 0:
                    time.sleep(1.0)
                print(f'Setpoint {i}')
                time.sleep(0.01)


    # Fit PCA on the 10k_poses data to get axes
    pca = PCA(n_components=3)
    pca.fit(total_poses)

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