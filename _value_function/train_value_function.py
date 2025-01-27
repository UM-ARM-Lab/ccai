from _value_function.screwdriver_problem import convert_full_to_partial_config
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

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def index_and_sort_regrasp_and_turn_trajs(regrasp_trajs, turn_trajs):

    regrasp_stacked = np.stack(regrasp_trajs, axis=0)
    regrasp_poses = regrasp_stacked.reshape(-1, 20)
    regrasp_poses = convert_full_to_partial_config(regrasp_poses)

    # poses = np.empty((regrasp_poses.shape[0], regrasp_poses.shape[1]), dtype=regrasp_poses.dtype)

    return regrasp_poses

def load_data(batch_size = 64, noisy = False, dataset_size = None):
    if noisy:
        filename = '/regrasp_to_turn_datasets/noisy_combined_regrasp_to_turn_dataset.pkl'
    else:
        filename = '/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset.pkl'

    validation_proportion = 0.1
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        pose_cost_tuples  = pkl.load(file)
        regrasp_trajs, regrasp_costs, turn_trajs, turn_costs = zip(*pose_cost_tuples)
    
    if dataset_size is not None:
        regrasp_trajs = regrasp_trajs[:dataset_size]
        regrasp_costs = regrasp_costs[:dataset_size]
        turn_trajs = turn_trajs[:dataset_size]
        turn_costs = turn_costs[:dataset_size]
    
    T_rg = regrasp_trajs[0].shape[0]
    T = T_rg
    n_trajs = len(regrasp_trajs)
    print(f'Loaded {n_trajs} trials, which will create {n_trajs*T} samples')

    poses = index_and_sort_regrasp_and_turn_trajs(regrasp_trajs, turn_trajs)
    
    num_samples = len(poses)
    split_idx = int(num_samples * (1 - validation_proportion))

    # normalize
    poses_mean, poses_std = np.mean(poses[:split_idx], axis=0), np.std(poses[:split_idx], axis=0)
    poses_norm = (poses - poses_mean) / poses_std

    turn_costs = np.array(turn_costs).flatten()
    # cost_weight = 1.0
    # discount_factor = 1.0
    # regrasp_costs = np.array(regrasp_costs).flatten()
    # costs = regrasp_costs + turn_costs * cost_weight
    # costs = np.repeat(costs, T)
    # powers = np.arange(T-1, -1, -1)
    # discounts = np.repeat(np.power(discount_factor, powers), n_trajs)
    # costs = costs * discounts
    costs = np.repeat(turn_costs, T)
    
    cost_mean, cost_std = np.mean(costs[:split_idx]), np.std(costs[:split_idx])
    costs_norm = (costs - cost_mean) / cost_std

    # add indices
    indices = np.tile(np.arange(T), n_trajs).reshape(-1, 1)
    poses_norm = np.hstack([poses_norm, indices])

    # Convert to tensors
    poses_tensor = torch.from_numpy(poses_norm).float()
    costs_tensor = torch.from_numpy(costs_norm).float()

    # Split the dataset into training and test sets
    dataset = torch.utils.data.TensorDataset(poses_tensor, costs_tensor)
    train_size = int((1-validation_proportion) * len(dataset))
    test_size = len(dataset) - train_size
    print(f'Train size: {train_size}, Test size: {test_size}')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    min_std_threshold = 1e-5
    poses_std = np.where(poses_std < min_std_threshold, min_std_threshold, poses_std)

    return train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std

class Net(nn.Module):
    def __init__(self, dim_in, dim_out, neurons = 12):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, dim_out)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output.squeeze()
    

def train(batch_size = 100, lr = 0.001, epochs = 205, neurons = 12, noisy = False, verbose="normal"):
    shape = (16,1)
    
    # Initialize W&B
    # wandb.init(project="value-function-training", config={
    #     "epochs": epochs,
    #     "batch_size": batch_size,
    #     "learning_rate": lr,
    # })
    
    train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std = load_data(batch_size=batch_size, noisy = noisy)

    # CHANGED FOR GPU: set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(shape[0], shape[1], neurons = neurons).to(device)  # CHANGED FOR GPU: move model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # CHANGED FOR GPU: move inputs/labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            predicted_costs = model(inputs)
            loss = criterion(predicted_costs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Eval
        if verbose == "very":
            freq = 1
        elif verbose == "normal":
            freq = epochs // 3
        else:
            freq = epochs + 1

        if epoch == 0 or (epoch+1) % freq == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    if np.isnan(inputs).any() or np.isnan(labels).any():
                        print("issue")
                    # CHANGED FOR GPU: move inputs/labels to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    predicted_costs = model(inputs)
                    loss = criterion(predicted_costs, labels)
                    test_loss += loss.item()
            test_loss /= len(test_loader)

            # Print and log losses
            train_loss = running_loss / len(train_loader)
            unnormalized_loss = train_loss * (cost_std**2)  # Scale the loss back to original cost scale
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Unnormalized Train Loss: {unnormalized_loss:.8f}")

            # wandb.log({
            #     "epoch": epoch + 1,
            #     "train_loss": train_loss,
            #     "test_loss": test_loss,
            #     "unnormalized_train_loss": unnormalized_loss
            # })

    # save
    # wandb.finish()
    model_to_save = {
        'model_state': model.state_dict(),
        'poses_mean': poses_mean,
        'poses_std': poses_std,
        'cost_mean': cost_mean,
        'cost_std': cost_std,
    }

    return model_to_save, test_loss
    

def save(model_to_save, path):
    torch.save(model_to_save, path)

def load_ensemble(device='cpu', model_name = "throwerror"):

    shape = (16,1)
    checkpoints = torch.load(f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
    neurons = checkpoints[0]["model_state"]["fc1.weight"].shape[0]
    models = []
    for checkpoint in checkpoints:
        model = Net(shape[0], shape[1], neurons = neurons)
        model.load_state_dict(checkpoint['model_state'])
        models.append(model.to(device))

    poses_mean = checkpoints[0]['poses_mean']
    poses_std = checkpoints[0]['poses_std']
    cost_mean = checkpoints[0]['cost_mean']
    cost_std = checkpoints[0]['cost_std']

    return models, poses_mean, poses_std, cost_mean, cost_std

def query_ensemble(poses, models, device='cpu'):
    costs = []
    for model in models:
        costs.append(model(poses.to(device)))
    costs = torch.stack(costs)
    return costs

def eval(model_name, ensemble=False):
    shape = (16, 1)
    
    #model_name = input("Enter model name: ")
    train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std = load_data()
    if not ensemble:
        model = Net(shape[0], shape[1])
        checkpoint = torch.load(f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
    else:
        models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name=model_name)
    
    def plot_loader(loader, n_samples, title=''):
        with torch.no_grad():
            prediction_stds = []
            actual_values = [] 
            predicted_values = []

            for inputs, labels in loader:
                if np.isnan(inputs).any() or np.isnan(labels).any():
                    print("issue")
                if not ensemble:
                    predictions = model(inputs)
                else:
                    ensemble_predictions = query_ensemble(inputs, models)
                    predictions = ensemble_predictions.mean(dim=0)
                    prediction_std = ensemble_predictions.std(dim=0)  # Calculate standard deviation for error bars
                    prediction_stds.append(prediction_std.numpy())

                if np.isnan(predictions).any():
                    print("issue")
                predicted_values.append(predictions.numpy())
                actual_values.append(labels.numpy())

            actual_values = np.concatenate(actual_values).flatten()
            actual_values = actual_values * cost_std + cost_mean
            predicted_values = np.concatenate(predicted_values).flatten()
            predicted_values = predicted_values * cost_std + cost_mean

            # select n_samples random samples to plot
            indices = np.random.choice(len(actual_values), n_samples, replace=False)
            actual_values = actual_values[indices]
            predicted_values = predicted_values[indices]

            if ensemble:
                prediction_stds = np.concatenate(prediction_stds).flatten() * cost_std  # Scale by cost_std
                prediction_stds = prediction_stds[indices]

            print(f'Actual values: {[f"{val:.4f}" for val in actual_values[:10]]}')
            print(f'Predicted values: {[f"{val:.4f}" for val in predicted_values[:10]]}')

            # Calculate and print average error
            prediction_errors = np.abs(actual_values - predicted_values)
            avg_error = np.mean(prediction_errors)
            print(f'Average Prediction Error: {avg_error:.4f}')

            num_samples = len(actual_values)
            plt.figure(figsize=(10, 6))

            # Plot with error bars if ensemble is True
            if ensemble:
                plt.errorbar(range(num_samples), predicted_values, yerr=prediction_stds, fmt='o', 
                             label='Predicted Values', color='blue', elinewidth=2, capsize=3)
            else:
                plt.scatter(range(num_samples), predicted_values, label='Predicted Values', color='blue')

            plt.scatter(range(num_samples), actual_values, label='Actual Values', color='green')
            
            # Draw lines between corresponding actual and predicted values
            for i in range(num_samples):
                plt.plot([i, i], [actual_values[i], predicted_values[i]], color='red', linestyle='--', linewidth=0.8, label='Prediction Error' if i == 0 else None)

            fs = 20
            plt.xlabel('Sample Index', fontsize=fs)
            plt.ylabel('Cost Value', fontsize=fs)
            plt.title(f'Actual vs Predicted Costs, {title}', fontsize=fs)
            plt.legend(loc='upper right', fontsize=fs)
            plt.tight_layout()
            plt.show()
    
    plot_loader(train_loader, 50, 'Training Set')
    plot_loader(test_loader, 50, 'Test Set')

if __name__ == "__main__":

    # model_name = "2"#input("Enter model name: ")
    # model_to_save, _ = train()
    # save(model_to_save, f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
    # eval(model_name = "2") 
    # exit()


    # epoch_vals = [150]
    # lr_vals = [0.01]
    # batch_size_vals = [100]
    # neuron_vals = [700, 800, 900]
    # results = []
    # total_epochs = np.sum(np.array(epoch_vals)) * len(lr_vals) * len(batch_size_vals) * len(neuron_vals)
    # epochs_completed = 0
    # min_test_loss = float('inf')
    # for epoch in epoch_vals:
    #     for lr in lr_vals:
    #         for batch_size in batch_size_vals:
    #             for neurons in neuron_vals:
    #                 print(f'On Epoch {epochs_completed}/{total_epochs}')
    #                 model_to_save, test_loss = train(batch_size = batch_size, lr = lr, epochs = epoch, neurons = neurons, verbose = "very")
    #                 results.append(f'Epochs: {epoch}, LR: {lr}, Batch Size: {batch_size}, Neurons: {neurons}, Test Loss: {test_loss}')
    #                 if test_loss < min_test_loss:
    #                     min_test_loss = test_loss
    #                     best_result = results[-1]
    #                 epochs_completed += epoch
    
    # for result in results:
    #     print(result)
    # print(f'Best result: {best_result}')
    # exit()

    noisy = False
    if noisy:
        path = f'{fpath.resolve()}/value_functions/value_function_ensemble_noisy.pkl'
        model_name = "ensemble_noisy"
    else:
        path = f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl'
        model_name = "ensemble"

    ensemble = []
    for i in range(16):
        # net, _ = train(noisy=noisy, epochs=151, neurons = 512, verbose='normal')
        net, _ = train(noisy=noisy, epochs=301, neurons = 512, verbose='very')
        # net, _ = train(noisy=noisy, epochs=30, neurons = 32, verbose='very')
        ensemble.append(net)
    torch.save(ensemble, path)
    eval(model_name = model_name, ensemble = True)

    ######################################################
    # Training networks with different dataset sizes

    # for dataset_size in [100, 500, 1000, 5000]:
    #     path = f'{fpath.resolve()}/value_functions/value_function_ensemble_{dataset_size}_samples.pkl'
    #     # model_name = f'ensemble_{dataset_size}_samples'
    #     ensemble = []
    #     for i in range(16):
    #         net, _ = train(noisy=noisy, epochs=61)
    #         ensemble.append(net)
    #     torch.save(ensemble, path)