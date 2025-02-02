from _value_function.screwdriver_problem import convert_full_to_partial_config, init_env, emailer
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def index_and_sort_regrasp_and_turn_trajs(regrasp_trajs, turn_trajs):

    regrasp_stacked = np.stack(regrasp_trajs, axis=0)
    # regrasp_poses = regrasp_stacked[:,-1,:]
    regrasp_poses = regrasp_stacked.reshape(-1, 20)
    regrasp_poses = convert_full_to_partial_config(regrasp_poses)

    return regrasp_poses

def save_train_test_splits(noisy=False, dataset_size=None, validation_proportion=0.1, seed=None):

    if noisy:
        filename = '/regrasp_to_turn_datasets/noisy_combined_regrasp_to_turn_dataset.pkl'
    else:
        filename = '/regrasp_to_turn_datasets/combined_regrasp_to_turn_dataset.pkl'
    
    if seed is not None:
        np.random.seed(seed)
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        pose_cost_tuples = pkl.load(file)
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

    turn_costs = np.array(turn_costs).flatten()
    costs = np.repeat(turn_costs, T)

    indices = np.arange(len(poses))
    # np.random.shuffle(indices)

    split_idx = int(n_trajs * (1 - validation_proportion))*T

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    poses_train = poses[train_indices]
    poses_test  = poses[test_indices]
    costs_train = costs[train_indices]
    costs_test  = costs[test_indices]

    # Compute stats from TRAIN only
    poses_mean = np.mean(poses_train, axis=0)
    poses_std = np.std(poses_train, axis=0)
    cost_mean = np.mean(costs_train)
    cost_std = np.std(costs_train)

    min_std_threshold = 1e-5
    poses_std = np.where(poses_std < min_std_threshold, min_std_threshold, poses_std)
    cost_std = max(cost_std, min_std_threshold)

    # Normalize
    poses_train_norm = (poses_train - poses_mean) / poses_std
    costs_train_norm = (costs_train - cost_mean) / cost_std
    poses_test_norm = (poses_test - poses_mean) / poses_std
    costs_test_norm = (costs_test - cost_mean) / cost_std

    # Add the time-index column just like in load_data
    # (We assume that data was stacked in order, so we replicate the logic)
    n_trajs_train = len(costs_train)
    n_trajs_test = len(costs_test)
    T_array_train = np.tile(np.arange(T), n_trajs_train // T).reshape(-1, 1)
    T_array_test = np.tile(np.arange(T), n_trajs_test // T).reshape(-1, 1)

    poses_train_norm = np.hstack([poses_train_norm, T_array_train])
    poses_test_norm = np.hstack([poses_test_norm, T_array_test])

    # Convert to tensors
    poses_train_tensor = torch.from_numpy(poses_train_norm).float()
    costs_train_tensor = torch.from_numpy(costs_train_norm).float()
    poses_test_tensor = torch.from_numpy(poses_test_norm).float()
    costs_test_tensor = torch.from_numpy(costs_test_norm).float()

    path = f'{fpath.resolve()}/value_functions/dataloader.pkl'

    pkl.dump((poses_train_tensor, costs_train_tensor, poses_test_tensor, costs_test_tensor, poses_mean, poses_std, cost_mean, cost_std), open(path, 'wb'))


def load_data_from_saved_splits(batch_size = 100):

    path = f'{fpath.resolve()}/value_functions/dataloader.pkl'
    poses_train_tensor, costs_train_tensor, poses_test_tensor, costs_test_tensor, poses_mean, poses_std, cost_mean, cost_std = pkl.load(open(path, 'rb'))
    
    train_dataset = TensorDataset(poses_train_tensor, costs_train_tensor)
    test_dataset = TensorDataset(poses_test_tensor, costs_test_tensor)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    
def train(batch_size = 100, lr = 0.001, epochs = 205, neurons = 12, verbose="normal"):
    shape = (16,1)
    
    train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std = load_data_from_saved_splits(batch_size=batch_size)

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

def eval(model_name):
    
    train_loader, test_loader, _,_,_,_ = load_data_from_saved_splits()
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name=model_name)
    
    def plot_loader(loader, n_samples, title=''):
        with torch.no_grad():
            prediction_stds = []
            actual_values = [] 
            predicted_values = []

            for inputs, labels in loader:

                ensemble_predictions = query_ensemble(inputs, models)
                predictions = ensemble_predictions.mean(dim=0)
                prediction_std = ensemble_predictions.std(dim=0)  # Calculate standard deviation for error bars
                prediction_stds.append(prediction_std.numpy())

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
            plt.errorbar(range(num_samples), predicted_values, yerr=prediction_stds, fmt='o', 
                            label='Predicted Values', color='blue', elinewidth=2, capsize=3)
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
    plot_loader(test_loader, 10, 'Test Set')

if __name__ == "__main__":

    noisy = False
    if noisy:
        path = f'{fpath.resolve()}/value_functions/value_function_ensemble_noisy.pkl'
        model_name = "ensemble_noisy"
    else:
        path = f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl'
        model_name = "ensemble"

    # save_train_test_splits(noisy=noisy, dataset_size=None, validation_proportion=0.05, seed=1)
    # exit()

    path = f'{fpath.resolve()}/value_functions/value_function_ensemble_test.pkl'
    model_name = "ensemble_test"

    ensemble = []
    for i in range(16):
        net, _ = train(epochs=20, neurons = 16, verbose='very', lr=1e-3, batch_size=100)
        ensemble.append(net)
    torch.save(ensemble, path)
    emailer().send()
    eval(model_name = model_name)
    exit()
    

    ######################################################
    # Training networks with different dataset sizes

    # for dataset_size in [800]:
    #     path = f'{fpath.resolve()}/value_functions/value_function_ensemble_{dataset_size}_samples.pkl'
    #     model_name = f'ensemble_{dataset_size}_samples'
    #     ensemble = []
    #     for i in range(16):
    #         net, _ = train(noisy=noisy, epochs=61, neurons = 12, dataset_size = dataset_size)
    #         ensemble.append(net)

    #     torch.save(ensemble, path)
    #     eval(model_name = model_name, ensemble = True)
    

    def hyperparam_search(
        lr_candidates=[1e-3],
        epochs_candidates=[800],
        neurons_candidates=[512, 1028, 2056, 4112],
        batch_size=100,
        verbose="normal"
    ):
        """
        Perform a grid search over the given hyperparameter candidates and 
        return the best combination (lowest test loss) along with the trained model.
        """

        lowest_test_loss = float('inf')
        best_hparams = None
        best_model = None  # This will store the best model state dict

        # Iterate over all combinations of the hyperparameter candidates
        for lr in lr_candidates:
            for epochs in epochs_candidates:
                for neurons in neurons_candidates:
                    print(f"\nTraining with lr={lr}, epochs={epochs}, neurons={neurons}")
                    # Train your model
                    model_to_save, test_loss = train(
                        batch_size=batch_size,
                        lr=lr,
                        epochs=epochs,
                        neurons=neurons,
                        verbose=verbose
                    )

                    # Compare test_loss to see if it is the best so far
                    if test_loss < lowest_test_loss:
                        lowest_test_loss = test_loss
                        best_hparams = (lr, epochs, neurons)
                        best_model = model_to_save

        print("\n=====================================")
        print("Finished hyperparameter search.")
        print(f"Best hyperparameters found: LR={best_hparams[0]}, "
            f"Epochs={best_hparams[1]}, "
            f"Neurons={best_hparams[2]}")
        print(f"Best (lowest) test loss: {lowest_test_loss:.8f}")
        print("=====================================\n")

    hyperparam_search()