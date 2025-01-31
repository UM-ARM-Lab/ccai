from _value_function.screwdriver_problem import convert_full_to_partial_config
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

def save_train_test_splits(noisy=False, validation_proportion=0.1, seed=None):

    if noisy:
        filename = 'trajgen_datasets/noisy_combined_trajgen_dataset.pkl'
    else:
        filename = 'trajgen_datasets/combined_trajgen_dataset.pkl'
    
    if seed is not None:
        np.random.seed(seed)
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        trajgen_tuples = pkl.load(file)
        pregrasp_poses, regrasp_poses, regrasp_trajs, turn_trajs = zip(*trajgen_tuples)

    T_rg = regrasp_trajs[0].shape[0]
    T = T_rg
    n_trajs = len(regrasp_trajs)
    print(f'Loaded {n_trajs} samples')

    pregrasp_poses = np.stack(pregrasp_poses, axis=0)
    pregrasp_poses = pregrasp_poses.reshape(-1, 20)
    pregrasp_poses = convert_full_to_partial_config(pregrasp_poses)

    regrasp_poses = np.stack(regrasp_poses, axis=0)
    regrasp_poses = regrasp_poses.reshape(-1, 20)
    regrasp_poses = convert_full_to_partial_config(regrasp_poses)

    regrasp_trajs = np.stack(regrasp_trajs, axis=0)
    regrasp_trajs = regrasp_trajs.reshape(-1, 20)
    regrasp_trajs = convert_full_to_partial_config(regrasp_trajs)
    regrasp_trajs = regrasp_trajs.reshape(-1, T, 15)

    turn_trajs = np.stack(turn_trajs, axis=0)
    turn_trajs = turn_trajs.reshape(-1, 20)
    turn_trajs = convert_full_to_partial_config(turn_trajs)
    turn_trajs = turn_trajs.reshape(-1, T, 15)
    
    poses = pregrasp_poses
    trajs = regrasp_trajs.reshape(-1, T*15)

    indices = np.arange(len(poses))
    np.random.shuffle(indices)

    split_idx = int(n_trajs * (1 - validation_proportion))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    poses_train = poses[train_indices]
    poses_test  = poses[test_indices]
    trajs_train = trajs[train_indices]
    trajs_test  = trajs[test_indices]

    # Compute stats from TRAIN only
    poses_mean = np.mean(poses_train, axis=0)
    poses_std = np.std(poses_train, axis=0)
    trajs_mean = np.mean(trajs_train)
    trajs_std = np.std(trajs_train)

    min_std_threshold = 1e-5
    poses_std = np.where(poses_std < min_std_threshold, min_std_threshold, poses_std)
    trajs_std = max(trajs_std, min_std_threshold)

    # Normalize
    poses_train_norm = (poses_train - poses_mean) / poses_std
    trajs_train_norm = (trajs_train - trajs_mean) / trajs_std
    poses_test_norm = (poses_test - poses_mean) / poses_std
    trajs_test_norm = (trajs_test - trajs_mean) / trajs_std

    # Convert to tensors
    poses_train_tensor = torch.from_numpy(poses_train_norm).float()
    trajs_train_tensor = torch.from_numpy(trajs_train_norm).float()
    poses_test_tensor = torch.from_numpy(poses_test_norm).float()
    trajs_test_tensor = torch.from_numpy(trajs_test_norm).float()

    path = f'{fpath.resolve()}/trajgen_networks/dataloader.pkl'

    pkl.dump((poses_train_tensor, trajs_train_tensor, poses_test_tensor, trajs_test_tensor, poses_mean, poses_std, trajs_mean, trajs_std), open(path, 'wb'))


def load_data_from_saved_splits(batch_size = 100):

    path = f'{fpath.resolve()}/trajgen_networks/dataloader.pkl'
    poses_train_tensor, trajs_train_tensor, poses_test_tensor, trajs_test_tensor, poses_mean, poses_std, trajs_mean, trajs_std = pkl.load(open(path, 'rb'))
    
    train_dataset = TensorDataset(poses_train_tensor, trajs_train_tensor)
    test_dataset = TensorDataset(poses_test_tensor, trajs_test_tensor)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, poses_mean, poses_std, trajs_mean, trajs_std

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
    shape = (15,13*15)
    
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

def load_net(device='cpu', model_name = "throwerror"):

    shape = (15,15*13)

    checkpoint = torch.load(f'{fpath.resolve()}/trajgen_networks/trajgen_network_{model_name}.pkl')
    neurons = checkpoint["model_state"]["fc1.weight"].shape[0]

    model = Net(shape[0], shape[1], neurons = neurons)
    model.load_state_dict(checkpoint['model_state'])

    poses_mean = checkpoint['poses_mean']
    poses_std = checkpoint['poses_std']
    cost_mean = checkpoint['cost_mean']
    cost_std = checkpoint['cost_std']

    return model, poses_mean, poses_std, cost_mean, cost_std

def query_net(poses, model, device='cpu'):

    costs = model(poses.to(device))
    return costs

def eval(model_name):
    
    train_loader, test_loader, _,_,_,_ = load_data_from_saved_splits()
    model, poses_mean, poses_std, cost_mean, cost_std = load_net(model_name=model_name)
    
    def plot_loader(loader, n_samples, title=''):
        with torch.no_grad():
            prediction_stds = []
            actual_values = [] 
            predicted_values = []

            for inputs, labels in loader:

                net_predictions = query_net(inputs, model)
                predicted_values.append(net_predictions.numpy())
                actual_values.append(labels.numpy())

            actual_values = np.concatenate(actual_values).flatten()
            actual_values = actual_values * cost_std + cost_mean
            predicted_values = np.concatenate(predicted_values).flatten()
            predicted_values = predicted_values * cost_std + cost_mean

            # select n_samples random samples to plot
            indices = np.random.choice(len(actual_values), n_samples, replace=False)
            actual_values = actual_values[indices]
            predicted_values = predicted_values[indices]

            print(f'Actual values: {[f"{val:.4f}" for val in actual_values[:10]]}')
            print(f'Predicted values: {[f"{val:.4f}" for val in predicted_values[:10]]}')

            # Calculate and print average error
            prediction_errors = np.abs(actual_values - predicted_values)
            avg_error = np.mean(prediction_errors)
            print(f'Average Prediction Error: {avg_error:.4f}')

            num_samples = len(actual_values)
            plt.figure(figsize=(10, 6))
            plt.scatter(range(num_samples), predicted_values, 
                            label='Predicted Values', color='blue')
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
    
    plot_loader(train_loader, 100, 'Training Set')
    plot_loader(test_loader, 100, 'Test Set')

if __name__ == "__main__":

    noisy = False
    if noisy:
        path = f'{fpath.resolve()}/trajgen_networks/trajgen_network_net.pkl'
        model_name = "net_noisy"
    else:
        path = f'{fpath.resolve()}/trajgen_networks/trajgen_network_net.pkl'
        model_name = "net"

    # save_train_test_splits(noisy=noisy, validation_proportion=0.05)
    # exit()

    net, _ = train(epochs=30, neurons = 30, verbose='very', lr=1e-3, batch_size=100)
    torch.save(net, path)
    eval(model_name = model_name)
    

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
        epochs_candidates=[50],
        neurons_candidates=[12, 14, 16],
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

    # hyperparam_search()