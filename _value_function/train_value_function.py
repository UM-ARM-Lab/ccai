from screwdriver_problem import convert_full_to_partial_config
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import wandb
import matplotlib.pyplot as plt

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def load_data(batch_size = 64):
    filename = '/value_datasets/combined_value_dataset.pkl'
    validation_proportion = 0.1
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        pose_cost_tuples  = pkl.load(file)
        full_poses, costs = zip(*[(t[0], t[1]) for t in pose_cost_tuples])

    full_poses = np.array(full_poses).reshape(-1,20)
    poses = convert_full_to_partial_config(full_poses)

    num_samples = len(poses)
    split_idx = int(num_samples * (1 - validation_proportion))

    poses_mean, poses_std = np.mean(poses[:split_idx], axis=0), np.std(poses[:split_idx], axis=0)
    poses_norm = (poses - poses_mean) / poses_std

    costs = np.array(costs).flatten()
    cost_mean, cost_std = np.mean(costs[:split_idx]), np.std(costs[:split_idx])
    costs_norm = (costs - cost_mean) / cost_std

    # Convert to tensors
    poses_tensor = torch.from_numpy(poses_norm).float()
    costs_tensor = torch.from_numpy(costs_norm).float()

    # Split the dataset into training and test sets
    dataset = torch.utils.data.TensorDataset(poses_tensor, costs_tensor)
    train_size = int((1-validation_proportion) * len(dataset))
    test_size = len(dataset) - train_size
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
    

def train(batch_size = 100, lr = 0.01, epochs = 205, neurons = 12):
    shape = (15,1)
    
    # Initialize W&B
    # wandb.init(project="value-function-training", config={
    #     "epochs": epochs,
    #     "batch_size": batch_size,
    #     "learning_rate": lr,
    # })
    
    train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std = load_data(batch_size=batch_size)

    model = Net(shape[0], shape[1], neurons = neurons)  # Instantiate the neural network model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # forward pass
            predicted_costs = model(inputs)
            loss = criterion(predicted_costs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Eval
        if epoch == 0 or (epoch+1) % 100 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
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

def query_ensemble(poses, models):
    costs = []
    for model in models:
        costs.append(model(poses))
    costs = torch.stack(costs)
    return costs

def eval(model_name, ensemble = False):
    shape = (15,1)
    
    #model_name = input("Enter model name: ")
    train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std = load_data()
    if not ensemble:
        model = Net(shape[0], shape[1])
        checkpoint = torch.load(f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
    else:
        checkpoints = torch.load(f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
        models = []
        for checkpoint in checkpoints:
            model = Net(shape[0], shape[1])
            model.load_state_dict(checkpoint['model_state'])
            models.append(model)
    
    def plot_loader(loader, n_samples, title=''):
        with torch.no_grad():
            prediction_stds = []
            actual_values = [] 
            predicted_values = []

            for inputs, labels in loader:
                if not ensemble:
                    predictions = model(inputs)
                else:
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

            if ensemble:
                prediction_stds = np.concatenate(prediction_stds).flatten() * cost_std  # Scale by cost_std
                prediction_stds = prediction_stds[indices]

            print(f'Actual values: {[f"{val:.4f}" for val in actual_values[:10]]}')
            print(f'Predicted values: {[f"{val:.4f}" for val in predicted_values[:10]]}')

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
                plt.plot([i, i], [actual_values[i], predicted_values[i]], color='red', linestyle='--', linewidth=0.8)

            plt.xlabel('Sample Index')
            plt.ylabel('Cost Value')
            plt.title('Actual vs Predicted Costs')
            plt.legend()
            plt.show()
    
    plot_loader(train_loader, 100, 'Training Set')
    plot_loader(test_loader, 100, 'Test Set')

if __name__ == "__main__":

    model_name = "2"#input("Enter model name: ")
    model_to_save, _ = train()
    save(model_to_save, f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
    eval(model_name = "2") 
    exit()

    # epoch_vals = [205, 210, 215]
    # lr_vals = [0.01]
    # batch_size_vals = [100]
    # neuron_vals = [12]
    # results = []
    # total_epochs = np.sum(np.array(epoch_vals)) * len(lr_vals) * len(batch_size_vals) * len(neuron_vals)
    # epochs_completed = 0
    # min_test_loss = float('inf')
    # for epoch in epoch_vals:
    #     for lr in lr_vals:
    #         for batch_size in batch_size_vals:
    #             for neurons in neuron_vals:
    #                 print(f'On Epoch {epochs_completed}/{total_epochs}')
    #                 model_to_save, test_loss = train(batch_size = batch_size, lr = lr, epochs = epoch, neurons = neurons)
    #                 results.append(f'Epochs: {epoch}, LR: {lr}, Batch Size: {batch_size}, Neurons: {neurons}, Test Loss: {test_loss}')
    #                 if test_loss < min_test_loss:
    #                     min_test_loss = test_loss
    #                     best_result = results[-1]
    #                 epochs_completed += epoch
    
    # for result in results:
    #     print(result)
    # print(f'Best result: {best_result}')


    # ensemble = []
    # for i in range(16):
    #     net, _ = train()
    #     ensemble.append(net)
    # torch.save(ensemble, f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl')
    eval(model_name = "ensemble", ensemble = True)


