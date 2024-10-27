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


def load_data():
    filename = '/value_datasets/combined_value_dataset.pkl'
    validation_proportion = 0.1
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        pose_cost_tuples  = pkl.load(file)
        poses, costs = zip(*pose_cost_tuples)

    num_samples = len(poses)
    split_idx = int(num_samples * (1 - validation_proportion))

    poses = np.array(poses).reshape(-1,20)
    poses_mean, poses_std = np.mean(poses[:split_idx], axis=0), np.std(poses[:split_idx], axis=0)
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)

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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std

class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        neurons = 32
        self.fc1 = nn.Linear(dim_in, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, dim_out)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output.squeeze()
    

def train():
    shape = (20,1)
    
    # Initialize W&B
    wandb.init(project="value-function-training", config={
        "epochs": 300,
        "batch_size": 64,
        "learning_rate": 0.0001,
    })
    
    train_loader, test_loader, poses_mean, poses_std, cost_mean, cost_std = load_data()
    # print(cost_std)
    # print(cost_mean)
    # exit()

    model = Net(shape[0], shape[1])  # Instantiate the neural network model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = wandb.config.epochs
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

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "unnormalized_train_loss": unnormalized_loss
            })

    # save
    wandb.finish()
    model_to_save = {
        'model_state': model.state_dict(),
        'poses_mean': poses_mean,
        'poses_std': poses_std,
        'cost_mean': cost_mean,
        'cost_std': cost_std,
    }
    return model_to_save
    

def save(model_to_save, path):
    torch.save(model_to_save, path)

def query_ensemble(poses, models):
    costs = []
    for model in models:
        costs.append(model(poses))
    costs = torch.stack(costs)
    return costs

def eval(model_name, ensemble = False):
    shape = (20,1)
    
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
    
    def plot_loader(loader, title = ''):
        with torch.no_grad():
            actual_values = []
            predicted_values = []

            for inputs, labels in loader:
                if not ensemble:
                    predictions = model(inputs)
                else:
                    predictions = query_ensemble(inputs, models).mean(dim=0)
                predicted_values.append(predictions.numpy())
                actual_values.append(labels.numpy())
                break

        actual_values = np.concatenate(actual_values).flatten()
        predicted_values = np.concatenate(predicted_values).flatten()
        actual_values = actual_values * cost_std + cost_mean
        predicted_values = predicted_values * cost_std + cost_mean

        print(f'Actual values: {[f"{val:.4f}" for val in actual_values[:10]]}')
        print(f'Predicted values: {[f"{val:.4f}" for val in predicted_values[:10]]}')

        num_samples = len(actual_values)
        plt.figure(figsize=(10, 6))
        plt.scatter(range(num_samples), actual_values, label='Actual Values', color='green')
        plt.scatter(range(num_samples), predicted_values, label='Predicted Values', color='blue')
        # Draw lines between corresponding actual and predicted values
        for i in range(num_samples):
            plt.plot([i, i], [actual_values[i], predicted_values[i]], color='red', linestyle='--', linewidth=0.8)

        plt.xlabel('Sample Index')
        plt.ylabel('Cost Value')
        plt.title('Actual vs Predicted Values with Error Lines, '+title)
        plt.legend()
        plt.show()
    
    plot_loader(train_loader, 'Training Set')
    plot_loader(test_loader, 'Test Set')

if __name__ == "__main__":

    # model_name = "2"#input("Enter model name: ")
    # model_to_save = train()
    # save(model_to_save, f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
    # eval(model_name = "2") 
    # exit()


    ensemble = []
    for i in range(8):
        net = train()
        ensemble.append(net)
    torch.save(ensemble, f'{fpath.resolve()}/value_functions/value_function_ensemble.pkl')
    eval(model_name = "ensemble", ensemble = True)

