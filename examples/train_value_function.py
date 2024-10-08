import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import wandb
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_out)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output
    
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    
    # Initialize W&B
    wandb.init(project="value-function-training", config={
        "epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0001
    })
    validation_proportion = 0.1
    
    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    with open(f'{fpath.resolve()}/value_dataset_100.pkl', 'rb') as file:
        pose_cost_tuples  = pkl.load(file)

    poses, costs = zip(*pose_cost_tuples)
    num_samples = len(poses)
    split_idx = int(num_samples * (1 - validation_proportion))

    poses = np.array([t.numpy() for t in poses]).reshape(-1,20)
    poses_mean, poses_std = np.mean(poses[:split_idx], axis=0), np.std(poses[:split_idx], axis=0)
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)

    costs = np.array(costs).reshape(-1,1)
    cost_mean, cost_std = np.mean(costs[:split_idx]), np.std(costs[:split_idx])
    costs_norm = (costs - cost_mean) / cost_std

    # Convert to PyTorch tensors
    poses_tensor = torch.from_numpy(poses_norm).float()
    costs_tensor = torch.from_numpy(costs_norm).float()

    # Split the dataset into training and test sets (90% training, 10% testing)
    dataset = torch.utils.data.TensorDataset(poses_tensor, costs_tensor)
    train_size = int((1-validation_proportion) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net(poses_tensor.shape[1], 1)  # Instantiate the neural network model
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

        # get average loss per prediction
        avg_train_loss = running_loss / len(train_loader)

        # Eval
        if (epoch + 1) % 100 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    predicted_costs = model(inputs)
                    loss = criterion(predicted_costs, labels)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)

            # Print training loss, test loss, and unnormalized loss
            unnormalized_loss = avg_train_loss * (cost_std**2)  # Scale the loss back to original cost scale
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, Test Loss: {avg_test_loss:.8f}, Unnormalized Train Loss: {unnormalized_loss:.8f}")

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "unnormalized_train_loss": unnormalized_loss
            })

    # Save the model
    save_model = {
        'model_state': model.state_dict(),
        'poses_mean': poses_mean,
        'poses_std': poses_std,
        'cost_mean': cost_mean,
        'cost_std': cost_std,
    }

    torch.save(save_model, f'{fpath.resolve()}/value_function.pkl')
    
    # Mark the run as complete in W&B
    wandb.finish()

    # plot some predictions
    model.eval()
    with torch.no_grad():
        actual_values = []
        predicted_values = []
        
        for inputs, labels in test_loader:
            predictions = model(inputs)
            predicted_values.append(predictions.numpy())
            actual_values.append(labels.numpy())
            break

    actual_values = np.concatenate(actual_values).flatten()
    predicted_values = np.concatenate(predicted_values).flatten()
    actual_values = actual_values * cost_std + cost_mean
    predicted_values = predicted_values * cost_std + cost_mean

    num_samples = len(actual_values)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_samples), actual_values, label='Actual Values', color='green')
    plt.scatter(range(num_samples), predicted_values, label='Predicted Values', color='blue')
    # Draw lines between corresponding actual and predicted values
    for i in range(num_samples):
        plt.plot([i, i], [actual_values[i], predicted_values[i]], color='red', linestyle='--', linewidth=0.8)

    plt.xlabel('Sample Index')
    plt.ylabel('Cost Value')
    plt.title('Actual vs Predicted Values with Error Lines')
    plt.legend()
    plt.show()