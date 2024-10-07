import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

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
    
    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    with open(f'{fpath.resolve()}/final_costs.pkl', 'rb') as file:
        pose_cost_tuples  = pkl.load(file)

    poses, costs = zip(*pose_cost_tuples)

    poses = np.array([t.numpy() for t in poses]).reshape(-1,20)
    poses_mean, poses_std = np.mean(poses, axis=0), np.std(poses, axis=0)
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)

    costs = np.array(costs).reshape(-1,1)
    cost_mean, cost_std = np.mean(costs), np.std(costs)
    costs_norm = (costs - cost_mean) / cost_std

    # Convert to PyTorch tensors
    poses_tensor = torch.from_numpy(poses_norm).float()
    costs_tensor = torch.from_numpy(costs_norm).float()

    # Split the dataset into training and test sets (90% training, 10% testing)
    dataset = torch.utils.data.TensorDataset(poses_tensor, costs_tensor)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net(poses_tensor.shape[1], 1)  # Instantiate the neural network model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100000
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

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)

        # Test the model
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

    # Save the trained model
    torch.save(model.state_dict(), f'{fpath.resolve()}/value_function.pkl')