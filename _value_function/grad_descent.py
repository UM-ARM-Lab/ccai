import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from train_value_function import Net

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
shape = (20, 1)

def get_inputs():
    filename = '/initial_poses/initial_poses_10k.pkl'
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        poses  = pkl.load(file)
        poses = poses[:3]
    
    inputs = np.array([t.numpy() for t in poses]).reshape(-1, 20)
    inputs = torch.from_numpy(inputs)

    return inputs

def grad_descent():
    torch.manual_seed(42)
    model_name = "2"
    model = Net(shape[0], shape[1])
    checkpoint = torch.load(f'{fpath.resolve()}/value_functions/value_function_{model_name}.pkl')
    model.load_state_dict(checkpoint['model_state'])
    poses_mean = checkpoint['poses_mean']
    poses_std = checkpoint['poses_std']
    
    poses = get_inputs()
    poses_norm = (poses - poses_mean) / (poses_std + 0.000001)
    poses_norm = poses_norm.float()
    #print("Initial poses:", poses_norm)
    poses_norm.requires_grad_(True)
    
    optimizer = optim.Adam([poses_norm], lr=1e-2)
    loss_fn = nn.MSELoss()
    model.eval()

    # gradient descent
    num_iterations = 500
    target_value = torch.tensor([0.0], dtype=torch.float32)

    for i in range(num_iterations):
        optimizer.zero_grad()  
        predictions = model(poses_norm)
        loss = loss_fn(predictions, target_value)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    # Detach poses_norm to avoid further gradient tracking after optimization
    optimized_poses = poses_norm.detach().cpu().numpy()
    #print("Optimized poses:", optimized_poses)

    # Dump initial and optimized poses as tuples into a pkl file
    tuples = [(initial, optimized) for initial, optimized in zip(poses.numpy(), optimized_poses)]
    
    output_filename = f'{fpath.resolve()}/eval/initial_and_optimized_poses.pkl'
    with open(output_filename, 'wb') as f:
        pkl.dump(tuples, f)


if __name__ == "__main__":
    grad_descent()