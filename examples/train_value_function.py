# from isaac_victor_envs.utils import get_assets_dir
# from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
# import torch
# import time
# import copy
# import yaml
# 
# from functools import partial
# import pytorch_volumetric as pv
# import pytorch_kinematics as pk
# import pytorch_kinematics.transforms as tf
# import matplotlib.pyplot as plt
# from utils.allegro_utils import *
# from scipy.spatial.transform import Rotation as R
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    #print(poses_std)
    poses = (poses - poses_mean) / (poses_std + 0.000001)

    costs = np.array(costs).reshape(-1,1)
    cost_mean, cost_std = np.mean(costs), np.std(costs)
    #print(cost_std)
    costs = (costs - cost_mean) / cost_std
    #print(costs[:20])

    #print(poses.shape)
    #print(costs.shape)
    #print(poses[:5])
    #exit()

    model = Net(poses[0].shape[0],1)  # Instantiate your neural network model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    criterion = torch.nn.MSELoss()

    poses_tensor = torch.from_numpy(poses).float()
    costs_tensor = torch.from_numpy(costs).float()
    
    # Training loop
    num_epochs = 100000
    for epoch in range(num_epochs):

        # forward
        predicted_costs = model(poses_tensor)
        #print(predicted_costs[:10])
        #print(costs_tensor[:10])
        #print(predicted_costs.shape, costs_tensor.shape)
        #exit()
        loss = criterion(predicted_costs, costs_tensor)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress
        
        if (epoch + 1) % 100 == 0:
            #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item() / costs.shape[0]:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}")

    torch.save(model.state_dict(), f'{fpath.resolve()}/value_function.pkl')
