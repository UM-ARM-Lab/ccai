from _value_function.screwdriver_problem import init_env
import pathlib
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
import numpy as np
import pickle as pkl
import torch
import time
fpath = pathlib.Path(f'{CCAI_PATH}/data/value_datasets')
which_data = 'grasp'
filename = 'combined_value_dataset_{which_data}.pkl'
pose_cost_tuples  = pkl.load(open(f'{fpath.resolve()}/{filename}', 'rb'))
poses, costs, trajectories = zip(*pose_cost_tuples)
poses = np.array(poses).reshape(-1,1,20)
costs = np.array(costs).flatten()

#print(costs)
high_cost = np.argsort(costs)[-10:][::-1]
low_cost = np.argsort(costs)[:10]
print("high costs: ", costs[high_cost])
print("low costs: ", costs[low_cost])

high_initial_poses = poses[high_cost]
low_initial_poses = poses[low_cost]

filename = 'combined_final_poses.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    final_poses = pkl.load(file).reshape(-1,1,20)

high_final_poses = final_poses[high_cost]
low_final_poses = final_poses[low_cost]



if __name__ == "__main__":
    
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    for i in range(10):
        #env.reset(dof_pos = torch.tensor(high_initial_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        env.reset(dof_pos = torch.tensor(low_initial_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        time.sleep(1)
        #env.reset(dof_pos = torch.tensor(high_final_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        env.reset(dof_pos = torch.tensor(low_final_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        sd = low_final_poses[i][:, -4:-1]
        time.sleep(0.5)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)