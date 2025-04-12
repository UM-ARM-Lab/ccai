import numpy as np
import pickle as pkl
import pathlib
import sys
import time
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import emailer, delete_imgs
from card.card_problem import init_env, pull_index, pull_middle
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def get_initialization(env, sim_device, card_noise_mag0, card_noise_mag1, finger_noise_mag):

    index_noise_mag = torch.tensor([finger_noise_mag]*4)
    index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
    middle_noise_mag = torch.tensor([finger_noise_mag]*4)
    middle_noise = middle_noise_mag * (2 * torch.rand(4) - 1)
    card_noise = torch.tensor([
        np.random.uniform(-card_noise_mag0, card_noise_mag0),   
        0.0,
        0.0,
        0.0,
        0.0,
        np.random.uniform(-card_noise_mag1, card_noise_mag1),  
    ])
    #fingers=['index', 'middle', 'ring', 'thumb']
    initialization = torch.cat((
        torch.tensor([[0, 0.135, 0.5, 0.225]]).float().to(device=sim_device) + index_noise,
        torch.tensor([[0, 0.135, 0.5, 0.225]]).float().to(device=sim_device) + middle_noise,
        torch.tensor([[0, 0.2, 0.3, 0.2]]).float().to(device=sim_device),
        torch.tensor([[1.2, 0.3, 0.0, 0.8]]).float().to(device=sim_device),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).float().to(device=sim_device) + card_noise
    ), dim=1).to(sim_device)
     
    env.reset(dof_pos= initialization)
    for _ in range(1000):
        env._step_sim()
    solved_initialization = env.get_full_q().to(device=sim_device)

    return solved_initialization

# do main code

if __name__ == "__main__":
            
    prog_id = 0
    trials_per_save = 5
    # warmup_iters = 35
    # online_iters = 150

    if len(sys.argv) == 2:
        config_path = f'card{sys.argv[1]}.yaml'
    else:
        config_path = 'card0.yaml'

    visualize = False
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer = init_env(visualize=visualize, config_path=config_path)
    sim_device = config['sim_device']
    computer_id = config['data_collection_id']


    while True:

        pose_tuples = []
        
        trials_done = 0

        while trials_done < trials_per_save:

            img_save_dir = None
            env.frame_fpath = img_save_dir
            env.frame_id = 0

            print(f"Starting Trial {trials_done+1}")
            initialization = get_initialization(env, sim_device, card_noise_mag0=0.06, card_noise_mag1=0.2, finger_noise_mag=0.2)
            
            pose_index1, traj_index1 = pull_index(env, config, chain)
            print("done index1")
            pose_middle, traj_middle = pull_middle(env, config, chain)
            print("done middle")
            pose_index2, traj_index2 = pull_index(env, config, chain)
            print("done index2")

            pose_tuples.append((initialization, traj_index1, traj_middle, traj_index2, pose_index2))
            trials_done += 1

        savepath = f'{fpath.resolve()}/card_datasets/card_dataset_{computer_id}_{prog_id}.pkl'

        while Path(savepath).exists():
            prog_id += 1
            savepath = f'{fpath.resolve()}/card_datasets/card_dataset_{computer_id}_{prog_id}.pkl'

        pkl.dump(pose_tuples, open(savepath, 'wb'))