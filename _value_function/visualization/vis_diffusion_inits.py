import numpy as np
import pickle as pkl
import pathlib
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, delete_imgs, convert_partial_to_full_config
from _value_function.test.test_method import get_initialization
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import time

config_path = 'allegro_screwdriver_adam0.yaml'
config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True, config_path=config_path)
sim_device = config['sim_device']
computer_id = config['data_collection_id']


rg_traj = pkl.load(open(f'{fpath}/diffusion/initial_samples/raw/regrasp_init.pkl', 'rb'))
t_traj = pkl.load(open(f'{fpath}/diffusion/initial_samples/raw/turn_init.pkl', 'rb'))

rg_traj = convert_partial_to_full_config(rg_traj[0,:,:15])
t_traj = convert_partial_to_full_config(t_traj[0,:,:15])

traj = rg_traj

while True:

    for i in range(traj.shape[0]):
        env.reset(traj[i].reshape(1,20).float())
        time.sleep(0.5)
    time.sleep(1.0)
   