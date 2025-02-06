import numpy as np
import pickle as pkl
import pathlib
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, delete_imgs
from _value_function.test.test_method import get_initialization
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import time

def validate_pregrasp_pose(pregrasp_pose):
    screwdriver = pregrasp_pose[0,-4:-2]
    # print(screwdriver)
    # print(np.linalg.norm(screwdriver))
    if np.linalg.norm(screwdriver) > 0.3 :
        return False
    else:
        return True

prog_id = 0
trials_per_save = 5
perception_noise = 0.0
pregrasp_iters = 80
regrasp_iters = 100
turn_iters = 200

if len(sys.argv) == 2:
    config_path = f'allegro_screwdriver_adam{sys.argv[1]}.yaml'
else:
    config_path = 'allegro_screwdriver_adam0.yaml'

visualize = True
config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=visualize, config_path=config_path)
sim_device = config['sim_device']
computer_id = config['data_collection_id']

while True:

    initialization = get_initialization(env, sim_device, max_screwdriver_tilt=0.015, screwdriver_noise_mag=0.015, finger_noise_mag=0.087)
    
    env.reset(dof_pos=initialization)

    time.sleep(2)