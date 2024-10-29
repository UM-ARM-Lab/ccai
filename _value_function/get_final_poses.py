from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
import numpy as np
import pickle as pkl
import torch
import time
import copy
import yaml
import pathlib
from functools import partial
import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
import matplotlib.pyplot as plt
from utils.allegro_utils import state2ee_pos
from scipy.spatial.transform import Rotation as R
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from examples.allegro_valve_roll import PositionControlConstrainedSVGDMPC
from examples.allegro_screwdriver import AllegroScrewdriver
from tqdm import tqdm
from screwdriver_problem import init_env, do_turn, emailer

fpath = pathlib.Path(f'{CCAI_PATH}/data')
with open(f'{fpath.resolve()}/initial_poses/initial_poses_10k.pkl', 'rb') as file:
    initial_poses  = pkl.load(file)

pose_tuples = []

config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)
n_succ = 0
n_trials = 500
for i in tqdm(range(n_trials)):
    
    idx = i + config['start_idx']
    print("RUNNING TRIAL: ", idx)
    initial_pose = initial_poses[idx]
    _, final_pose, succ = do_turn(initial_pose, config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)
    pose_tuples.append((initial_pose, final_pose))
    if succ:
        n_succ += 1

print(f'Success rate: {n_succ/n_trials}')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

fpath = pathlib.Path(f'{CCAI_PATH}/data')
start_idx = config['start_idx']
savepath = f'{fpath.resolve()}/value_datasets/value_dataset_{start_idx}.pkl'
with open(savepath, 'wb') as f:
    pkl.dump(pose_tuples, f)

print(f'saved to {savepath}')
emailer().send()