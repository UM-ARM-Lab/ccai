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
prog_idx = 0
trials_per_save = 500
source_filename = f'initial_poses/initial_poses_{prog_idx}.pkl'
start_idx = prog_idx * trials_per_save

with open(f'{fpath.resolve()}/{source_filename}', 'rb') as file:
    initial_poses  = pkl.load(file)

pose_tuples = []
config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)

for j in range(10000/trials_per_save):
    for i in tqdm(range(trials_per_save)):
        
        idx = i + start_idx
        print("RUNNING TRIAL: ", idx)
        initial_pose = initial_poses[idx]
        _, final_pose, succ = do_turn(initial_pose, config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)
        pose_tuples.append((initial_pose, final_pose))

    savepath = f'{fpath.resolve()}/value_datasets/value_dataset_source_{prog_idx}_startid_{start_idx}.pkl'
    with open(savepath, 'wb') as f:
        pkl.dump(pose_tuples, f)

    start_idx += trials_per_save

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
emailer().send()