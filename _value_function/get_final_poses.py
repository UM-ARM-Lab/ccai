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
from pathlib import Path
from screwdriver_problem import init_env, do_turn, emailer

fpath = pathlib.Path(f'{CCAI_PATH}/data')

def refresh_initial_poses():
    all_poses = []
    for file in Path(f'{fpath.resolve()}/initial_poses').glob("initial_poses_*.pkl"):
        with open(file, 'rb') as f:
            poses = pkl.load(f)
            for pose in poses:
                all_poses.append(pose)

    print(f'Loaded {len(all_poses)} poses')
    return all_poses

prog_idx = 0
trials_per_save = 500

while True:
    initial_poses = refresh_initial_poses()
    pose_tuples = []
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)

    for _ in tqdm(range(trials_per_save)):
        idx = np.random.randint(0, len(initial_poses))
        initial_pose = initial_poses[idx]
        _, final_pose, succ, full_trajectory = do_turn(initial_pose, config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)
        pose_tuples.append((initial_pose, final_pose, full_trajectory))

    savepath = f'{fpath.resolve()}/value_datasets/value_dataset_b_{prog_idx}.pkl'
    with open(savepath, 'wb') as f:
        pkl.dump(pose_tuples, f)

    prog_idx += 1
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

emailer().send()