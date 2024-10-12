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

obj_dof = 3
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

fpath = pathlib.Path(f'{CCAI_PATH}/data')

filename = 'combined_value_dataset.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    pose_cost_tuples  = pkl.load(file)
    poses, costs = zip(*pose_cost_tuples)
poses = np.array(poses).reshape(-1,1,20)
costs = np.array(costs).flatten()


#print(costs)
high_cost = np.argsort(costs)[-100:][::-1]
low_cost = np.argsort(costs)[:100]
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
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
    from tqdm import tqdm
    sim_env = None
    ros_copy_node = None

    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=True,
                                gravity=True,
                                randomize_obj_start=True,
                                )
    sim, gym, viewer = env.get_sim()

    state = env.get_state()
    results = {}
    succ_rate = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    config['ee_names'] = ee_names
    config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]
    config['obj_dof'] = np.sum(config['obj_dof_code'])

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) 

    for i in range(100):
        #env.reset(dof_pos = torch.tensor(high_initial_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        env.reset(dof_pos = torch.tensor(low_initial_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        time.sleep(1)
        #env.reset(dof_pos = torch.tensor(high_final_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        env.reset(dof_pos = torch.tensor(low_final_poses[i]).to(device=config['sim_device']).float(), deterministic=True)
        sd = low_final_poses[i][:, -4:-1]
        time.sleep(0.5)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)