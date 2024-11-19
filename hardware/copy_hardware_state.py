
# from isaacsim_hand_envs.allegro import AllegroScrewdriverEnv # it needs to be imported before numpy and torch
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv, AllegroPegTurningEnv
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

import matplotlib.pyplot as plt # from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos, visualize_trajectory, all_finger_constraints
from scipy.spatial.transform import Rotation as R

from utils.allegro_utils import *
from tqdm import tqdm


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# torch.cuda.set_device(1)
# instantiate environment
img_save_dir = None

   
if __name__ == "__main__":
    # get config
    # task = 'screwdriver_turning'
    # task = 'valve_turning'
    # task = 'reorientation'
    task = 'peg_alignment'
    # task = 'peg_turning'

    method = 'csvgd'
    # method = 'ablation'
    # method = 'planning'

    if task == 'screwdriver_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]        
        config['num_env_force'] = 1
    elif task == 'valve_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_valve.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 0, 1, 0]
        config['num_env_force'] = 0
    elif task == 'peg_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_peg_turning.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 0
    elif task == 'peg_alignment':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_peg_alignment.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 1
    elif task == 'reorientation':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_reorientation.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 0

    obj_dof = sum(config['obj_dof_code'])
    config['obj_dof'] = obj_dof
    config['task'] = task
    config['method'] = method

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
        from hardware.hardware_env import HardwareEnv
        # TODO, think about how to read that in simulator
        if task == 'screwdriver_turning':
            default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                        torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                        torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                        torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                        dim=1)
            obj = 'screwdriver'
        elif task == 'peg_alignment':
            default_dof_pos = torch.cat((torch.tensor([[0, 0.7, 0.8, 0.8]]).float(),
                                    torch.tensor([[0, 0.8, 0.7, 0.6]]).float(),
                                    torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.05, 1.1]]).float()),
                                    dim=1)
            obj = 'peg'
        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj=obj,
                          mode='relative',
                          gradual_control=config['gradual_control'],
                          num_repeat=10)
        if task == 'screwdriver_turning':
            from isaac_victor_envs.utils import get_assets_dir
            from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
            root_coor, root_ori = env.obj_reader.get_state()
            root_coor = root_coor / 1000 # convert to meters
            # robot_p = np.array([-0.025, -0.1, 1.33])
            robot_p = np.array([0, -0.095, 1.33])
            root_coor = root_coor + robot_p
            sim_env = AllegroScrewdriverTurningEnv(num_envs=1, 
                                           control_mode='joint_impedance',
                                            use_cartesian_controller=False,
                                            viewer=True,
                                            steps_per_action=60,
                                            friction_coefficient=1.0,
                                            device=config['sim_device'],
                                            video_save_path=img_save_dir,
                                            joint_stiffness=config['kp'],
                                            fingers=config['fingers'],
                                            gradual_control=config['gradual_control'],
                                            arm_type=config['arm_type'],
                                            gravity=config['gravity'],
                                            obj_pose=root_coor,
                                            )
        elif task == 'peg_alignment':
            from isaac_victor_envs.utils import get_assets_dir
            from utils.isaacgym_utils import get_env
            sim_env = get_env(task, img_save_dir, config)
        sim, gym, viewer = sim_env.get_sim()
        if task == 'screwdriver_turning':
            assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.obj_pose = sim_env.obj_pose
        if task == 'peg_alignment':
            env.wall_pose = sim_env.wall_pose
            env.wall_dims = sim_env.wall_dims
        
    while True:
        set_state = env.get_state(return_dict=True)['q'].to(device=env.device)
        if task == 'screwdriver_turning':
            set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
        sim_env.set_pose(set_state)
