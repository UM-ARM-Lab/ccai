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
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
import matplotlib.pyplot as plt
from utils.allegro_utils import *
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R
from baselines.planning.ik import IKSolver

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    fingers=['index', 'middle', 'thumb']

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                #video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=True,
                                )

    sim, gym, viewer = env.get_sim()
    state = env.get_state()
    device = config['sim_device']

    dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float().to(device=device),
                        torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float().to(device=device),
                        torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=device),
                        torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float().to(device=device)),
                        dim=1).to(device)
    dof_pos = torch.cat((dof_pos, torch.zeros((1, 4)).float().to(device=device)),
                                         dim=1).to(device)
    dof_pos = dof_pos.repeat(1, 1)
    partial_default_dof_pos = np.concatenate((dof_pos[:, 0:8], dof_pos[:, 12:16]), axis=1)

    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in fingers]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=fingers, chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)

    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices)
    fk = forward_kinematics(dof_pos[:,:16])
    # list of poses for each of the three fingers, each pose is a Transform3d object
    default_poses = list(fk.values())

    lim = torch.tensor(chain.get_joint_limits(), device=device)
    
    ik = IKSolver(chain, fingers, device)

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    with open(f'{fpath.resolve()}/initial_poses.pkl', 'rb') as file:
        initial_poses  = pkl.load(file)
    

    with open(f'{fpath.resolve()}/contact_points.pkl', 'rb') as file:
        contact_points  = pkl.load(file)
    
    print(contact_points)

    #print(initial_poses)

    while True:
        for i in range(len(initial_poses)):
            env.reset(initial_poses[i])
            time.sleep(1)
        exit()


    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)