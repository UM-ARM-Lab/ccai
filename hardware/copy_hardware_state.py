import os
import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv, orientation_error, quat_change_convention
from isaac_victor_envs.utils import get_assets_dir

import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd
# from functorch import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.valve import ValveDynamics
from ccai.utils import rotate_jac
import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

import matplotlib.pyplot as plt
import pickle as pkl
from hardware.allegro_ros import RosNode
from utils.allegro_utils import partial_to_full_state

ros_node = RosNode()
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# asset = f'{get_assets_dir()}/victor/allegro.urdf'
# index_ee_name = 'index_ee'
# thumb_ee_name = 'thumb_ee'
asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
# thumb_ee_name = 'allegro_hand_oya_finger_link_15'
# index_ee_name = 'allegro_hand_hitosashi_finger_finger_link_3'
index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'

# combined chain
chain = pk.build_chain_from_urdf(open(asset).read())
index_ee_link = chain.frame_to_idx[index_ee_name]
thumb_ee_link = chain.frame_to_idx[thumb_ee_name]
frame_indices = torch.tensor([index_ee_link, thumb_ee_link])

device = 'cuda:0'
# device = 'cpu'
valve_location = torch.tensor([0.85, 0.70, 1.405]).to(device)  # the root of the valve
# instantiate environment
valve_type = 'cylinder'  # 'cuboid' or 'cylinder
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

env = AllegroValveTurningEnv(1, control_mode='joint_impedance', use_cartesian_controller=False,
                             viewer=True, steps_per_action=5, valve_velocity_in_state=False,
                             friction_coefficient=1.0,
                             device=device,
                             valve=valve_type,
                             video_save_path=img_save_dir,
                             configuration='screw_driver')

while True:
    state = torch.tensor(ros_node.current_joint_pose.position).to(env.device)
    input_state = torch.cat((state[:4], state[-4:])).unsqueeze(0)
    input_state = partial_to_full_state(input_state)
    input_state = torch.cat((input_state, torch.zeros(1, 1).to(env.device)), dim=-1)
    env.set_pose(input_state)
