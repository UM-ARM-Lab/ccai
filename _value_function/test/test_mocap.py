import numpy as np
import pickle as pkl

import pathlib
import shutil
from functools import partial
import yaml
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, convert_partial_to_full_config#, swap_index_middle
from _value_function.test.test_method import get_initialization
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
import pytorch_kinematics as pk
from isaac_victor_envs.utils import get_assets_dir

from ccai.utils.allegro_utils import state2ee_pos
from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv

import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
from itertools import product
import warnings
warnings.filterwarnings("ignore",message="You are using `torch.load` with `weights_only=False`",category=FutureWarning)
if len(sys.argv) == 2:
    config_path = f'allegro_screwdriver_adam{sys.argv[1]}.yaml'
else:
    config_path = 'allegro_screwdriver_adam0.yaml'

if __name__ == '__main__':

    perception_noise = 0.0
    max_screwdriver_tilt = 0.015
    screwdriver_noise_mag = 0.015
    finger_noise_mag = 0.05

    regrasp_iters = 80
    turn_iters = 100

    vf_weight_rg = 5.0
    other_weight_rg = 1.9
    variance_ratio_rg = 10.0

    vf_weight_t = 3.3
    other_weight_t = 1.9
    variance_ratio_t = 1.625

    pregrasp_path = fpath /'test'/'official_initializations'/'test_method_pregrasps_hardware.pkl'
    diffusion_path = 'data/training/allegro_screwdriver/adam_diffusion/allegro_screwdriver_diffusion_4999.pt'

    pregrasps = pkl.load(open(pregrasp_path, 'rb'))
    pregrasp_pose = pregrasps[2]

    ########################################################################################################################
    # HARDWARE ENVIRONMENT SETUP ###########################################################################################
    ########################################################################################################################

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{config_path}').read_text())

    from hardware.hardware_env import HardwareEnv
    pg = pregrasp_pose.clone()[:,:16]

    # file = f'{fpath.resolve()}/test/hardware_results/{"vf"}_trial_{0}.pkl'
    # with open(file, 'rb') as file:
    #     pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj, *initial_samples = pkl.load(file)
    # # pg = torch.from_numpy(turn_traj[-1,:16]).reshape(1,16)

    env = HardwareEnv(pg, 
                        finger_list=config['fingers'], 
                        kp=config['kp'],
                        obj='blue_screwdriver', 
                        # obj='screwdriver',
                        mode='relative',
                        gradual_control=True,
                        num_repeat=10)
    env.default_dof_pos = pg
    env.reset()

    env.get_state()
    for _ in range(5):
        root_coor, root_ori = env.obj_reader.get_state_world_frame_pos()
    print('Root coor:', root_coor)
    print('Root ori:', root_ori)
    root_coor = root_coor
    robot_p = np.array([0, -0.095, 1.33])
    root_coor = root_coor + robot_p

    sim_env = RosAllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=2.5,
                                device=config['sim_device'],
                                valve=config['object_type'],
                                video_save_path=None,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                table_pose=None,
                                gravity=False
                                )
    sim_env.reset(dof_pos= pregrasp_pose)
    
    sim, gym, viewer = sim_env.get_sim()
    assert (np.array(sim_env.robot_p) == robot_p).all()
    # assert (sim_env.initial_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
    env.world_trans = sim_env.world_trans
    env.joint_stiffness = sim_env.joint_stiffness
    env.device = sim_env.device
    env.table_pose = sim_env.table_pose

    ros_copy_node = None
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
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos_partial = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)

    while True:
        set_state = env.get_state()['q'].to(device=env.device)
        sim_env.set_pose(set_state)

