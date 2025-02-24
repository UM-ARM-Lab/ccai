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
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, convert_partial_to_full_config, swap_index_middle
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
import pytorch_kinematics as pk
from isaac_victor_envs.utils import get_assets_dir
from _value_function.test.test_method import get_initializations
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

    # max_screwdriver_tilt = 0.015
    # screwdriver_noise_mag = 0.015
    # finger_noise_mag = 0.0
    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True, config_path=config_path)
    # pregrasp_pose = get_initializations(env, config, chain, config['sim_device'], 1,
    #                         max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, save=False,
    #                         do_pregrasp=True, name='x')[0]
    
    # gym.destroy_viewer(viewer)
    # gym.destroy_sim(sim)
    # del config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial


    pregrasp_path = fpath /'test'/'official_initializations'/'test_method_pregrasps.pkl'

    pregrasps = pkl.load(open(pregrasp_path, 'rb'))
    pregrasp_pose = pregrasps[30]

    # pregrasp_pose = torch.cat((
    #         torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
    #         torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
    #         torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
    #         torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float(),
    #         torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float()
    #     ), dim=1)
    


    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{config_path}').read_text())

    from hardware.hardware_env import HardwareEnv
    pg = pregrasp_pose.clone()[:,:16]
    # swapped = swap_index_middle(pg)

    hardcode = torch.tensor([[ 
        1.0,  .0,  .0,  .0,  # middle  # index0123
        .0,  1.0,  .0,  .0, # index  # middle_0123
        .0,  .0,  .0,  .0,  # ring    #nothing
        .0,  .0,  .0,  .0,  # thumb # thumb0123
        ]])

    hardcode = pg
    
    # hardcode = torch.tensor([[-0.036893922835588455, 0.4413647949695587, 0.6582543253898621, 
    #        0.5270181894302368, 0.04545709490776062, 0.567008912563324, 
    #        0.7333610653877258, 0.6544234752655029, 0.0, 0.5, 
    #        0.6499999761581421, 0.6499999761581421, 1.3175060749053955, 
    #        0.36398831009864807, 0.13335226476192474, 1.026167869567871]])

    #home
    # hardcode = torch.tensor([[0.1000003848222671, 0.6000005636043506, 0.6000005636043506, 0.6000005636043506, -0.1000003848222671, 0.5000001787820835, 0.8999999727418999, 0.8999999727418999, 0.0, 0.5000001787820835, 0.6500007560154842, 0.6500007560154842, 1.1999993818794494, 0.29999940913754936, 0.29999940913754936, 1.1999993818794494]])

    # default
    # hardcode = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
    #                             torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
    #                             torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
    #                             torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
    #                             dim=1)
    
    env = HardwareEnv(hardcode, 
                        finger_list=config['fingers'], 
                        kp=config['kp'],
                        obj='blue_screwdriver', 
                        # obj='screwdriver',
                        mode='relative',
                        gradual_control=True,
                        num_repeat=10)
    env.default_dof_pos = hardcode
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
    
    full = torch.cat([hardcode, torch.zeros(1, 4)], dim=1)
    sim_env.reset(dof_pos= full)
    
    while True:
        sim_env.reset(dof_pos= full)
    
    