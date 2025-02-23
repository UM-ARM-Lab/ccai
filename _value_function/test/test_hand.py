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

    pregrasp_path = fpath /'test'/'official_initializations'/'test_method_pregrasps.pkl'

    pregrasps = pkl.load(open(pregrasp_path, 'rb'))
    pregrasp_pose = pregrasps[0]

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{config_path}').read_text())

    from hardware.hardware_env import HardwareEnv
    pg = pregrasp_pose.clone()[:,:16]
    swapped = swap_index_middle(pg)

    # swapped = torch.tensor([[ 
    #     0.0455,  0.5670,  0.7334,  0.6544,  # middle
    #     -0.0369,  0.4414,  0.6583,  0.5270, # index
    #     0.0369,  0.4414,  0.6583,  0.5270, # ring  
    #     1.3175,  0.3640,  0.1334,  1.0262]]) # thumb
    env = HardwareEnv(swapped, 
                        finger_list=config['fingers'], 
                        kp=config['kp'],
                        obj='blue_screwdriver', 
                        # obj='screwdriver',
                        mode='relative',
                        gradual_control=True,
                        num_repeat=10)
    env.default_dof_pos = swapped
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

    while True:
        pass
    
    