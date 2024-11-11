
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv
from isaacsim_hand_envs.allegro import AllegroScrewdriverEnv # it needs to be imported before numpy and torch
# from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
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
from torch.func import vmap, jacrev, hessian, jacfwd

import matplotlib.pyplot as plt# from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos, visualize_trajectory, all_finger_constraints
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.allegro_utils import *

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# torch.cuda.set_device(1)
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')
nominal_screwdriver_top = np.array([0, 0, 1.405])

    
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the screwdriver once"
    env.reset()
    num_fingers = len(params['fingers'])
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    
    if params['arm_type'] == 'robot':
        arm_dof = 7
    elif params['arm_type'] == 'floating_3d':
        arm_dof = 3
    elif params['arm_type'] == 'floating_6d':
        arm_dof = 6
    elif params['arm_type'] == 'None':
        arm_dof = 0
    else:
        raise ValueError('Invalid arm type')

    robot_dof = 4 * num_fingers + arm_dof

    action_list = np.load('/home/fanyang/github/ccai/data/experiments/allegro_screwdriver_isaacgym/csvgd/trial_1/action.pkl', allow_pickle=True)
    state = env.get_state()
    for k in range(params['num_steps'] + 4):
        action = action_list[k].unsqueeze(0).to(params['sim_device'])
        # action[:, :4] = 0
        env.step(action)
   
    ret = {}
    return ret  

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
    for controller in config['controllers']:
        algorithm_device = config['controllers'][controller]['device']
        break
    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if config['simulator'] == 'isaac_gym':
        from isaac_victor_envs.utils import get_assets_dir
        env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
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
                                    )
        sim, gym, viewer = env.get_sim()
    elif config['simulator'] == 'isaac_sim':
        from isaacsim_hand_envs.utils import get_assets_dir
        env = AllegroScrewdriverEnv(friction_coefficient=10.0,
                                    device=config['sim_device'],
                                    video_save_path=img_save_dir,
                                    joint_stiffness=config['kp'],
                                    fingers=config['fingers'],
                                    arm_type=config['arm_type'],
                                    gravity=config['gravity'],
                                    )
        gym, viewer = None, None

    if config['mode'] == 'hardware_copy':
        from hardware.hardware_env import RosNode
        ros_copy_node = RosNode()
        

    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass

    results = {}

    # set up the kinematic chain
    if config['arm_type'] == 'robot':
        asset = f'{get_assets_dir()}/xela_models/victor_allegro.urdf'
        arm_dof = 7
    elif config['arm_type'] == 'floating_3d':
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right_floating_3d.urdf'
        arm_dof = 3
    elif config['arm_type'] == 'floating_6d':
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right_floating_6d.urdf'
        arm_dof = 6
    elif config['arm_type'] == 'None':
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
        arm_dof = 0
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
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans, arm_dof=arm_dof)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])

    for controller in config['controllers'].keys():
        results[controller] = {}
        results[controller]['warmup_time'] = []
        results[controller]['dist2goal'] = []
        results[controller]['validity_flag'] = []
        results[controller]['avg_online_time'] = []

    for i in tqdm(range(config['num_trials'])):
        goal = - 90 / 180 * torch.tensor([0, 0, np.pi])
        for controller in config['controllers'].keys():
            validity = False
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['screwdriver_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            ret = do_trial(env, params, fpath, sim_env, ros_copy_node)
            

    if config['simulator'] == 'isaac_gym':
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
    elif config['simulator'] == 'isaac_sim':
        env.close()

