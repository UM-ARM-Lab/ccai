from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroCardSlidingEnv
import numpy as np
import pickle as pkl
import pickle
import torch
import time
import copy
from copy import deepcopy
import yaml
import pathlib
from functools import partial
import sys
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ccai.utils.allegro_utils import *
from ccai.allegro_contact import AllegroManipulationExternalContactProblem, PositionControlConstrainedSVGDMPC
from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.models.contact_samplers import GraphSearchCard, Node
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
import shutil
import warnings
from examples.allegro_card_short import AllegroCard
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)
def _partial_to_full(params, traj, mode):
        if mode == 'index':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 3).to(device=params['device'])), dim=-1)
        if mode == 'middle':
            traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -3:]), dim=-1)
        if mode == "reposition":
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)
        return traj

def _full_to_partial(traj, mode):
    if mode == 'index':
        traj = traj[..., :-3]
    if mode == 'middle':
        traj = torch.cat((traj[..., :-6], traj[..., -3:]), dim=-1)
    if mode == "reposition":
        traj = traj[..., :-6]
    return traj

def convert_sine_cosine_to_yaw(xu):
    """
    xu is shape (N, T, 37)
    Replace the sine and cosine in xu with yaw and return the new xu
    """
    sine = xu[..., 11]
    cosine = xu[..., 10]
    yaw = torch.atan2(sine, cosine)
    xu_new = torch.cat([xu[..., :10], yaw.unsqueeze(-1), xu[..., 12:]], dim=-1)
    return xu_new

def convert_yaw_to_sine_cosine(xu):
    """
    xu is shape (N, T, 36)
    Replace the yaw in xu with sine and cosine and return the new xu
    """
    yaw = xu[10]
    sine = torch.sin(yaw)
    cosine = torch.cos(yaw)
    xu_new = torch.cat([xu[:10], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[11:]], dim=-1)
    return xu_new

def init_env(visualize=False, config_path = 'card0.yaml'):
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{config_path}').read_text())
    env = AllegroCardSlidingEnv(1, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=visualize,
                                        steps_per_action=60,
                                        friction_coefficient=config['friction_coefficient'],
                                        # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                        device=config['sim_device'],
                                        video_save_path=None,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        gradual_control=False,
                                        gravity=True, # For data generation only
                                        randomize_obj_start=config.get('randomize_obj_start', False)
                                        )

    sim, gym, viewer = env.get_sim()
    sim_env = None
    ros_copy_node = None
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
        'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
        'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
        'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
        'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
    }
    config['ee_names'] = ee_names
    config['obj_dof'] = 3
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    # env.reset()
    return config, env, sim_env, ros_copy_node, chain, sim, gym, viewer

def pull_index(env, config, chain, image_path=None, warmup_iters=35, online_iters=150,
            model_name = 'index_vf', mode='no_vf', task = None,
            vf_weight = 0, other_weight = 10, variance_ratio = 1, 
            ):

    params = config.copy()
    controller = 'csvgd'
    params.pop('controllers')
    params.update(config['controllers'][controller])
    params['controller'] = controller
    obj_dof = params['obj_dof']
    params['chain'] = chain.to(device=params['device'])
    object_location = torch.tensor(env.card_pose).to(
        params['device'])  # TODO: confirm if this is the correct location
    params['object_location'] = object_location
    
    num_fingers = len(params['fingers'])
    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    
    problem = AllegroCard(
        start=start[:4 * num_fingers + obj_dof],
        goal=torch.tensor([0, 0.0, 0]).to(device=params['device']),
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.card_pose,
        table_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type='card',
        world_trans=env.world_trans,
        regrasp_fingers=['middle'],
        contact_fingers=['index'],
        obj_dof=obj_dof,
        obj_joint_dim=1,
        optimize_force=params['optimize_force'],
        default_dof_pos=env.initial_dof_pos[:, :16],
        obj_gravity=params.get('obj_gravity', False),
        # vf stuff
        task = task,
        initial_y = start[-5],
        model_name = model_name, mode=mode, 
        vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio,
    )

    planner = PositionControlConstrainedSVGDMPC(problem, params)
    planner.warmup_iters = warmup_iters
    planner.online_iters = online_iters

    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    
    data = {}
    for t in range(1, 1 + params['T']):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': []}

    state = env.get_state()
    state = state['q'].reshape(-1)[:11].to(device=params['device'])
    y = state[-2]
    # print('Current y:', y)

    goal = torch.tensor([0, -0.02 + state[-2], 0]).to(device=params['device'])

    state = env.get_state()
    state = state['q'].reshape(-1).to(device=params['device'])
    state = state[:planner.problem.dx]
    # print(params['T'], state.shape, initial_samples)
    planner.reset(state, T=params['T'], goal=goal, initial_x=None)
    initial_samples = planner.x.detach().clone()
    planned_trajectories = []
    actual_trajectory = []
    optimizer_paths = []
    plans = None
    for k in range(planner.problem.T):  # range(params['num_steps']):
        state = env.get_state()
        state = state['q'].reshape(4 * num_fingers + params['obj_dof']).to(device=params['device'])
        state = state[:planner.problem.dx]
        s = time.time()
        best_traj, plans = planner.step(state)
        # print('Solve time for step', time.time() - s)
        planned_trajectories.append(plans)
        optimizer_paths.append(copy.deepcopy(planner.path))
        N, T, _ = plans.shape

        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        # record the actual trajectory
        action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
        x = best_traj[0, :planner.problem.dx + planner.problem.du]
        x = x.reshape(1, planner.problem.dx + planner.problem.du)
        action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=env.device)

        xu = torch.cat((state[:-1].cpu(), action[0].cpu()))
        # actual_trajectory.append(xu)
        # actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
        actual_trajectory.append(env.get_full_q().to(device=params['device']))
        
        # print(action)
        action = action[:, :4 * num_fingers]
        action = action.to(device=env.device) + state[:4 * num_fingers].unsqueeze(0).to(env.device)
        env.step(action.to(device=env.device), path_override=image_path)
    
    full_q = env.get_full_q()

    actual_trajectory.append(full_q.to(device=params['device']))
    actual_trajectory = [tensor.to(device=params['device']) for tensor in actual_trajectory]
    actual_trajectory = torch.stack(actual_trajectory, dim=0)

    final_state = full_q.clone().detach().cpu()
    full_trajectory = actual_trajectory.clone().detach().cpu()
   
    return final_state, full_trajectory

def pull_middle(env, config, chain, image_path=None, warmup_iters=35, online_iters=150,
            model_name = 'middle_vf', mode='no_vf', task = None,
            vf_weight = 0, other_weight = 10, variance_ratio = 1, 
            ):

    params = config.copy()
    controller = 'csvgd'
    params.pop('controllers')
    params.update(config['controllers'][controller])
    params['controller'] = controller
    obj_dof = params['obj_dof']
    params['chain'] = chain.to(device=params['device'])
    object_location = torch.tensor(env.card_pose).to(
        params['device'])  # TODO: confirm if this is the correct location
    params['object_location'] = object_location
    
    num_fingers = len(params['fingers'])
    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    
    problem = AllegroCard(
        start=start[:4 * num_fingers + obj_dof],
        goal=torch.tensor([0, 0.0, 0]).to(device=params['device']),
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.card_pose,
        table_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type='card',
        world_trans=env.world_trans,
        regrasp_fingers=['index'],
        contact_fingers=['middle'],
        obj_dof=obj_dof,
        obj_joint_dim=1,
        optimize_force=params['optimize_force'],
        default_dof_pos=env.initial_dof_pos[:, :16],
        obj_gravity=params.get('obj_gravity', False),
        # vf stuff
        initial_y = start[-5],
        task = task,
        model_name = model_name, mode=mode, 
        vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio,
    )

    planner = PositionControlConstrainedSVGDMPC(problem, params)
    planner.warmup_iters = warmup_iters
    planner.online_iters = online_iters

    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    
    data = {}
    for t in range(1, 1 + params['T']):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': []}

    state = env.get_state()
    state = state['q'].reshape(-1)[:11].to(device=params['device'])
    y = state[-2]
    # print('Current y:', y)

    goal = torch.tensor([0, -0.02 + state[-2], 0]).to(device=params['device'])

    state = env.get_state()
    state = state['q'].reshape(-1).to(device=params['device'])
    state = state[:planner.problem.dx]
    # print(params['T'], state.shape, initial_samples)
    planner.reset(state, T=params['T'], goal=goal, initial_x=None)
    initial_samples = planner.x.detach().clone()
    planned_trajectories = []
    actual_trajectory = []
    optimizer_paths = []
    plans = None
    for k in range(planner.problem.T):  # range(params['num_steps']):
        state = env.get_state()
        state = state['q'].reshape(4 * num_fingers + params['obj_dof']).to(device=params['device'])
        state = state[:planner.problem.dx]
        s = time.time()
        best_traj, plans = planner.step(state)
        # print('Solve time for step', time.time() - s)
        planned_trajectories.append(plans)
        optimizer_paths.append(copy.deepcopy(planner.path))
        N, T, _ = plans.shape

        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        # record the actual trajectory
        action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
        x = best_traj[0, :planner.problem.dx + planner.problem.du]
        x = x.reshape(1, planner.problem.dx + planner.problem.du)
        action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=env.device)

        xu = torch.cat((state[:-1].cpu(), action[0].cpu()))
        # actual_trajectory.append(xu)
        # actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
        actual_trajectory.append(env.get_full_q().to(device=params['device']))
        
        # print(action)
        action = action[:, :4 * num_fingers]
        action = action.to(device=env.device) + state[:4 * num_fingers].unsqueeze(0).to(env.device)
        env.step(action.to(device=env.device), path_override=image_path)
    
    full_q = env.get_full_q()

    actual_trajectory.append(full_q.to(device=params['device']))
    actual_trajectory = [tensor.to(device=params['device']) for tensor in actual_trajectory]
    actual_trajectory = torch.stack(actual_trajectory, dim=0)

    final_state = full_q.clone().detach().cpu()
    full_trajectory = actual_trajectory.clone().detach().cpu()
   
    return final_state, full_trajectory


def delete_imgs():
    img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/card/imgs')
    # Check if the directory exists
    if img_save_dir.exists() and img_save_dir.is_dir():
        # Iterate through each item in the directory and delete
        for item in img_save_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # Remove files or symlinks
            elif item.is_dir():
                shutil.rmtree(item)  # Remove directories
        print(f"All contents of {img_save_dir} have been deleted.")
    else:
        print(f"The directory {img_save_dir} does not exist.")

if __name__ == "__main__":

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer = init_env(visualize=True)
   
    warmup_iters = 20
    online_iters = 20

    vf_weight_i = 1
    other_weight_i = 10
    variance_ratio_i = 1

    vf_weight_m = 1
    other_weight_m = 10
    variance_ratio_m = 1

    full_trajs = []
    for i in range(5):
        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/card/imgs/test/trial_{i}')
        pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        env.frame_fpath = img_save_dir
        env.frame_id = 0

        env.reset()

        final_state, full_traj0 = pull_index(env, config, chain, img_save_dir, warmup_iters, online_iters,
                        model_name = 'index_vf', mode='vf', 
                        vf_weight = vf_weight_i, other_weight = other_weight_i, variance_ratio = variance_ratio_i,
                        )
        final_state, full_traj1 = pull_middle(env, config, chain, img_save_dir, warmup_iters, online_iters,
                        model_name = 'middle_vf', mode='vf', 
                        vf_weight = vf_weight_m, other_weight = other_weight_m, variance_ratio = variance_ratio_m,
                        )
        final_state, full_traj2 = pull_index(env, config, chain, img_save_dir, warmup_iters, online_iters,
                        model_name = 'index_vf', mode='vf', 
                        vf_weight = vf_weight_i, other_weight = other_weight_i, variance_ratio = variance_ratio_i,
                        )

        full_traj = torch.cat((full_traj0, full_traj1, full_traj2), dim=0)
        full_trajs.append(full_traj)

    # save this to file
    with open(fpath / 'card'/ 'full_trajs.pkl', 'wb') as f:
        pickle.dump(full_trajs, f)
