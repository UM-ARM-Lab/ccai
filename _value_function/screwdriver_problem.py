import sys
from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

import numpy as np
import pickle as pkl
import torch
import time
import copy
import yaml
import pathlib
import shutil
from functools import partial
import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
import matplotlib.pyplot as plt
from ccai.utils.allegro_utils import state2ee_pos, visualize_trajectory
from scipy.spatial.transform import Rotation as R
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from examples.allegro_valve_roll import PositionControlConstrainedSVGDMPC
from examples.allegro_screwdriver import AllegroScrewdriver
from _value_function.nearest_neighbor import find_nn_0, find_nn_1
from tqdm import tqdm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 20 to 15
def convert_full_to_partial_config(full_config):
    assert(full_config.shape[1] == 20)

    if isinstance(full_config, torch.Tensor):
        full_config = full_config.clone().float()
        partial_config = torch.cat((
        full_config[:,:8],
        full_config[:,12:19]
        ), dim=1).float()
    else:
        full_config = full_config.copy()
        partial_config = np.concatenate((
        full_config[:,:8],
        full_config[:,12:19]
        ), axis=1)

    return partial_config.reshape(-1,15)

# 15 to 20
def convert_partial_to_full_config(partial_config):
    assert(partial_config.shape[1] == 15)

    if isinstance(partial_config, torch.Tensor):
        partial_config = partial_config.clone().float()
        full_config = torch.cat((
        partial_config[:,:8],
        torch.tensor([[0., 0.5, 0.65, 0.65]]) * torch.ones(partial_config.shape[0], 1),
        partial_config[:,8:],
        torch.zeros(partial_config.shape[0], 1)
        ), dim=1)
    else:
        partial_config = partial_config.copy()
        full_config = np.concatenate((
        partial_config[:,:8],
        np.array([[0., 0.5, 0.65, 0.65]]) * np.ones((partial_config.shape[0], 1)),
        partial_config[:,8:],
        np.zeros((partial_config.shape[0], 1))
        ), axis=1)

    return full_config.reshape(-1,20)

def init_env(visualize=False):
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_adam0.yaml').read_text())
    sim_env = None
    ros_copy_node = None
    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=visualize,
                                steps_per_action=60,
                                friction_coefficient=2.5,
                                device=config['sim_device'],
                                video_save_path=None,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=False,
                                gravity=True,
                                randomize_obj_start= True
                                #device = config['device'],
                                )
    sim, gym, viewer = env.get_sim()

    env.frame_fpath = None

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
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos_partial = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)

    return config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial

def pregrasp(env, config, chain, deterministic=True, initialization=None, perception_noise=0, 
             model_name='ensemble', mode='no_vf', vf_weight=0, other_weight=10, variance_ratio=1,
             vis_plan=False, image_path=None, iters=80):
    if mode == 'vf':
        print("mode 'vf' is not supported for pregrasp")
        exit()
    params = config.copy()
    controller = 'csvgd'
    params.pop('controllers')
    params.update(config['controllers'][controller])
    params['controller'] = controller
    params['chain'] = chain.to(device=params['device'])
    object_location = torch.tensor(env.table_pose).to(params['device']).float()
    params['object_location'] = object_location

    obj_dof = 3
    num_fingers = len(params['fingers'])
    device = params['device']
    sim_device = params['sim_device']
    
    # Random initialization of DOF positions if none are provided
    if initialization is None:
        if deterministic is False:
            index_noise_mag = torch.tensor([0.08]*4)
            index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
            middle_thumb_noise_mag = torch.tensor([0.15]*4)
            middle_thumb_noise = middle_thumb_noise_mag * (2 * torch.rand(4) - 1)
            screwdriver_noise = torch.tensor([
                np.random.uniform(-0.08, 0.08),
                np.random.uniform(-0.08, 0.08),
                np.random.uniform(0, 2*np.pi),
                0.0
            ])
        else:
            index_noise = torch.zeros(4)
            middle_thumb_noise = torch.zeros(4)
            screwdriver_noise = torch.zeros(4)

        default_dof_pos = torch.cat((
            torch.tensor([[0.1000, 0.6000, 0.6000, 0.6000]]).float().to(sim_device) + index_noise,
            torch.tensor([[-0.1000, 0.5000, 0.9000, 0.9000]]).float().to(sim_device) + middle_thumb_noise,
            torch.tensor([[0.0000, 0.5000, 0.6500, 0.6500]]).float().to(sim_device),
            torch.tensor([[1.2000, 0.3000, 0.3000, 1.2000]]).float().to(sim_device) + middle_thumb_noise,
            torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000]]).float().to(sim_device) + screwdriver_noise,
        ), dim=1).to(sim_device)
    else:
        default_dof_pos = initialization

    # Reset environment
    env.reset(dof_pos=default_dof_pos)

    # Current environment state
    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)
    screwdriver = start.clone()[-4:-1]
    screwdriver = torch.cat((screwdriver, torch.tensor([0]).to(device=device)), dim=0).reshape(1,4)

    # Choose which fingers to use
    if 'index' in params['fingers']:
        contact_fingers = params['fingers']
    else:
        contact_fingers = ['index'] + params['fingers']    

    # If mode == 'baseline0', do a quick NN-based action
    if mode == 'baseline0':
        full_initial, full_action = find_nn_0(start)
        action = np.concatenate((full_action[:, :8], full_action[:, 12:16]), axis=1)
        action = torch.tensor(action).to(device=device).reshape(1, 12)
        env.step(action.to(device=sim_device), path_override=image_path)
        end_state = env.get_state()['q'].reshape(4 * num_fingers + obj_dof + 1)
        end_state = end_state.unsqueeze(0) 
        end_state_full = torch.cat((
            end_state.clone()[:, :8], 
            torch.tensor([[0., 0.5, 0.65, 0.65]]), 
            end_state.clone()[:, 8:], 
        ), dim=1)
        return end_state_full.cpu()

    # Set up the problem
    pregrasp_problem = AllegroScrewdriver(
        start=start[:4 * num_fingers + obj_dof],
        goal=torch.tensor([0., 0., 0.]).to(device=device),
        T=4,
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type=params['object_type'],
        world_trans=env.world_trans,
        regrasp_fingers=contact_fingers,
        contact_fingers=[],
        obj_dof=obj_dof,
        obj_joint_dim=1,
        optimize_force=params['optimize_force'],
        obj_gravity=params.get('obj_gravity', False),
        model_name=model_name,
        mode=mode,
        vf_weight=vf_weight,
        other_weight=other_weight,
        variance_ratio=variance_ratio,
    )
    
    # Initialize planner
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
    pregrasp_planner.warmup_iters = iters

    # Add optional perception noise to the last 3 DOFs (the screwdriver part)
    noisy_start = start.clone()
    if perception_noise:
        noise = (torch.rand(3, device=device) - 0.5) * perception_noise * 2
        noisy_start[-4:-1] += noise

    # Key memory-usage fix: no_grad()
    with torch.no_grad():
        # Step the planner (no gradients needed)
        best_traj, _ = pregrasp_planner.step(noisy_start[:4 * num_fingers + obj_dof])
        # Detach the resulting trajectory to avoid keeping the graph
        best_traj = best_traj.detach()

    # Take the last step of the best found trajectory
    action = best_traj[-1, :4 * num_fingers]
    action = action.reshape(-1, 4 * num_fingers).to(device=device).detach()

    # Step the environment
    env.step(action.to(device=sim_device), path_override=image_path)
    end_state = env.get_state()['q'].reshape(4 * num_fingers + obj_dof + 1)
    end_state = end_state.unsqueeze(0)
    end_state_full = torch.cat((
        end_state.clone()[:, :8],
        torch.tensor([[0., 0.5, 0.65, 0.65]]),
        end_state.clone()[:, 8:], 
    ), dim=1)
    
    # Add the screwdriver dof back to action for final record
    action = torch.cat((
        action.clone()[:, :8],
        torch.tensor([[0., 0.5, 0.65, 0.65]]).to(device=device),
        action.clone()[:, 8:],
        screwdriver.clone(),
    ), dim=1)

    return end_state_full.cpu(), action.cpu()


def regrasp(env, config, chain, state2ee_pos_partial, perception_noise = 0, initialization = None, 
             model_name = 'ensemble', mode='no_vf', vf_weight = 0, other_weight = 10, variance_ratio = 1,
            image_path = None, vis_plan = False, iters = 200):
    
    params = config.copy()
    controller = 'csvgd'
    params.pop('controllers')
    params.update(config['controllers'][controller])
    params['controller'] = controller
    params['chain'] = chain.to(device=params['device'])
    object_location = torch.tensor(env.table_pose).to(params['device']).float()
    params['object_location'] = object_location

    obj_dof = 3
    num_fingers = len(params['fingers'])
    device = params['device']
    sim_device = params['sim_device']

    #['index', 'middle', 'ring', 'thumb']
    # add random noise to each finger joint other than the ring finger
    if initialization is not None:
        default_dof_pos = initialization
        env.reset(dof_pos= default_dof_pos)
        
    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)
    # print("start: ", start)

    screwdriver = start.clone()[-4:-1]
    initial_yaw = screwdriver[2].item()
    # print("initial yaw: ", initial_yaw)
    #print("start screwdriver: ", screwdriver)
    # screwdriver = torch.cat((screwdriver, torch.tensor([0]).to(device=device)),dim=0).reshape(1,4)

    regrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=torch.tensor([0., 0., initial_yaw]).to(device=device),
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            contact_fingers=['index'],
            regrasp_fingers=['middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.initial_dof_pos[:, :16],
            obj_gravity=params.get('obj_gravity', False),
            min_force_dict=None,
            model_name=model_name,
            mode = mode,
            vf_weight=vf_weight,
            other_weight=other_weight,
            variance_ratio=variance_ratio,
        )
    
    regrasp_planner = PositionControlConstrainedSVGDMPC(regrasp_problem, params)
    regrasp_planner.warmup_iters = iters
    # regrasp_planner.online_iters = online_iters

    # print("n steps: ", params['T'])
    # print("n online iters: ", regrasp_planner.online_iters)

    actual_trajectory = []

    fig = plt.figure()
    axes = {params['fingers'][i]: fig.add_subplot(int(f'1{num_fingers}{i+1}'), projection='3d') for i in range(num_fingers)}
    for finger in params['fingers']:
        axes[finger].set_title(finger)
        axes[finger].set_aspect('equal')
        axes[finger].set_xlabel('x', labelpad=20)
        axes[finger].set_ylabel('y', labelpad=20)
        axes[finger].set_zlabel('z', labelpad=20)
        axes[finger].set_xlim3d(-0.05, 0.1)
        axes[finger].set_ylim3d(-0.06, 0.04)
        axes[finger].set_zlim3d(1.32, 1.43)
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []

    for finger in params['fingers']:
        ee = state2ee_pos_partial(start[:4 * num_fingers], regrasp_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    num_fingers_to_plan = num_fingers

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())

        # add screwdriver noise here
        noise = (torch.rand(3, device=params['device']) - 0.5) * perception_noise * 2
        noisy_start = start.clone()
        noisy_start[-4:-1] += noise

        # -------------------------
        # Key memory-usage fix: no_grad()
        # -------------------------
        with torch.no_grad():
            best_traj, _ = regrasp_planner.step(noisy_start[:4 * num_fingers + obj_dof])
            best_traj = best_traj.detach()

        if torch.isnan(best_traj).any().item():
            env.reset()
            break

        x = best_traj[0, :regrasp_problem.dx+regrasp_problem.du]
        x = x.reshape(1, regrasp_problem.dx+regrasp_problem.du)
        regrasp_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = regrasp_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = regrasp_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        #print("--------------------------------------")

        action = x[:, regrasp_problem.dx:regrasp_problem.dx+regrasp_problem.du].to(device=env.device)
        action = action[:, :4 * num_fingers_to_plan]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers].to(env.device) # NOTE: this is required since we define action as delta action
        
        env.step(action, path_override = image_path)
        regrasp_problem._preprocess(best_traj.unsqueeze(0))
        
        # gym.clear_lines(viewer)
        # state = env.get_state()
        # start = state['q'][:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
        # for finger in params['fingers']:
        #     ee = state2ee_pos_partial(start[:4 * num_fingers], regrasp_problem.ee_names[finger])
        #     finger_traj_history[finger].append(ee.detach().cpu().numpy())
        # for finger in params['fingers']:
        #     traj_history = finger_traj_history[finger]
        #     temp_for_plot = np.stack(traj_history, axis=0)
        #     if k >= 2:
        #         axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')

    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    actual_trajectory.append(state.clone()[:4 * num_fingers + obj_dof])
    actual_trajectory = [tensor.to(device=params['device']) for tensor in actual_trajectory]
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    regrasp_problem.T = actual_trajectory.shape[0]

    state = env.get_state()['q']
    final_state = torch.cat((
                    state.clone()[:, :8], 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]), 
                    state.clone()[:, 8:], 
                    ), dim=1).detach().cpu()
     
    full_trajectory = torch.cat((
                    actual_trajectory.clone()[:, :8].detach().cpu(), 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]) * torch.ones(actual_trajectory.shape[0], 1),
                    actual_trajectory.clone()[:, 8:].detach().cpu(), 
                    torch.zeros(actual_trajectory.shape[0], 1)
                    ), dim=1).numpy()
    
    return final_state, full_trajectory


def solve_turn(env, gym, viewer, params, fpath, initial_pose, state2ee_pos_partial, perception_noise = 0, 
               image_path = None, sim_viz_env=None, ros_copy_node=None, model_name = "ensemble", iters = 200,
               mode='vf', vf_weight = 100.0, other_weight = 0.1, variance_ratio = 5):

    obj_dof = 3
    fpath = pathlib.Path(f'{CCAI_PATH}/data')

    "only turn the screwdriver once"
    screwdriver_goal = params['screwdriver_goal'].cpu()
    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()
    num_fingers = len(params['fingers'])
    action_list = []

    env.reset(dof_pos = initial_pose)

    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    min_force_dict = None
    turn_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=screwdriver_goal.to(device=params['device']),
            # goal=torch.tensor(screwdriver_goal).to(device=params['device']),
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            contact_fingers=['index', 'middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.initial_dof_pos[:, :16],
            turn=True,
            obj_gravity=params.get('obj_gravity', False),
            min_force_dict=min_force_dict,
            model_name=model_name,
            mode = mode,
            vf_weight=vf_weight,
            other_weight=other_weight,
            variance_ratio=variance_ratio,
        )
    turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)
    turn_planner.warmup_iters = iters

    actual_trajectory = []

    # fig = plt.figure()
    # axes = {params['fingers'][i]: fig.add_subplot(int(f'1{num_fingers}{i+1}'), projection='3d') for i in range(num_fingers)}
    # for finger in params['fingers']:
    #     axes[finger].set_title(finger)
    #     axes[finger].set_aspect('equal')
    #     axes[finger].set_xlabel('x', labelpad=20)
    #     axes[finger].set_ylabel('y', labelpad=20)
    #     axes[finger].set_zlabel('z', labelpad=20)
    #     axes[finger].set_xlim3d(-0.05, 0.1)
    #     axes[finger].set_ylim3d(-0.06, 0.04)
    #     axes[finger].set_zlim3d(1.32, 1.43)
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []

    for finger in params['fingers']:
        ee = state2ee_pos_partial(start[:4 * num_fingers], turn_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    num_fingers_to_plan = num_fingers
    # info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())
        start_time = time.time()

        noise = (torch.rand(3, device=params['device']) - 0.5) * perception_noise * 2
        noisy_start = start.clone()
        noisy_start[-4:-1] += noise

        # -------------------------
        # Key memory-usage fix: no_grad()
        # -------------------------
        with torch.no_grad():
            best_traj, _ = turn_planner.step(noisy_start[:4 * num_fingers + obj_dof])
            best_traj = best_traj.detach()

        if torch.isnan(best_traj).any().item():
            env.reset()
            break

        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        #print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        action = action[:, :4 * num_fingers_to_plan]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers].to(env.device) # NOTE: this is required since we define action as delta action
        
        env.step(action, path_override = image_path)
        action_list.append(action)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        screwdriver_state = env.get_state()['q'][:, -obj_dof-1: -1].cpu()
        screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
            torch.tensor(screwdriver_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()

        if torch.isnan(distance2goal).item():
            env.reset()
            break
        if torch.isnan(distance2goal).any().item():
            env.reset()
            break

        # gym.clear_lines(viewer)
        # state = env.get_state()
        # start = state['q'][:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
        # for finger in params['fingers']:
        #     ee = state2ee_pos_partial(start[:4 * num_fingers], turn_problem.ee_names[finger])
        #     finger_traj_history[finger].append(ee.detach().cpu().numpy())
        # for finger in params['fingers']:
        #     traj_history = finger_traj_history[finger]
        #     temp_for_plot = np.stack(traj_history, axis=0)
        #     if k >= 2:
        #         axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')

    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    actual_trajectory.append(state.clone()[:4 * num_fingers + obj_dof])
    actual_trajectory = [tensor.to(device=params['device']) for tensor in actual_trajectory]
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    turn_problem.T = actual_trajectory.shape[0]
    screwdriver_state = actual_trajectory[:, -obj_dof:].cpu()
    screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
    distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
        torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0],1,1), cos_angle=False).detach().cpu()

    final_distance_to_goal = torch.min(distance2goal.abs())

    state = env.get_state()['q']
    final_state = torch.cat((
                    state.clone()[:, :8], 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]), 
                    state.clone()[:, 8:], 
                    ), dim=1).detach().cpu().numpy()
     
    full_trajectory = torch.cat((
                    actual_trajectory.clone()[:, :8].detach().cpu(), 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]) * torch.ones(actual_trajectory.shape[0], 1),
                    actual_trajectory.clone()[:, 8:].detach().cpu(), 
                    torch.zeros(actual_trajectory.shape[0], 1)
                    ), dim=1).numpy()
    
    return final_distance_to_goal.cpu().detach().item(), final_state, full_trajectory

def do_turn( initial_pose, config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, image_path = None, 
            iters = 200, perception_noise = 0, turn_angle = np.pi/2, mode='vf', vf_weight = 100.0, other_weight = 0.1, variance_ratio = 5):

    params = config.copy()
    controller = 'csvgd'
    succ = False
    fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_0')
    pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
    # set up params
    params.pop('controllers')
    params.update(config['controllers'][controller])
    params['controller'] = controller
    #change goal depending on initial screwdriver pose
    screwdriver_pose = initial_pose[0,-4:-1]
    goal = torch.tensor([0, 0, -turn_angle]) + screwdriver_pose.clone()

    params['screwdriver_goal'] = goal.to(device=params['device'])
    params['chain'] = chain.to(device=params['device'])
    object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
    params['object_location'] = object_location

    final_distance_to_goal, final_pose, full_trajectory = solve_turn(env, gym, viewer, params, fpath, initial_pose, state2ee_pos_partial, image_path = None,
                                                                     sim_viz_env=sim_env, ros_copy_node=ros_copy_node, perception_noise=perception_noise, iters=iters,
                                                                     mode=mode, vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio)
    
    if final_distance_to_goal < 30 / 180 * np.pi:
        succ = True

    return initial_pose, final_pose, succ, full_trajectory

class emailer():
    def __init__(self):
        self.sender_email = "eburner813@gmail.com"
        self.receiver_email = "adamhung@umich.edu"  # You can send it to yourself
        self.password = "yhpffhhnwbhpluty"
    def send(self, *args, **kwargs):
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = "program finished"
        body = "program finished"
        msg.attach(MIMEText(body, 'plain'))
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(self.sender_email, self.password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text)
            print("Email sent")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            server.quit()

# def show_state(env, state, t=1):
#     env.reset(dof_pos=state)
#     time.sleep(t)

def delete_imgs():
    img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs')
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
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    
    img_save_dir = None
    pregrasp_iters = 80#200
    regrasp_iters = 100
    perception_noise = 0.0

    pregrasp_pose, planned_pose = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise, 
                        image_path = img_save_dir, initialization = None, mode='no_vf', iters = pregrasp_iters)

    regrasp_pose, regrasp_traj = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=perception_noise, 
                            image_path = img_save_dir, initialization = pregrasp_pose, mode='no_vf', iters = regrasp_iters,
                            vf_weight = 100.0, other_weight = 0.1, variance_ratio = 5)
    
    _, turn_pose, succ, turn_traj = do_turn(regrasp_pose, config, env, 
                    sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                    perception_noise=perception_noise, image_path = img_save_dir, iters = regrasp_iters,
                    mode='vf', vf_weight = 100.0, other_weight = 0.1, variance_ratio = 5)
    

    # print("done regrasp")
    # while True:
    #     for idx, state in enumerate(regrasp_traj):
    #         print(f"step {idx}")
    #         env.reset(dof_pos=torch.tensor(state).reshape(1,20))
    #         time.sleep(0.1)
    
    # print("enter to continue")
    # wait = input()
    # while True:
    #     for traj in regrasp_traj:
    #         env.reset(dof_pos=torch.tensor(traj).reshape(1,20))
    #         time.sleep(0.2)