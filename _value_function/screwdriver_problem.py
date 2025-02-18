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
from ccai.models.trajectory_samplers import TrajectorySampler
from _value_function.nearest_neighbor import find_nn_0, find_nn_1
from tqdm import tqdm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def _full_to_partial(traj, mode):
    if mode == 'index':
        traj = torch.cat((traj[..., :-9], traj[..., -6:]), dim=-1)
    if mode == 'thumb_middle':
        traj = traj[..., :-6]
    if mode == 'pregrasp':
        traj = traj[..., :-9]
    return traj
def convert_sine_cosine_to_yaw(xu):
    """
    xu is shape (N, T, 37)
    Replace the sine and cosine in xu with yaw and return the new xu
    """
    sine = xu[..., 15]
    cosine = xu[..., 14]
    yaw = torch.atan2(sine, cosine)
    xu_new = torch.cat([xu[..., :14], yaw.unsqueeze(-1), xu[..., 16:]], dim=-1)
    return xu_new
def convert_yaw_to_sine_cosine(xu):
    """
    xu is shape (N, T, 36)
    Replace the yaw in xu with sine and cosine and return the new xu
    """
    yaw = xu[14]
    sine = torch.sin(yaw)
    cosine = torch.cos(yaw)
    xu_new = torch.cat([xu[:14], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[15:]], dim=-1)
    return xu_new
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

def init_env(visualize=False, config_path = 'allegro_screwdriver_adam0.yaml'):
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{config_path}').read_text())
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
             model_name=None, mode='no_vf', vf_weight=0, other_weight=10, variance_ratio=1,
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

def regrasp(env, config, chain, state2ee_pos_partial, use_diffusion = False, diffusion_path = None, perception_noise = 0, initialization = None,
             model_name = 'ensemble_rg', mode='no_vf', use_contact_cost = False, vf_weight = 0, other_weight = 10, variance_ratio = 1,
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

    last_diffused_q = None
    # diffusion ############################################################################################################
    
    if use_diffusion:

        state = env.get_state()
        state = state['q'].reshape(-1)[:15].to(device=params['device'])

        contact = -torch.ones(params['N'], 3).to(device=params['device'])
        contact[:, 0] = 1
        # contact[:, :] = 1 # THIS ONE FOR TURNING
        start = state.clone()

        trajectory_sampler = get_diffusion(params, diffusion_path)
    
        start_for_diff = convert_yaw_to_sine_cosine(start)

        initial_samples, _, _ = trajectory_sampler.sample(N=params['N'], start=start_for_diff.reshape(1, -1),
                                                                    H=params['T'] + 1,
                                                                    constraints=contact,
                                                                    project=params['project_state'],)
    
        initial_samples = convert_sine_cosine_to_yaw(initial_samples)
        fpath = pathlib.Path(f'{CCAI_PATH}/data')
        pkl.dump(initial_samples, open(f'{fpath}/diffusion/initial_samples/raw/regrasp_init.pkl', 'wb'))

        if initial_samples is not None:
    
            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :15]#:planner.problem.dx]
            initial_u = initial_samples[:, :-1, -15:]#-planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)


        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:15]#planner.problem.dx]

        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:15]#planner.problem.dx]

    ############################################################################################################
    
        last_diffused_q = initial_samples[:, -1, :15].clone()

    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)

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
            last_diffused_q=last_diffused_q,
            use_contact_cost = use_contact_cost,
        )
   
    regrasp_planner = PositionControlConstrainedSVGDMPC(regrasp_problem, params)
    regrasp_planner.warmup_iters = iters

    if params['visualize_plan'] and use_diffusion:
        fname = 'diffusion'
        iter_set = [('initial_samples', initial_samples)]
        for (name, traj_set) in iter_set:
            for k in range(params['N']):
                traj_for_viz = traj_set[k, :, :15]#:planner.problem.dx]
                tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                device=traj_for_viz.device)  # add the joint for the screwdriver cap
                traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{name}/{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, regrasp_problem.contact_scenes, viz_fpath,
                                    regrasp_problem.fingers, regrasp_problem.obj_dof + 1)

    torch.cuda.empty_cache()
    
    if use_diffusion:
        regrasp_planner.reset(state, T=params['T'], initial_x=initial_samples)

    actual_trajectory = []
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []

    for finger in params['fingers']:
        ee = state2ee_pos_partial(start[:4 * num_fingers], regrasp_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    num_fingers_to_plan = num_fingers

    all_regrasp_plans = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())

        # add screwdriver noise here
        noise = (torch.rand(3, device=params['device']) - 0.5) * perception_noise * 2
        noisy_start = start.clone()
        noisy_start[-4:-1] += noise

        with torch.no_grad():
            best_traj, _ = regrasp_planner.step(noisy_start[:4 * num_fingers + obj_dof])
            best_traj = best_traj.detach()

        if torch.isnan(best_traj).any().item():
            env.reset()
            break
        # get plan here
        single_regrasp_plan = best_traj.clone()
        single_regrasp_plan = single_regrasp_plan[:, :regrasp_problem.dx+regrasp_problem.du]
        all_regrasp_plans.append(single_regrasp_plan)

        x = best_traj[0, :regrasp_problem.dx+regrasp_problem.du]
        x = x.reshape(1, regrasp_problem.dx+regrasp_problem.du)
        regrasp_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = regrasp_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = regrasp_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)

        action = x[:, regrasp_problem.dx:regrasp_problem.dx+regrasp_problem.du].to(device=env.device)
        action = action[:, :4 * num_fingers_to_plan]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers].to(env.device) # NOTE: this is required since we define action as delta action
       
        env.step(action, path_override = image_path)
        regrasp_problem._preprocess(best_traj.unsqueeze(0))

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
   
    return final_state, full_trajectory, all_regrasp_plans


def solve_turn(env, gym, viewer, params, initial_pose, state2ee_pos_partial, use_diffusion = False, diffusion_path=None, perception_noise = 0,
               image_path = None, sim_viz_env=None, ros_copy_node=None, model_name = "ensemble_t", iters = 200,
               mode='vf', initial_yaw = None, vf_weight = 100.0, other_weight = 0.1, variance_ratio = 5):

    obj_dof = 3

    "only turn the screwdriver once"
    screwdriver_goal = params['screwdriver_goal'].cpu()
    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()
    num_fingers = len(params['fingers'])
    action_list = []

    env.reset(dof_pos = initial_pose)


    # diffusion setup ############################################################################################################

    if use_diffusion:

        state = env.get_state()
        state = state['q'].reshape(-1)[:15].to(device=params['device'])

        contact = -torch.ones(params['N'], 3).to(device=params['device'])
        contact[:, :] = 1
        start = state.clone()

        trajectory_sampler = get_diffusion(params, diffusion_path)
    
        start_for_diff = convert_yaw_to_sine_cosine(start)

        initial_samples, _, _ = trajectory_sampler.sample(N=params['N'], start=start_for_diff.reshape(1, -1),
                                                                    H=params['T'] + 1,
                                                                    constraints=contact,
                                                                    project=params['project_state'],)
    
        initial_samples = convert_sine_cosine_to_yaw(initial_samples)
        fpath = pathlib.Path(f'{CCAI_PATH}/data') 
        pkl.dump(initial_samples, open(f'{fpath}/diffusion/initial_samples/raw/turn_init.pkl', 'wb'))

        if initial_samples is not None:
    
            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :15]#:planner.problem.dx]
            initial_u = initial_samples[:, :-1, -21:]#-planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)


        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:15]#:planner.problem.dx]

        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:15]#:planner.problem.dx]
    
    ############################################################################################################

    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
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
            initial_yaw = initial_yaw,
            vf_weight=vf_weight,
            other_weight=other_weight,
            variance_ratio=variance_ratio,
        )
    turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)
    turn_planner.warmup_iters = iters

    if params['visualize_plan'] and use_diffusion:
        fname = 'diffusion'
        iter_set = [('initial_samples', initial_samples)]
        for (name, traj_set) in iter_set:
            for k in range(params['N']):
                traj_for_viz = traj_set[k, :, :15]#planner.problem.dx]
                tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                device=traj_for_viz.device)  # add the joint for the screwdriver cap
                traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{name}/{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes, viz_fpath,
                                    turn_problem.fingers, turn_problem.obj_dof + 1)

    torch.cuda.empty_cache()
    
    if use_diffusion:
        turn_planner.reset(state, T=params['T'], initial_x=initial_samples)

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

    all_turn_plans = []

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
        # get plan here
        single_turn_plan = best_traj.clone()
        single_turn_plan = single_turn_plan[:, :turn_problem.dx+turn_problem.du]
        all_turn_plans.append(single_turn_plan)

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
   
    return final_distance_to_goal.cpu().detach().item(), final_state, full_trajectory, all_turn_plans

def do_turn( initial_pose, config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
            use_diffusion = False, diffusion_path = None,
            image_path = None,
            iters = 200, perception_noise = 0, turn_angle = np.pi/2, model_name = "ensemble_t", mode='no_vf', initial_yaw = None,
            vf_weight = 0.0, other_weight = 10.0, variance_ratio = 0.0):

    params = config.copy()
    controller = 'csvgd'
    succ = False
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

    final_distance_to_goal, final_pose, full_trajectory, turn_plan = solve_turn(env, gym, viewer, params, initial_pose, state2ee_pos_partial, image_path = image_path,
                                                                     sim_viz_env=sim_env, ros_copy_node=ros_copy_node, perception_noise=perception_noise, iters=iters,
                                                                     use_diffusion=use_diffusion, diffusion_path=None,
                                                                     mode=mode, model_name=model_name, initial_yaw = initial_yaw, vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio)
   
    if final_distance_to_goal < 30 / 180 * np.pi:
        succ = True

    return initial_pose, final_pose, succ, full_trajectory, turn_plan

def get_diffusion(params, path = None):

    if path is None:
        model_path = params.get('model_path', None)
    else:
        model_path = path

    trajectory_sampler = TrajectorySampler(T=params['T'] + 1, dx=(15 + (1 if params['sine_cosine'] else 0)), du=21, type=params['type'],
                                               timesteps=256, hidden_dim=128,
                                               context_dim=3, generate_context=False,
                                               constrain=params['projected'],
                                               problem=None,
                                               inits_noise=None, noise_noise=None,
                                               guided=params['use_guidance'],
                                               state_control_only=params.get('state_control_only', False),
                                               vae=None)
    
    trajectory_sampler.model.diffusion_model.classifier = None
    d = torch.load(f'{CCAI_PATH}/{model_path}', map_location=torch.device(params['device']))
    d = {k:v for k, v in d.items() if 'classifier' not in k}
    trajectory_sampler.load_state_dict(d, strict=False)
    trajectory_sampler.to(device=params['device'])
    trajectory_sampler.send_norm_constants_to_submodels()
    # print('Loaded trajectory sampler')

    return trajectory_sampler
    


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
   
    pregrasp_iters = 50
    regrasp_iters = 40
    turn_iters = 50
    perception_noise = 0.0

    vf_weight_rg = 5.0
    other_weight_rg = 1.9
    variance_ratio_rg = 10.0

    vf_weight_t = 3.3
    other_weight_t = 1.9
    variance_ratio_t = 1.625

    img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/regrasp_trial_test')
    pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  


    for i in range(1):
       
        pregrasp_pose, planned_pose = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise,
                            image_path = img_save_dir, initialization = None, mode='no_vf', iters = pregrasp_iters)

        # regrasp_pose, regrasp_traj, rg_plan = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
        #                         use_diffusion=False,
        #                         image_path = img_save_dir, initialization = pregrasp_pose, mode='vf', iters = regrasp_iters, model_name = 'ensemble_rg',
        #                         vf_weight = vf_weight_rg, other_weight = other_weight_rg, variance_ratio = variance_ratio_rg)
       
        # _, turn_pose, succ, turn_traj, t_plan = do_turn(regrasp_pose, config, env,
        #                 sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
        #                 use_diffusion = False, 
        #                 perception_noise=perception_noise, image_path = img_save_dir, iters = turn_iters, initial_yaw = regrasp_pose[0, -2], model_name= 'ensemble_t',
        #                 mode='vf', vf_weight = vf_weight_t, other_weight = other_weight_t, variance_ratio = variance_ratio_t)
        
        regrasp_pose, regrasp_traj, rg_plan = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                                use_diffusion=True,
                                image_path = img_save_dir, initialization = pregrasp_pose, mode='no_vf', iters = regrasp_iters, model_name = 'ensemble_rg',
                                )
       
        _, turn_pose, succ, turn_traj, t_plan = do_turn(regrasp_pose, config, env,
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                        use_diffusion = True, 
                        perception_noise=perception_noise, image_path = img_save_dir, iters = turn_iters, initial_yaw = regrasp_pose[0, -2], model_name= 'ensemble_t',
                        mode='no_vf')
       

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