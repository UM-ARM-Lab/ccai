# from isaacsim_hand_envs.allegro import AllegroScrewdriverEnv # it needs to be imported before numpy and torch
import os
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
from allegro_valve_roll import PositionControlConstrainedSVGDMPC
from allegro_screwdriver_w_force import AllegroScrewdriver, AllegroContactProblem
from scipy.spatial.transform import Rotation as R

from utils.allegro_utils import *
from utils.math_utils import *
from utils.isaacgym_utils import *
from tqdm import tqdm
import argparse
import re


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# torch.cuda.set_device(1)
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

    
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the screwdriver once"
    obj_dof = params['obj_dof']
    if params['arm_type'] == 'robot':
        camera_params = "screwdriver_w_arm"
    elif params['arm_type'] == 'None' or params['arm_type'] == 'floating_3d' or params['arm_type'] == 'floating_6d':
        camera_params = "screwdriver"
    goal = params['goal'].cpu()
    num_fingers = len(params['fingers'])
    arm_dof = get_arm_dof(params['arm_type'])
    robot_dof = 4 * num_fingers + arm_dof
    if params['object_type'] == 'screwdriver':
        obj_joint_dim = 1 # compensate for the screwdriver cap
    else:
        obj_joint_dim = 0

    env.reset()
    if params['visualize']:
        if params['mode'] == 'hardware':
            sim_viz_env.frame_fpath = fpath
            sim_viz_env.frame_id = 0
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    state = env.get_state()
    action_list = []
    start = state.reshape(robot_dof + obj_dof).to(device=params['device'])       

    # pregrasp
    # num_pregrasp_attempts = 0
    # pregrasp_succ = False
    # while pregrasp_succ == False:
    #     if num_pregrasp_attempts >= 3:
    #         print("skip pregrasp")
    #         break
    #     pregrasp_dx = pregrasp_du = robot_dof
    #     pregrasp_problem = AllegroContactProblem(
    #         dx=pregrasp_dx,
    #         du=pregrasp_du,
    #         start=start[:pregrasp_dx + obj_dof],
    #         goal=None,
    #         T=4,
    #         chain=params['chain'],
    #         device=params['device'],
    #         object_asset_pos=env.obj_pose,
    #         object_type=params['object_type'],
    #         world_trans=env.world_trans,
    #         fingers=params['fingers'],
    #         obj_dof_code=params['obj_dof_code'],
    #         obj_joint_dim=obj_joint_dim,
    #         fixed_obj=True,
    #         geometry_grad=params['geometry_grad'],
    #         arm_type=params['arm_type'],
    #         obj_id=params['obj_id']
    #     )

    #     pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
    #     pregrasp_planner.warmup_iters = 50 
    #     # else:
    #     #     raise ValueError('Invalid controller')
        
    #     start_time = time.time()
    #     best_traj, _ = pregrasp_planner.step(start[:pregrasp_dx])
    #     print(f"pregrasp solve time: {time.time() - start_time}")

    #     if params['visualize_plan']:
    #         traj_for_viz = best_traj[:, :pregrasp_problem.dx]
    #         tmp = start[pregrasp_dx:pregrasp_dx+obj_dof].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
    #         tmp_2 = torch.zeros((traj_for_viz.shape[0], 1)).to(traj_for_viz.device) # the top jint
    #         traj_for_viz = torch.cat((traj_for_viz, tmp, tmp_2), dim=1)    
    #         viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
    #         img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
    #         gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
    #         pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
    #         pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
    #         visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, pregrasp_problem.fingers, pregrasp_problem.obj_dof + obj_joint_dim,
    #                             camera_params=camera_params, arm_dof=arm_dof)


    #     for x in best_traj[:, :pregrasp_dx]:
    #         action = x.reshape(-1, pregrasp_dx).to(device=env.device) # move the rest fingers
    #         if params['mode'] == 'hardware':
    #             set_state = env.get_state(return_dict=True)['q'].to(device=env.device)
    #             if params['task'] == 'screwdriver_turning':
    #                 set_state = torch.cat((set_state, torch.zeros(1).unsqueeze(0).float().to(env.device)), dim=-1)
    #             sim_viz_env.set_pose(set_state)
    #             sim_viz_env.step(action)
    #         env.step(action)
    #         action_list.append(action)
    #         if params['mode'] == 'hardware_copy':
    #             ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, pregrasp_dx)[0], params['fingers']))
    #     if params['mode'] == 'simulation':
    #         pregrasp_succ = env.check_pregrasp_validity(env.get_state().cpu()[0])
    #     elif params['mode'] == 'hardware':
    #         pregrasp_succ = True
    #     if pregrasp_succ == False:
    #         print("pregrasp failed, replanning")
    #         env.reset() 
    #         num_pregrasp_attempts += 1
    
    state = env.get_state()
    start = state.reshape(robot_dof + obj_dof).to(device=params['device'])
    if config['method'] == 'csvgd':
        if config['task'] == 'screwdriver_turning':
            manipulation_problem = AllegroScrewdriver(
                start=start[:robot_dof + obj_dof],
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.obj_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                finger_stiffness=params['kp'],
                arm_stiffness=500,
                world_trans=env.world_trans,
                fingers=params['fingers'],
                optimize_force=params['optimize_force'],
                force_balance=False,
                collision_checking=params['collision_checking'],
                obj_gravity=params['obj_gravity'],
                contact_region=params['contact_region'],
                geometry_grad=params['geometry_grad'],
                arm_type=params['arm_type'],
                obj_id=params['obj_id'],
            )

        manipulation_planner = PositionControlConstrainedSVGDMPC(manipulation_problem, params)
    actual_trajectory = []
    duration = 0

    info_list = []
    validity_flag = True
    warmup_time = 0

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state.reshape(robot_dof + obj_dof).to(device=params['device'])
        current_theta = state[:, -obj_dof:].detach().cpu().numpy()
        actual_trajectory.append(start[:robot_dof + obj_dof].clone())
        start_time = time.time()
        if config['method'] == 'planning':
            action = manipulation_planner.step(start[:robot_dof + obj_dof])
            solve_time = time.time() - start_time
            if k>= 1:
                duration += solve_time
            action = action.unsqueeze(0).to(env.device)
        else:
            best_traj, trajectories = manipulation_planner.step(start[:robot_dof + obj_dof])
        
            solve_time = time.time() - start_time
            print(f"solve time: {solve_time}")
            if k == 0:
                warmup_time = solve_time
            else:
                duration += solve_time
            planned_theta_traj = best_traj[:, robot_dof: robot_dof + obj_dof].detach().cpu().numpy()
            # print(f"current theta: {current_theta}")
            # print(f"planned theta: {planned_theta_traj}")

            if params['visualize_plan']:
                # manipulation_problem.viz_contact_scenes.visualize_robot(partial_to_full_state(start[:manipulation_problem.robot_dof], fingers=params['fingers'], arm_dof=arm_dof), start[robot_dof:robot_dof + obj_dof])
                traj_for_viz = best_traj[:, :manipulation_problem.dx]
                traj_for_viz = torch.cat((start[:manipulation_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                if obj_joint_dim > 0:
                    tmp = torch.zeros((traj_for_viz.shape[0], obj_joint_dim), device=best_traj.device) # add the joint for the screwdriver cap
                    traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])
            
                viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, manipulation_problem.viz_contact_scenes, viz_fpath, manipulation_problem.fingers, manipulation_problem.obj_dof + obj_joint_dim, 
                                    camera_params=camera_params, arm_dof=arm_dof)
            
            x = best_traj[0, :manipulation_problem.dx+manipulation_problem.du]
            x = x.reshape(1, manipulation_problem.dx+manipulation_problem.du)
            manipulation_problem._preprocess(best_traj.unsqueeze(0))
            equality_constr_dict = manipulation_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
            inequality_constr_dict = manipulation_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
            print("--------------------------------------")

            action = x[:, manipulation_problem.dx:manipulation_problem.dx+manipulation_problem.du].to(device=env.device)
            action = action[:, :robot_dof]
            action = action + start.unsqueeze(0)[:, :robot_dof].to(action.device) # NOTE: this is required since we define action as delta action
            if params['mode'] == 'hardware':
                set_state = env.get_state(return_dict=True)['q'].to(device=env.device)
                if params['task'] == 'screwdriver_turning':
                    set_state = torch.cat((set_state, torch.zeros(1).unsqueeze(0).float().to(env.device)), dim=-1)
                sim_viz_env.set_pose(set_state)
                sim_viz_env.step(action)
            elif params['mode'] == 'hardware_copy':
                ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        env.step(action)
        action_list.append(action)
        # manipulation_problem._preprocess(best_traj.unsqueeze(0))
        
        obj_state = env.get_state()[:, -obj_dof:].cpu()
        if params['task'] == 'screwdriver_turning':
            distance2goal = euler_diff(obj_state, goal.unsqueeze(0)).detach().cpu().abs().item()
        elif params['task'] == 'valve_turning':
            distance2goal = (obj_state[0] - goal).detach().item()
        elif params['task'] == 'peg_turning' or params['task'] == 'peg_alignment' or params['task'] == 'reorientation':
            distance2goal = euler_diff(obj_state[:, -3:], goal[-3:].unsqueeze(0)).detach().cpu().abs().item()
        
        if params['mode'] != 'hardware':
            if not env.check_validity(env.get_state().cpu()[0]):
                validity_flag = False

        print(distance2goal, validity_flag)
        if config['method'] == 'planning':
            info = {'distance2goal': distance2goal, 'validity_flag': validity_flag}
        else:
            info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal, 'validity_flag': validity_flag}}
        info_list.append(info)

        # if params['simulator'] == 'isaac_gym':
        #     gym.clear_lines(viewer)
        state = env.get_state()
        start = state.squeeze(0).to(device=params['device'])
        if not validity_flag:
            break
    validity_flag = env.check_validity(env.get_state().cpu()[0])

    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()
    state = state.reshape(robot_dof + obj_dof).to(device=params['device'])
    actual_trajectory.append(state.clone()[:robot_dof + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, robot_dof + obj_dof)
    # manipulation_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    obj_state = actual_trajectory[:, -obj_dof:].cpu()
    if params['task'] == 'screwdriver_turning':
        distance2goal = euler_diff(obj_state, goal.unsqueeze(0).repeat(obj_state.shape[0], 1)).detach().cpu().abs()
    elif params['task'] == 'valve_turning':
        distance2goal = (obj_state - goal).detach().cpu().abs()
    elif params['task'] == 'peg_turning' or params['task'] == 'peg_alignment' or params['task'] == 'reorientation':
        distance2goal = euler_diff(obj_state[:, -3:], goal[-3:].unsqueeze(0).repeat(obj_state.shape[0], 1)).detach().cpu().abs()

    # final_distance_to_goal = torch.min(distance2goal.abs())
    final_distance_to_goal = distance2goal[-1].cpu().detach().item()
    print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}, validity: {validity_flag}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal)
    env.reset()
    ret = {'warmup_time': warmup_time,
    'final_distance_to_goal': final_distance_to_goal, 
    'validity_flag': validity_flag,
    'avg_online_time': duration / (params["num_steps"] - 1)}
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the turning after catching task')
    parser.add_argument('--data_dir', type=str, 
                        # default=None,
                        default="/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-09-10_22-42-31/nn/state_data_105.pt",
                        )
    args = parser.parse_args()
    # get config
    task = 'screwdriver_turning'

    method = 'csvgd'

    if task == 'screwdriver_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]        
        config['num_env_force'] = 1

    obj_dof = sum(config['obj_dof_code'])
    config['obj_dof'] = obj_dof
    config['task'] = task
    config['method'] = method

    sim_env = None
    ros_copy_node = None

    if args.data_dir is not None:
        match = re.search(r'state_data_(\d+)\.pt', args.data_dir)
        if match:
            obj_id = int(match.group(1))
    else:
        obj_id = None
    config['obj_id'] = obj_id
    data = torch.load(args.data_dir, map_location=config['sim_device'])
    results = {}
    for controller in config['controllers'].keys():
        results[controller] = {}
    for trial_idx in range(5):
        setup_dict = get_screwdriver_setup(data[trial_idx]['screwdriver_config'], data[trial_idx]['screwdriver_config']['center_height'])
        print(setup_dict)
        obj_pos = data[trial_idx]['tip_position'].squeeze()
        obj_quat = data[trial_idx]['obj_quat'].squeeze() # wxyz
        hand_pos = data[trial_idx]['hand_pos'].squeeze()
        hand_quat = data[trial_idx]['hand_quat'].squeeze() # wxyz
        # hand_quat = torch.tensor([hand_quat[1], hand_quat[2], hand_quat[3], hand_quat[0]]).to(config['sim_device']).float() # convert to xyzw
        robot_initial_q = data[trial_idx]['robot_joint_pos'].squeeze().to(config['sim_device']).float()[7:]
        robot_initial_q = swap_joint_angle(robot_initial_q)

        # hand_quat = transform_robot_to_object_frame(hand_quat.clone(), obj_quat).astype(np.float32)
        hand_pos, hand_quat, obj_pos, obj_quat = transform_pose_to_object_aligned_frame(hand_pos, hand_quat, obj_pos, obj_quat)
        
        # Center the object at (0,0,1.2) by subtracting x,y coordinates and setting z to 1.2
        # Move both hand and object positions accordingly
        offset_xy = obj_pos[:2].copy()  # Make a copy of x,y coordinates
        offset_z = obj_pos[2].copy() - 1.22  # Calculate z offset to move object to z=1.2
        
        obj_pos[:2] = 0.0  # Set object x,y to 0
        obj_pos[2] = 1.22   # Set object z to 1.2
        hand_pos[:2] -= offset_xy  # Subtract the same x,y offset from hand position
        hand_pos[2] -= offset_z    # Subtract the same z offset from hand position
        
        hand_pos, hand_quat, obj_pos, obj_quat = hand_pos.astype(np.float32), hand_quat.astype(np.float32), obj_pos.astype(np.float32), obj_quat.astype(np.float32)
        init_pos = torch.cat((robot_initial_q, torch.zeros(4).to(config['sim_device']))).float()
            
        # the code does not reset screwdriver to upright be keep the current config
        # obj_euler = quaternion_to_intrinsic_xyz_euler(obj_quat)
        # obj_euler = torch.tensor(obj_euler).to(config['sim_device']).float()
        # init_pos = torch.cat((robot_initial_q, obj_euler, torch.zeros(1).to(config['sim_device'])))
        if config['simulator'] == 'isaac_gym':
            from isaac_victor_envs.utils import get_assets_dir
            from utils.isaacgym_utils import get_env
            env = AllegroScrewdriverTurningEnv(num_envs=1, 
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
                                                init_pos=init_pos,
                                                # obj_pose=obj_pos.detach().cpu().numpy(),
                                                # hand_p= hand_pos.detach().cpu().numpy(),
                                                # hand_r=hand_quat.detach().cpu().numpy(),
                                                obj_pose=obj_pos,
                                                hand_p=hand_pos,
                                                hand_r=hand_quat,
                                                obj_id = obj_id,
                                                )
            sim, gym, viewer = env.get_sim()
        elif config['simulator'] == 'isaac_sim':
            from isaacsim_hand_envs.utils import get_assets_dir
            env = AllegroScrewdriverEnv(friction_coefficient=1.0,
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
        
        state = env.get_state()
        # try:
        #     while True:
        #         start = env.get_state()[:, :-3]
        #         env.step(start)
        #         print('waiting for you to finish camera adjustment, ctrl-c when done')
        #         time.sleep(0.1)
        # except KeyboardInterrupt:
        #     pass
        # ee_names = {
        #         'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
        #         'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
        #         'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
        #         'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        #         }
        ee_names = {
                'index': 'hitosashi_ee',
                'middle': 'naka_ee',
                'ring': 'kusuri_ee',
                'thumb': 'oya_ee',
                }
        config['ee_names'] = ee_names

        chain = pk.build_chain_from_urdf(open(asset).read())

        # setup the helper function
        frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
        frame_indices = torch.tensor(frame_indices)
        state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans, arm_dof=arm_dof)
        forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])

        config['goal'] = torch.tensor(config['goal']).to(device=config['device']).float()
        for controller in config['controllers'].keys():
            if controller == config['method']: # only evaluate the method specified in the config
                env.reset()
                fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{trial_idx + 1}')
                pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
                # set up params
                params = config.copy()
                params.pop('controllers')
                params.update(config['controllers'][controller])
                params['controller'] = controller
                params['chain'] = chain.to(device=params['device'])
                object_location = torch.tensor(env.obj_pose).to(params['device']).float() # NOTE: this is true for the tasks we have now. We need to pay attention if the root joint is not the root of the asset
                params['object_location'] = object_location
                ret = do_trial(env, params, fpath, sim_env, ros_copy_node)
                for key in ret.keys():
                    if key not in results[controller]:
                        results[controller][key] = []
                    results[controller][key].append(ret[key])
        print(results)

        if config['simulator'] == 'isaac_gym':
            gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
        elif config['simulator'] == 'isaac_sim':
            env.env.close()
    for key in results[method].keys():
        print(f"{method} {key}: avg: {np.array(results[method][key]).mean()}, std: {np.array(results[method][key]).std()}")
    valid_distance2goal = []
    for validity, distance2goal in zip(results[method]['validity_flag'], results[method]['final_distance_to_goal']):
        if validity:
            valid_distance2goal.append(distance2goal)
    if len(valid_distance2goal) == 0:
        print("No valid trials")
    else:
        print(f"{method} valid distance2goal: avg: {np.rad2deg(np.array(valid_distance2goal).mean())} degrees, std: {np.rad2deg(np.array(valid_distance2goal).std())} degrees")
    results['csvgd']['valid_distance2goal'] = valid_distance2goal
    save_dir = os.path.dirname(args.data_dir)
    print( os.path.join(save_dir, f'turning_{obj_id}.pt'))
    torch.save(results, os.path.join(save_dir, f'turning_{obj_id}.pt'))

