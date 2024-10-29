
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv
# from isaacsim_hand_envs.allegro import AllegroScrewdriverEnv # it needs to be imported before numpy and torch
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
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC
from allegro_screwdriver_w_force import AllegroScrewdriver
from allegro_peg_turning import AllegroPegTurning
from allegro_peg_alignment_w_force import AllegroPegAlignment
from scipy.spatial.transform import Rotation as R

from utils.allegro_utils import *
from tqdm import tqdm


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
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    state = env.get_state()
    action_list = []
    start = state.reshape(robot_dof + obj_dof).to(device=params['device'])        
    
    # setup the pregrasp problem
    if config['task'] == 'peg_turning':
        pass
        # action = torch.cat((env.default_dof_pos[:,:8], env.default_dof_pos[:, 12:16]), dim=-1)
        # env.step(action) # step one step to resolve penetration
    
    else:
        if params['controller'] == 'csvgd':
            pregrasp_dx = pregrasp_du = robot_dof
            pregrasp_problem = AllegroContactProblem(
                dx=pregrasp_dx,
                du=pregrasp_du,
                start=start[:pregrasp_dx + obj_dof],
                goal=None,
                T=4,
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.obj_pose,
                object_type=params['object_type'],
                world_trans=env.world_trans,
                fingers=params['fingers'],
                obj_dof_code=params['obj_dof_code'],
                obj_joint_dim=obj_joint_dim,
                fixed_obj=True,
                arm_type=params['arm_type'],
            )

            pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
            pregrasp_planner.warmup_iters = 50 
        else:
            raise ValueError('Invalid controller')
        
        start_time = time.time()
        best_traj, _ = pregrasp_planner.step(start[:pregrasp_dx])
        print(f"pregrasp solve time: {time.time() - start_time}")

        if params['visualize_plan']:
            traj_for_viz = best_traj[:, :pregrasp_problem.dx]
            tmp = start[pregrasp_dx:pregrasp_dx+obj_dof].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
            tmp_2 = torch.zeros((traj_for_viz.shape[0], 1)).to(traj_for_viz.device) # the top jint
            traj_for_viz = torch.cat((traj_for_viz, tmp, tmp_2), dim=1)    
            viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, pregrasp_problem.fingers, pregrasp_problem.obj_dof + obj_joint_dim,
                                camera_params=camera_params, arm_dof=arm_dof)


        for x in best_traj[:, :pregrasp_dx]:
            action = x.reshape(-1, pregrasp_dx).to(device=env.device) # move the rest fingers
            if params['mode'] == 'hardware':
                set_state = env.get_state()['all_state'].to(device=env.device)
                set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
                sim_viz_env.set_pose(set_state)
                sim_viz_env.step(action)
            env.step(action)
            action_list.append(action)
            if params['mode'] == 'hardware_copy':
                ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, pregrasp_dx)[0], params['fingers']))
    if params['task'] == 'peg_alignment':
        desired_table_pose = torch.tensor([0, 0, -1.0, 0, 0, 0, 1]).float().to(env.device)
        env.set_table_pose(env.handles['table'][0], desired_table_pose)
        state = env.get_state()
        state = env.step(state[:, :robot_dof])

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
                arm_type=params['arm_type'],
            ) # TODO: add other tasks
        elif config['task'] == 'valve_turning':
            manipulation_problem = AllegroValveTurning(
                start=start,
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.obj_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                world_trans=env.world_trans,
                fingers=params['fingers'],
                arm_type=params['arm_type'],
                optimize_force=params['optimize_force'],
                obj_dof_code=params['obj_dof_code'],
            )
        elif config['task'] == 'peg_turning':
            manipulation_problem = AllegroPegTurning(
                start=start,
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                object_asset_pos=env.obj_pose,
                world_trans=env.world_trans,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                optimize_force=params['optimize_force'],
                device=params['device'],
                fingers=params['fingers'],
                obj_dof_code=params['obj_dof_code'],
                obj_gravity=params['obj_gravity'],
            )
        elif config['task'] == 'peg_alignment':
            manipulation_problem = AllegroPegAlignment(
                            start=start,
                            goal=params['goal'],
                            T=params['T'],
                            chain=params['chain'],
                            device=params['device'],
                            peg_asset_pos=env.obj_pose,
                            wall_asset_pos=env.wall_pose,
                            wall_dims = env.wall_dims,
                            object_location=params['object_location'],
                            object_type=params['object_type'],
                            friction_coefficient=params['friction_coefficient'],
                            world_trans=env.world_trans,
                            fingers=params['fingers'],
                            optimize_force=params['optimize_force'],
                            obj_gravity = params['obj_gravity'],
                            arm_type = params['arm_type'],
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
        best_traj, trajectories = manipulation_planner.step(start[:robot_dof + obj_dof])
        
        solve_time = time.time() - start_time
        print(f"solve time: {solve_time}")
        if k == 0:
            warmup_time = solve_time
        else:
            duration += solve_time
        planned_theta_traj = best_traj[:, robot_dof: robot_dof + obj_dof].detach().cpu().numpy()
        print(f"current theta: {current_theta}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        # if k < params['num_steps'] - 1:
        #     if params['mode'] == 'hardware':
        #         pass # debug TODO: fix it
        #         # add_trajectories_hardware(trajectories, best_traj, axes, env, config=params, state2ee_pos_func=state2ee_pos)
        #     else:
        #         add_trajectories(trajectories, best_traj, axes, env, sim=sim, gym=gym, viewer=viewer,
        #                         config=params, state2ee_pos_func=state2ee_pos)

        if params['visualize_plan']:
            # manipulation_problem.viz_contact_scenes.visualize_robot(partial_to_full_state(start[:manipulation_problem.robot_dof], fingers=params['fingers'], arm_dof=arm_dof), start[robot_dof:robot_dof + obj_dof])
            traj_for_viz = best_traj[:, :manipulation_problem.dx]
            traj_for_viz = torch.cat((start[:manipulation_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device) # add the joint for the screwdriver cap
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
        if params['optimize_force']:
            print("planned force")
            print(action[:, robot_dof:].reshape(num_fingers + params['num_env_force'], 3)) # print out the action for debugging
            print("delta action")
            if params['arm_type'] != 'None':
                print(action[:, :arm_dof])
                print(action[:, arm_dof:manipulation_problem.robot_dof].reshape(num_fingers, 4))
            else:
                print(action[:, :robot_dof].reshape(num_fingers, 4))
        action = action[:, :robot_dof]
        action = action + start.unsqueeze(0)[:, :robot_dof].to(action.device) # NOTE: this is required since we define action as delta action
        if params['mode'] == 'hardware':
            set_state = env.get_state()['all_state'].to(device=env.device)
            set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
            sim_viz_env.set_pose(set_state)
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        env.step(action)
        action_list.append(action)
        manipulation_problem._preprocess(best_traj.unsqueeze(0))
        
        obj_state = env.get_state()[:, -obj_dof:].cpu()
        if params['task'] == 'screwdriver_turning':
            screwdriver_mat = R.from_euler('xyz', obj_state).as_matrix() # TODO: fix it for different tasks
            screwdriver_goal_mat = R.from_euler('xyz', goal).as_matrix() # TODO: change it for different tasks
            distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
                torch.tensor(screwdriver_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()
        elif params['task'] == 'valve_turning':
            distance2goal = (obj_state[0] - goal).detach().item()
        elif params['task'] == 'peg_turning' or params['task'] == 'peg_alignment':
            peg_mat = R.from_euler('xyz', obj_state[:, -3:]).as_matrix() # TODO: fix it for different tasks
            peg_goal_mat = R.from_euler('xyz', goal[-3:]).as_matrix() # TODO: change it for different tasks
            distance2goal = tf.so3_relative_angle(torch.tensor(peg_mat), \
                torch.tensor(peg_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()
        
        if not manipulation_problem.check_validity(env.get_state().cpu()[0]):
            validity_flag = False

        print(distance2goal, validity_flag)
        info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal, 'validity_flag': validity_flag}}
        info_list.append(info)

        # if params['simulator'] == 'isaac_gym':
        #     gym.clear_lines(viewer)
        state = env.get_state()
        start = state.squeeze(0).to(device=params['device'])
    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()
    state = state.reshape(robot_dof + obj_dof).to(device=params['device'])
    actual_trajectory.append(state.clone()[:robot_dof + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, robot_dof + obj_dof)
    manipulation_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    obj_state = actual_trajectory[:, -obj_dof:].cpu()
    if params['task'] == 'screwdriver_turning':
        screwdriver_mat = R.from_euler('xyz', obj_state).as_matrix() # TODO: fix it for different tasks
        screwdriver_goal_mat = R.from_euler('xyz', goal).as_matrix() # TODO: change it for different tasks
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
            torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0],1,1), cos_angle=False).detach().cpu()
    elif params['task'] == 'valve_turning':
        distance2goal = (obj_state - goal).detach().cpu()
    elif params['task'] == 'peg_turning' or params['task'] == 'peg_alignment':
        peg_mat = R.from_euler('xyz', obj_state[:, -3:]).as_matrix()
        peg_goal_mat = R.from_euler('xyz', goal[-3:]).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(peg_mat), \
            torch.tensor(peg_goal_mat).unsqueeze(0).repeat(peg_mat.shape[0],1,1), cos_angle=False).detach().cpu()

    # final_distance_to_goal = torch.min(distance2goal.abs())
    final_distance_to_goal = distance2goal.abs()[-1].cpu().detach().item()

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
    # get config
    # task = 'screwdriver_turning'
    # task = 'valve_turning'
    # task = 'peg_turning'
    task = 'peg_alignment'
    if task == 'screwdriver_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]        
        config['num_env_force'] = 1
    elif task == 'valve_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_valve.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 0, 1, 0]
        config['num_env_force'] = 0
    elif task == 'peg_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_peg_turning.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 0
    elif task == 'peg_alignment':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_peg_alignment.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 1
    obj_dof = sum(config['obj_dof_code'])
    config['obj_dof'] = obj_dof
    config['task'] = task

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
        from hardware.hardware_env import HardwareEnv
        # TODO, think about how to read that in simulator
        default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                    torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                    torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                    dim=1)
        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj='screwdriver',
                          mode='relative',
                          gradual_control=config['gradual_control'],
                          num_repeat=10)
        root_coor, root_ori = env.obj_reader.get_state()
        root_coor = root_coor / 1000 # convert to meters
        # robot_p = np.array([-0.025, -0.1, 1.33])
        robot_p = np.array([0, -0.095, 1.33])
        root_coor = root_coor + robot_p
        sim_env = RosAllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                 use_cartesian_controller=False,
                                 viewer=True,
                                 steps_per_action=60,
                                 friction_coefficient=1.0,
                                 device=config['sim_device'],
                                 valve=config['object_type'],
                                 video_save_path=img_save_dir,
                                 joint_stiffness=config['kp'],
                                 fingers=config['fingers'],
                                 obj_pose=root_coor,
                                 )
        sim, gym, viewer = sim_env.get_sim()
        assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.obj_pose = sim_env.obj_pose
    else:
        if config['simulator'] == 'isaac_gym':
            from isaac_victor_envs.utils import get_assets_dir
            from utils.isaacgym_utils import get_env
            env = get_env(task, img_save_dir, config)
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

    chain = pk.build_chain_from_urdf(open(asset).read())

    # setup the helper function
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans, arm_dof=arm_dof)
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])

    for controller in config['controllers'].keys():
        results[controller] = {}

    for i in tqdm(range(config['num_trials'])):
        config['goal'] = torch.tensor(config['goal']).to(device=config['device']).float()
        for controller in config['controllers'].keys():
            if controller == config['method']: # only evaluate the method specified in the config
                env.reset()
                fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
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
                    else:
                        results[controller][key].append(ret[key])
        print(results)

    for key in results[controller].keys():
        print(f"{controller} {key}: avg: {np.array(results[controller][key]).mean()}, std: {np.array(results[controller][key]).std()}")
    if config['simulator'] == 'isaac_gym':
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
    elif config['simulator'] == 'isaac_sim':
        env.env.close()

