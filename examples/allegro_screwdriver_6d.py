from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverEnv
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroValveTurningEnv

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
# import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos, visualize_trajectory
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

device = 'cuda:0'
obj_dof = 6
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class AllegroScrewdriver(AllegroValveTurning):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 optimize_force=False,
                 obj_dof_code=[1, 1, 1, 1, 1, 1],
                 device='cuda:0', **kwargs):
        self.num_fingers = len(fingers)
        self.optimize_force = optimize_force
        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=obj_dof_code, 
                                                 obj_joint_dim=1, optimize_force=optimize_force, device=device)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, start, goal):
        # TODO: consider using quaternion difference for the orientation.
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        if self.optimize_force:
            action_cost = 0
        else:
            action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 1 * torch.sum((state[1:] - state[:-1]) ** 2)
        # smoothness_cost += 10 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        # smoothness_cost += 5000 * torch.sum((state[1:, -self.obj_dof:-self.obj_dof+3] - state[:-1, -self.obj_dof:-self.obj_dof+3]) ** 2) # the position should stay smooth
        # upright_cost = 1000 * torch.sum((state[:, -self.obj_dof:-self.obj_dof+3]) ** 2) # the screwdriver should only rotate in z direction

        # goal_cost = torch.sum((10 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((0.1 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))

        goal_cost = 0
        if self.obj_translational_dim:
            obj_position = state[:, -self.obj_dof:-self.obj_dof+self.obj_translational_dim]
            # terminal cost
            goal_cost = goal_cost + torch.sum((100 * (obj_position[-1] - goal[:self.obj_translational_dim]) ** 2))
            # running cost
            goal_cost = goal_cost + torch.sum((1 * (obj_position - goal[:self.obj_translational_dim]) ** 2))
            smoothness_cost = smoothness_cost + 100 * torch.sum((obj_position[1:] - obj_position[:-1]) ** 2)
        if self.obj_rotational_dim:
            obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
            obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
            obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
            goal_orientation = tf.euler_angles_to_matrix(goal[-self.obj_rotational_dim:], convention='XYZ')
            goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
            # terminal cost
            goal_cost = goal_cost + torch.sum((10 * (obj_orientation[-1] - goal_orientation) ** 2))
            # running cost 
            goal_cost = goal_cost + torch.sum((1 * (obj_orientation - goal_orientation) ** 2))
            smoothness_cost = smoothness_cost + 50 * torch.sum((obj_orientation[1:] - obj_orientation[:-1]) ** 2)
        # goal_cost = torch.sum((1000 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((10 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))


        if self.optimize_force:
            force = xu[:, self.dx + 4 * self.num_fingers: self.dx + (4 + 3) * self.num_fingers]
            force = force.reshape(force.shape[0], self.num_fingers, 3)
            force_norm = torch.norm(force, dim=-1)
            force_norm = force_norm - 0.3 # desired maginitute
            force_cost = 10 * torch.sum(force_norm ** 2)
            action_cost += force_cost
        return smoothness_cost + action_cost + goal_cost 
    
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the valve once"
    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    if params['controller'] == 'csvgd':
        # index finger is used for stability
        if 'index' in params['fingers']:
            contact_fingers = params['fingers']
        else:
            contact_fingers = ['index'] + params['fingers']        
        pregrasp_problem = AllegroContactProblem(
            dx=4 * num_fingers,
            du=4 * num_fingers,
            start=start[:4 * num_fingers + obj_dof],
            goal=None,
            T=4,
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            fingers=contact_fingers,
            obj_dof_code=params['obj_dof_code'],
            obj_joint_dim=1,
            fixed_obj=True
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
        pregrasp_planner.warmup_iters = 75
    else:
        raise ValueError('Invalid controller')
    
    
    start = env.get_state()['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])

    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        env.step(action)
        action_list.append(action)
        if params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, 4 * num_fingers)[0], params['fingers']))

    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    if params['exclude_index']:
            turn_problem_fingers = copy.copy(params['fingers'])
            turn_problem_fingers.remove('index')
            turn_problem_start = start[4:4 * num_fingers + obj_dof]
    else:
        turn_problem_fingers = params['fingers']
        turn_problem_start = start[:4 * num_fingers + obj_dof]
    turn_problem = AllegroScrewdriver(
        start=turn_problem_start,
        goal=params['valve_goal'],
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type=params['object_type'],
        friction_coefficient=params['friction_coefficient'],
        world_trans=env.world_trans,
        fingers=turn_problem_fingers,
        optimize_force=params['optimize_force']
    )
    turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)

    actual_trajectory = []
    duration = 0

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
        ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    if params['exclude_index']:
        num_fingers_to_plan = num_fingers - 1
    else:
        num_fingers_to_plan = num_fingers
    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())
        start_time = time.time()
        if params['exclude_index']:
            best_traj, trajectories = turn_planner.step(start[4:4 * num_fingers + obj_dof])
        else:
            best_traj, trajectories = turn_planner.step(start[:4 * num_fingers + obj_dof])

        print(f"solve time: {time.time() - start_time}")
        planned_theta_traj = best_traj[:, 4 * num_fingers_to_plan: 4 * num_fingers_to_plan + obj_dof].detach().cpu().numpy()
        print(f"current theta: {state['q'][0, -(obj_dof+1): -1].detach().cpu().numpy()}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        if params['mode'] == 'hardware':
            add_trajectories_hardware(trajectories, best_traj, axes, env, config=params, state2ee_pos_func=state2ee_pos)
        else:
            add_trajectories(trajectories, best_traj, axes, env, sim=sim, gym=gym, viewer=viewer,
                            config=params, state2ee_pos_func=state2ee_pos)

        if params['visualize_plan']:
            traj_for_viz = best_traj[:, :turn_problem.dx]
            if params['exclude_index']:
                traj_for_viz = torch.cat((start[4:4 + turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            else:
                traj_for_viz = torch.cat((start[:turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device) # add the joint for the screwdriver cap
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])
        
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, turn_problem.contact_scenes, viz_fpath, turn_problem.fingers, turn_problem.obj_dof+1)
        
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        if params['optimize_force']:
            print("planned force")
            print(action[:, 4 * num_fingers_to_plan:].reshape(num_fingers_to_plan, 3)) # print out the action for debugging
        # print(action)
        action = action[:, :4 * num_fingers_to_plan]
        if params['exclude_index']:
            action = start.unsqueeze(0)[:, 4:4 * num_fingers] + action
            action = torch.cat((start.unsqueeze(0)[:, :4], action), dim=1) # add the index finger back
        else:
            action = action + start.unsqueeze(0)[:, :4 * num_fingers] # NOTE: this is required since we define action as delta action
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        # action = x[:, :4 * num_fingers].to(device=env.device)
        # NOTE: DEBUG ONLY
        # action = best_traj[1, :4 * turn_problem.num_fingers].unsqueeze(0)
        # if params['exclude_index'] == True:
        #     action = torch.cat((start.unsqueeze(0)[:, :4], action), dim=1)
        #     action[:, 2] += 0.003
        #     action[:, 3] += 0.008
        env.step(action)
        action_list.append(action)
        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        # print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
        #                                                         compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - object_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - object_location[0].unsqueeze(0))**2)
        distance2goal = (params['valve_goal'].cpu() - env.get_state()['q'][:, -obj_dof-1: -1].cpu()).detach().cpu()
        print(distance2goal)
        # info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
        info = {'distance2goal': distance2goal} # DEBUG ONLY
        info_list.append(info)

        gym.clear_lines(viewer)
        state = env.get_state()
        start = state['q'][:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
        for finger in params['fingers']:
            ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
            finger_traj_history[finger].append(ee.detach().cpu().numpy())
        for finger in params['fingers']:
            traj_history = finger_traj_history[finger]
            temp_for_plot = np.stack(traj_history, axis=0)
            if k >= 2:
                axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    fig.tight_layout()
    fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    plt.savefig(f'{fpath.resolve()}/traj.png')
    plt.close()
    # plt.show()



    env.reset()
    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = (actual_trajectory[:, -obj_dof:] - params['valve_goal']).abs()

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_6d.yaml').read_text())
    from tqdm import tqdm

    if config['mode'] == 'hardware':
        env = RosAllegroValveTurningEnv(1, control_mode='joint_impedance',
                                 use_cartesian_controller=False,
                                 viewer=True,
                                 steps_per_action=60,
                                 friction_coefficient=1.0,
                                 device=config['sim_device'],
                                 valve=config['object_type'],
                                 video_save_path=img_save_dir,
                                 joint_stiffness=config['kp'],
                                 fingers=config['fingers'],
                                 )
    else:
        env = AllegroScrewdriverEnv(1, control_mode='joint_impedance',
                                    use_cartesian_controller=False,
                                    viewer=True,
                                    steps_per_action=60,
                                    friction_coefficient=2.0,
                                    device=config['sim_device'],
                                    video_save_path=img_save_dir,
                                    joint_stiffness=config['kp'],
                                    fingers=config['fingers'],
                                    )

    sim, gym, viewer = env.get_sim()


    state = env.get_state()
    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass

    sim_env = None
    ros_copy_node = None
    if config['mode'] == 'hardware':
        sim_env = env
        from hardware.hardware_env import HardwareEnv
        env = HardwareEnv(sim_env.default_dof_pos[:, :16], finger_list=['index', 'thumb'], kp=config['kp'])
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.valve_pose = sim_env.valve_pose
    elif config['mode'] == 'hardware_copy':
        from hardware.hardware_env import RosNode
        ros_copy_node = RosNode()



    results = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    config['ee_names'] = ee_names
    config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
    config['obj_dof'] = np.sum(config['obj_dof_code'])

    # screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver_6d.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    # screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])


    for i in tqdm(range(config['num_trials'])):
        # goal = torch.tensor([0, 0, 0, 0, 0, 0])
        goal = torch.tensor([0, 0, 0, 0, -0.75, 0])
        # goal = torch.tensor([0, 0, 0.04, 0, -1.57, 0]) # debug
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['valve_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.table_pose).to(device).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node)
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

