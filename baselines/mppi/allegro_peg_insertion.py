from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroPegInsertionEnv

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
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import *
from examples.allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pytorch_mppi import MPPI
from allegro_screwdriver import DynamicsModel

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
obj_dof = 6
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')


# class DynamicsModel:
#     "This uses the simulation environment as the dynamcis model"
#     def __init__(self, env, num_fingers, include_velocity=False):
#         self.env = env
#         self.num_fingers = num_fingers
#         self.include_velocity = include_velocity
#     def __call__(self, state, action):
#         N = action.shape[0]
#         # for the 1st env, action is to repeat the current state
#         if self.include_velocity:
#             state = state.reshape((N, -1, 2))
#             self.env.set_pose(state, semantic_order=False, zero_velocity=False)
#             action = env.get_state()['q'][0, : 4 * self.num_fingers] + action
#             self.env.step(action, ignore_img=True)
#             ret = env.dof_states.clone().reshape(N, -1)
#         else:
#             self.env.set_pose(state, semantic_order=True, zero_velocity=True)
#             action = state[:, :4 * self.num_fingers] + action
#             ret = self.env.step(action, ignore_img=True)['q']
#         return ret

class RunningCost:
    def __init__(self, start, goal, include_velocity=False):
        self.start = start
        self.goal = goal
        self.obj_dof = 6
        self.obj_translational_dim = 3
        self.obj_rotational_dim = 3
        self.include_velocity = include_velocity
    
    def __call__(self, state, action):
        N = action.shape[0]
        if self.include_velocity:
            state = state.reshape(N, -1 ,2)
            state = state[:, :, 0] # no the cost is only on the position
                
        action_cost = torch.sum(action ** 2, dim=-1)

        goal_cost = 0
        obj_position = state[:, -self.obj_dof:-self.obj_dof+self.obj_translational_dim]
        # terminal cost
        goal_cost = goal_cost + torch.sum((100 * (obj_position - self.goal[:self.obj_translational_dim].unsqueeze(0)) ** 2), dim=-1)

        obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
        obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
        obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
        goal_orientation = tf.euler_angles_to_matrix(self.goal[-self.obj_rotational_dim:], convention='XYZ')
        goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
        # terminal cost
        goal_cost = goal_cost + torch.sum((20 * (obj_orientation - goal_orientation.unsqueeze(0)) ** 2), dim=-1)
        # dropping cost
        dropping_cost = torch.sum((100 * ((obj_position[:, 2] < 0) * obj_position[:, 2])) ** 2, dim=-1)

        return action_cost + goal_cost + dropping_cost

   
def do_trial(env, params, fpath):
    peg_goal = params['object_goal'].cpu()
    peg_goal_pos = peg_goal[:3]
    peg_goal_mat = R.from_euler('xyz', peg_goal[-3:]).as_matrix()
    # step multiple times untile it's stable
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    for i in range(1):
        if len(params['fingers']) == 3:
            action = torch.cat((env.default_dof_pos[:,:8], env.default_dof_pos[:, 12:16]), dim=-1)
        elif len(params['fingers']) == 4:
            action = env.default_dof_pos[:, :16]
        state = env.step(action)

    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []

    start = state['q'][0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
  
    pregrasp_problem = AllegroContactProblem(
        dx=4 * num_fingers,
        du=4 * num_fingers,
        start=start[:4 * num_fingers + obj_dof],
        goal=None,
        T=4,
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.peg_pose,
        object_type='peg',
        world_trans=env.world_trans,
        fingers=params['fingers'],
        obj_dof_code=params['obj_dof_code'],
        obj_joint_dim=0,
        fixed_obj=True
    )
    pregrasp_params = params.copy()
    pregrasp_params['N'] = 10
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)
    pregrasp_planner.warmup_iters = 50
    
    
    start = env.get_state()['q'][0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])


    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        action = action.repeat(params['N'], 1)
        env.step(action)
        action_list.append(action)

    desired_table_pose = torch.tensor([0, 0, -1.0, 0, 0, 0, 1]).float().to(env.device)
    env.set_table_pose(env.handles['table'][0], desired_table_pose)

    prime_dof_state = env.dof_states.clone()[0]
    prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
    env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
    state = env.get_state()
    start = state['q'][0].reshape(1, 4 * num_fingers + obj_dof).to(device=params['device'])

    
    

    actual_trajectory = []
    duration = 0

    num_fingers_to_plan = num_fingers
    info_list = []
    dynamics = DynamicsModel(env, num_fingers=len(params['fingers']), include_velocity=params['include_velocity'])
    running_cost = RunningCost(start, params['object_goal'], include_velocity=params['include_velocity'])
    u_max = torch.ones(4 * len(params['fingers'])) * np.pi / 5 
    u_min = - torch.ones(4 * len(params['fingers'])) * np.pi / 5
    noise_sigma = torch.eye(4 * len(params['fingers'])).to(params['device']) * params['variance']
    if params['include_velocity']:
        nx = env.dof_states.shape[1] * 2
    else:
        nx = start.shape[1]
    ctrl = MPPI(dynamics=dynamics, running_cost=running_cost, nx=nx, noise_sigma=noise_sigma, 
                num_samples=params['N'], horizon=params['T'], lambda_=params['lambda'], u_min=u_min, u_max=u_max,
                device=params['device'])
    validity_flag = True
    contact_list = []
    with torch.no_grad():
        for k in range(params['num_steps']):
            state = env.get_state()
            start = state['q'][0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])

            actual_trajectory.append(state['q'][0, :4 * num_fingers + obj_dof].clone())
            start_time = time.time()

            prime_dof_state = env.dof_states.clone()[0]
            prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
            finger_state = start[:4 * num_fingers].clone()

            if params['include_velocity']:
                action = ctrl.command(prime_dof_state[0].reshape(-1))
            else:
                action = ctrl.command(start[:4 * num_fingers + obj_dof].unsqueeze(0)) # this call will modify the environment
            
            action = finger_state + action
            action = action.unsqueeze(0).repeat(params['N'], 1)
            # repeat the primary environment state to all the virtual environments        
            env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
            state = env.step(action)

            # if k < params['num_steps'] - 1:
            #     ctrl.change_horizon(ctrl.T - 1)
            solve_time = time.time() - start_time
            if k >= 0:
                duration += solve_time
            contacts = gym.get_env_rigid_contacts(env.envs[0])
            for body0, body1 in zip (contacts['body0'], contacts['body1']):
                if body0 == 31 and body1 == 33:
                    print("contact with wall")
                    contact_list.append(True)
                    break
                elif body0 == 33 and body1 == 31:
                    print("contact with wall")
                    contact_list.append(True)
                    break
            print(f"solve time: {solve_time}")
            # add trajectory lines to sim

            
            action_list.append(action)
            peg_state = state['q'][0, -obj_dof:].cpu()
            peg_mat = R.from_euler('xyz', peg_state[-3:]).as_matrix()
            distance2goal_ori = tf.so3_relative_angle(torch.tensor(peg_mat).unsqueeze(0), \
            torch.tensor(peg_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()
            distance2goal_pos = (peg_state[:3].unsqueeze(0) - peg_goal_pos.unsqueeze(0)).norm(dim=-1).detach().cpu()
            print(f"distance to goal pos: {distance2goal_pos}, ori: {distance2goal_ori}")
            print(distance2goal_pos, distance2goal_ori)
            if not check_peg_validity(peg_state):
                validity_flag = False

            info = {'distance2goal_pos': distance2goal_pos, 'distance2goal_ori': distance2goal_ori, 
            'validity_flag': validity_flag, 'contact': contact_list}
            info_list.append(info)

    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()
    state = state['q'][0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal_pos = distance2goal_pos
    final_distance_to_goal_ori = distance2goal_ori
    contact_rate = np.array(contact_list).mean()

    print(f'Controller: {params["controller"]} Final distance to goal pos: {final_distance_to_goal_pos.item()}, ori: {final_distance_to_goal_pos.item()}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"])}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            d2goal_pos=final_distance_to_goal_pos.item(),
            d2goal_ori=final_distance_to_goal_ori.item())
    env.reset()
    ret = {'final_distance_to_goal_pos': final_distance_to_goal_pos.item(), 
    'final_distance_to_goal_ori': final_distance_to_goal_ori.item(), 
    'contact_rate': contact_rate,
    'validity_flag': validity_flag,
    'avg_online_time': duration / (params["num_steps"])}
    return ret

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_peg_insertion.yaml').read_text())
    from tqdm import tqdm

    env = AllegroPegInsertionEnv(config['controllers']['mppi']['N'], control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                )

    sim, gym, viewer = env.get_sim()


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

    for controller in config['controllers'].keys():
        results[controller] = {}
        results[controller]['dist2goal_pos'] = []
        results[controller]['dist2goal_ori'] = []
        results[controller]['contact_rate'] = []
        results[controller]['validity_flag'] = []
        results[controller]['avg_online_time'] = []
        
    for i in tqdm(range(config['num_trials'])):
        goal = torch.tensor([0, 0, 0, 0, 0, 0])
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            config_file_path = pathlib.PurePath.joinpath(fpath, 'config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(config, f)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['object_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.peg_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            ret = do_trial(env, params, fpath)
            # final_distance_to_goal = turn(env, params, fpath)

            results[controller]['dist2goal_pos'].append(ret['final_distance_to_goal_pos'])
            results[controller]['dist2goal_ori'].append(ret['final_distance_to_goal_ori'])
            results[controller]['contact_rate'].append(ret['contact_rate'])
            results[controller]['validity_flag'].append(ret['validity_flag'])
            results[controller]['avg_online_time'].append(ret['avg_online_time'])

        print(results)
        for key in results[controller].keys():
            print(f"{controller} {key}: avg: {np.array(results[controller][key]).mean()}, std: {np.array(results[controller][key]).std()}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

