from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

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
from ccai.utils.allegro_utils import *
from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pytorch_mppi import MPPI


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')
nominal_screwdriver_top = np.array([0, 0, 1.405])

class AllegroScrewdriver(AllegroManipulationProblem):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 regrasp_fingers=[],
                 contact_fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                #  friction_coefficient=0.5,
                #  friction_coefficient=1000,
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 optimize_force=False,
                 turn=False,
                 obj_gravity=False,
                 min_force_dict=None,
                 device='cuda:0',
                 proj_path=None,
                 full_dof_goal=False, 
                 project=False,
                 test_recovery_trajectory=False, **kwargs):
        self.obj_mass = 0.0851
        self.obj_dof_type = None
        self.object_type = 'screwdriver'
        if obj_dof == 3:
            object_link_name = 'screwdriver_body'
            self.obj_translational_dim = 0
            self.obj_rotational_dim = 3
        elif obj_dof == 1:
            object_link_name = 'valve'
            self.obj_translational_dim = 0
            self.obj_rotational_dim = 1
        elif obj_dof == 6:
            object_link_name = 'card'
            self.obj_translational_dim = 2
            self.obj_rotational_dim = 1
        self.obj_link_name = object_link_name


        self.contact_points = None
        contact_points_object = None
        if proj_path is not None:
            self.proj_path = proj_path.to(device=device)
        else:
            self.proj_path = None

        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1,
                                                 optimize_force=optimize_force, device=device,
                                                 turn=turn, obj_gravity=obj_gravity,
                                                 min_force_dict=min_force_dict, 
                                                 full_dof_goal=full_dof_goal,
                                                  contact_points_object=contact_points_object,
                                                  contact_points_dict = self.contact_points,
                                                  project=project,
                                                   **kwargs)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=False):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 0
        if not self.project:
            upright_cost = 500 * torch.sum(
                (state[:, -self.obj_dof:-1] + goal[-self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=projected_diffusion)
    

class DynamicsModel:
    "This uses the simulation environment as the dynamcis model"
    def __init__(self, env, num_fingers, include_velocity=False, obj_joint_dim=0):
        self.env = env
        self.num_fingers = num_fingers
        self.include_velocity = include_velocity
        self.obj_joint_dim = obj_joint_dim
    def __call__(self, state, action):
        N = action.shape[0]
        # for the 1st env, action is to repeat the current state
        if self.include_velocity:
            tmp_obj_joint = torch.zeros((state.shape[0], self.obj_joint_dim, 2)).to(device=state.device)
            state = state.reshape((N, -1, 2))
            full_state = torch.cat((state, tmp_obj_joint), dim=-2)
            self.env.set_pose(full_state, semantic_order=False, zero_velocity=False)
            action = self.env.get_state()['q'][0, : 4 * self.num_fingers] + action
            self.env.step(action, ignore_img=True)
            ret = self.env.dof_states.clone().reshape(N, -1)
        else:
            tmp_obj_joint = torch.zeros((state.shape[0], self.obj_joint_dim)).to(device=state.device)
            full_state = torch.cat((state, tmp_obj_joint), dim=-1)
            self.env.set_pose(full_state, semantic_order=True, zero_velocity=True)
            action = state[:, :4 * self.num_fingers] + action
            ret = self.env.step(action, ignore_img=True)['q']
            if self.obj_joint_dim > 0:
                ret = ret[:, :-self.obj_joint_dim]
        return ret

class RunningCost:
    def __init__(self, start, goal, include_velocity=False):
        self.start = start
        self.goal = goal
        self.obj_dof = 3
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 3
        self.include_velocity = include_velocity
    
    def __call__(self, state, action):
        N = action.shape[0]
        if self.include_velocity:
            state = state.reshape(N, -1 ,2)
            state = state[:, :, 0] # no the cost is only on the position
            obj_orientation = state[:, -4:-1]
        else:
            obj_orientation = state[:, -4:-1]
                
        action_cost = torch.sum(action ** 2, dim=-1)

        goal_cost = 0
        # terminal cost
        # obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
        # obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
        # goal_orientation = tf.euler_angles_to_matrix(self.goal[-self.obj_rotational_dim:], convention='XYZ')
        # goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
        # terminal cost
        # goal_cost = goal_cost + torch.sum((20 * (obj_orientation - self.goal.unsqueeze(0)) ** 2), dim=-1)
        goal_cost = 20 * obj_orientation[..., -1]

        #upright cost
        upright_cost = 1000 * torch.sum(obj_orientation[:, :-1] ** 2, dim=-1)
        # dropping cost
        cost = action_cost + goal_cost + upright_cost
        cost = torch.nan_to_num(cost, nan=1e6)

        return cost

   
def do_trial(env, params, fpath):
    screwdriver_goal = params['screwdriver_goal'].cpu()
    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()
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

    start = state['q'].reshape(-1, 4 * num_fingers + 4).to(device=params['device'])

    pregrasp_params = copy.deepcopy(params)
    pregrasp_params['warmup_iters'] = 80
    pregrasp_problem = AllegroScrewdriver(
        start=start[0, :4 * num_fingers + obj_dof],
        goal=torch.zeros_like(start[0, -obj_dof:]),
        T=2,
        chain=pregrasp_params['chain'],
        device=pregrasp_params['device'],
        object_asset_pos=env.table_pose,
        object_location=pregrasp_params['object_location'],
        object_type=pregrasp_params['object_type'],
        world_trans=env.world_trans,
        regrasp_fingers=['index', 'middle', 'thumb'],
        contact_fingers=[],
        obj_dof=obj_dof,
        obj_joint_dim=1,
        optimize_force=pregrasp_params['optimize_force'],
        default_dof_pos=env.default_dof_pos[0, :16],
        obj_gravity=pregrasp_params.get('obj_gravity', False),
    )
    pregrasp_params = params.copy()
    pregrasp_params['N'] = 16
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)
    pregrasp_planner.warmup_iters = 80
    
    
    start = env.get_state()['q'].reshape(-1, 4 * num_fingers + 4).to(device=params['device'])[0, :15]
    best_traj, _ = pregrasp_planner.step(start)


    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        action = action.repeat(params['N'], 1)
        env.step(action)
        action_list.append(action)

    prime_dof_state = env.dof_states.clone()[0]
    prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
    env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
    state = env.get_state()
    start = state['q'][0].reshape(1, 4 * num_fingers + obj_dof + 1)[:, :4 * num_fingers + obj_dof].to(device=params['device'])
    

    actual_trajectory = []
    duration = 0

    num_fingers_to_plan = num_fingers
    info_list = []
    dynamics = DynamicsModel(env, num_fingers=len(params['fingers']), include_velocity=params['include_velocity'], obj_joint_dim=1)
    running_cost = RunningCost(start, params['screwdriver_goal'], include_velocity=params['include_velocity'])
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

    with torch.no_grad():
        for k in range(params['num_steps']):
            state = env.get_state()
            start = state['q'][0, :4 * num_fingers + obj_dof].to(device=params['device'])

            actual_trajectory.append(state['q'][0, :4 * num_fingers + obj_dof].clone())
            start_time = time.time()

            prime_dof_state = env.dof_states.clone()[0]
            prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
            finger_state = start[:4 * num_fingers].clone()

            orig_wrench_perturb = env.external_wrench_perturb
            env.set_external_wrench_perturb(False)
            if params['include_velocity']:
                action = ctrl.command(prime_dof_state[0].reshape(-1))
            else:
                action = ctrl.command(start[:4 * num_fingers + obj_dof].unsqueeze(0)) # this call will modify the environment
            env.set_external_wrench_perturb(orig_wrench_perturb)
            solve_time = time.time() - start_time
            duration += solve_time

            action = finger_state + action
            action = action.unsqueeze(0).repeat(params['N'], 1)
            # repeat the primary environment state to all the virtual environments        
            env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
            state = env.step(action)

            # screwdriver_top_pos = get_screwdriver_top_in_world(state['q'][0, -(obj_dof + 1): -1], object_chain, env.world_trans, env.table_pose)
            # screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
            # distance2nominal = np.linalg.norm(screwdriver_top_pos - nominal_screwdriver_top)
            # if distance2nominal > 0.02:
            #     validity_flag = False
            
            # if k < params['num_steps'] - 1:
            #     ctrl.change_horizon(ctrl.T - 1)

            print(f"solve time: {time.time() - start_time}")
            print(f"current theta: {state['q'][0, -(obj_dof+1):-1].detach().cpu().numpy()}")
            # add trajectory lines to sim

            
            action_list.append(action)
            screwdriver_state = env.get_state()['q'][0, -obj_dof-1: -1].cpu().unsqueeze(0)
            # distance2goal = euler_diff(screwdriver_state, screwdriver_goal.unsqueeze(0)).detach().cpu().abs()
            # print(distance2goal, validity_flag)

    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()
    state = state['q'][0, :4 * num_fingers + obj_dof].to(device=params['device'])
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    screwdriver_state = actual_trajectory[:, -obj_dof:].cpu()
    distance2goal = euler_diff(screwdriver_state, screwdriver_goal.unsqueeze(0).repeat(screwdriver_state.shape[0],1)).detach().cpu().abs()

    # final_distance_to_goal = torch.min(distance2goal.abs())
    final_distance_to_goal = distance2goal.abs()[-1].item()

    print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"])}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal)
    env.reset()
    ret = {
    'final_distance_to_goal': final_distance_to_goal, 
    'validity_flag': validity_flag,
    'avg_online_time': duration / (params["num_steps"])}
    return ret

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/baselines/config/allegro_screwdriver.yaml').read_text())
    from tqdm import tqdm

    env = AllegroScrewdriverTurningEnv(config['controllers']['mppi']['N'], control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=True,
                                        steps_per_action=60,
                                        friction_coefficient=2.5,
                                        # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                        device=config['sim_device'],
                                        video_save_path=img_save_dir,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        gradual_control=False,
                                        gravity=True, # For data generation only
                                        randomize_obj_start=config.get('randomize_obj_start', False),
                                        )
    sim, gym, viewer = env.get_sim()
    asset_object = get_assets_dir() + '/screwdriver/screwdriver.urdf'
    object_chain = pk.build_chain_from_urdf(open(asset_object).read()).to(device=config['sim_device'])



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
    config['obj_dof_code'] = [1, 1, 1, 0, 0, 0]
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
        results[controller]['dist2goal'] = []
        results[controller]['validity_flag'] = []
        results[controller]['avg_online_time'] = []

    for i in tqdm(range(config['num_trials'])):
        goal = torch.tensor([0, 0, -1.57])
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
            params['screwdriver_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            ret = do_trial(env, params, fpath)
            results[controller]['dist2goal'].append(ret['final_distance_to_goal'])
            results[controller]['validity_flag'].append(ret['validity_flag'])
            results[controller]['avg_online_time'].append(ret['avg_online_time'])
        print(results)

    for key in results[controller].keys():
        print(f"{controller} {key}: avg: {np.array(results[controller][key]).mean()}, std: {np.array(results[controller][key]).std()}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)