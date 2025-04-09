from isaac_victor_envs.utils import get_assets_dir
import isaac_victor_envs.tasks.allegro 

import numpy as np
import pickle as pkl

import torch
import time
import yaml
import pathlib

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

from ccai.utils.allegro_utils import *
from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC
from pytorch_mppi import MPPI
from dynamics_model import DynamicsModel


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos') # this does not really matter, as it will be overwritten
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


def do_trial(env, params, fpath, sim_viz_env=None):
    obj_dof = params['obj_dof']
    goal = params['goal'].cpu()
    arm_dof = get_arm_dof(params['arm_type'])
    num_fingers = len(params['fingers'])
    robot_dof = 4 * num_fingers + arm_dof
    if params['arm_type'] == 'robot':
        camera_params = "screwdriver_w_arm"
    elif params['arm_type'] == 'None' or params['arm_type'] == 'floating_3d' or params['arm_type'] == 'floating_6d':
        camera_params = "screwdriver"
    # step multiple times untile it's stable
    if params['visualize']:
        if params['mode'] == 'hardware':
            sim_viz_env.frame_fpath = fpath
            sim_viz_env.frame_id = 0
        else:
            env.frame_fpath = fpath
            env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    if params['mode'] == 'simulation':
        for i in range(1):
            if len(params['fingers']) == 3:
                action = torch.cat((env.default_dof_pos[:,:8], env.default_dof_pos[:, 12:16]), dim=-1)
            elif len(params['fingers']) == 4:
                action = env.default_dof_pos[:, :16]
            state = env.step(action)

    num_fingers = len(params['fingers'])
    if params['object_type'] == 'screwdriver':
        obj_joint_dim = 1 # compensate for the screwdriver cap
    else:
        obj_joint_dim = 0
    state = env.get_state()['q']
    action_list = []

    start = state[0, :4 * num_fingers + obj_dof].to(device=params['device'])

    # setup the pregrasp problem
    pregrasp_flag = False
    if config['task'] == 'peg_turning' or config['task'] == 'reorientation' or config['task'] == 'peg_alignment':
        pass
        # action = torch.cat((env.default_dof_pos[:,:8], env.default_dof_pos[:, 12:16]), dim=-1)
        # env.step(action) # step one step to resolve penetration
    else:
        pregrasp_succ = False
        while pregrasp_succ == False:
            pregrasp_dx = pregrasp_du = robot_dof
            pregrasp_problem = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['goal'],
                T=2,
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=env.obj_pose,
                object_type=params['object_type'],
                world_trans=env.world_trans,
                regrasp_fingers=['index', 'middle', 'thumb'],
                contact_fingers=[],
                obj_dof=obj_dof,
                obj_joint_dim=1,
                optimize_force=True,
                default_dof_pos=env.default_dof_pos[0, :16],
                obj_gravity=True,
                full_dof_goal=False,
                # proj_path=proj_path,
            )
            pregrasp_params = params.copy()
            pregrasp_params['N'] = 16
            pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)
            pregrasp_planner.warmup_iters = 50 
            # else:
            #     raise ValueError('Invalid controller')
            
            start_time = time.time()
            best_traj, _ = pregrasp_planner.step(start)
            print(f"pregrasp solve time: {time.time() - start_time}")

            for x in best_traj[:, :pregrasp_dx]:
                action = x.reshape(-1, pregrasp_dx).to(device=env.device) # move the rest fingers
                if params['mode'] == 'hardware':
                    set_state = env.get_state(return_dict=True)['q'].to(device=env.device)
                    if params['task'] == 'screwdriver_turning':
                        set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
                    sim_viz_env.set_pose(set_state)
                    sim_viz_env.step(action)
                env.step(action)
                action_list.append(action)
                if params['mode'] == 'hardware_copy':
                    ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, pregrasp_dx)[0], params['fingers']))
            pregrasp_succ = True
            # pregrasp_succ = env.check_validity(env.get_state()['q'].cpu()[0])
            # if pregrasp_succ == False:
            #     print("pregrasp failed, replanning")
            #     env.reset()
    # prime_dof_state = env.dof_states.clone()[0]
    # prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
    # env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
    # state = env.get_state()
    # start = state[0].reshape(1, 4 * num_fingers + obj_dof).to(device=params['device'])
    

    actual_trajectory = []
    duration = 0
    warmup_time = 0

    if params['mode'] == 'hardware':
        dynamics = DynamicsModel(sim_viz_env, num_fingers=len(params['fingers']), include_velocity=params['include_velocity'], obj_joint_dim=obj_joint_dim, hardware=True)
    elif params['mode'] == 'simulation':
        dynamics = DynamicsModel(env, num_fingers=len(params['fingers']), include_velocity=params['include_velocity'], obj_joint_dim=obj_joint_dim, hardware=False)
    if config['task'] == 'screwdriver_turning':
        from baselines.allegro_screwdriver import RunningCost
    elif config['task'] == 'peg_alignment':
        from baselines.allegro_peg_alignment import RunningCost
    elif config['task'] == 'valve_turning':
        from baselines.allegro_valve_turning import RunningCost   
    elif config['task'] == 'peg_turning':
        from baselines.allegro_peg_turning import RunningCost  
    elif config['task'] == 'reorientation':
        from baselines.allegro_reorientation import RunningCost 

    running_cost = RunningCost(params['goal'], include_velocity=params['include_velocity'])
    u_max = torch.ones(4 * len(params['fingers'])) * np.pi / 5 
    u_min = - torch.ones(4 * len(params['fingers'])) * np.pi / 5
    noise_sigma = torch.eye(4 * len(params['fingers'])).to(params['device']) * params['variance']
    if params['include_velocity']:
        if params['mode'] == 'hardware':
            nx = sim_viz_env.dof_states.shape[1] * 2
        else:
            nx = env.dof_states.shape[1] * 2
    else:
        nx = 4 * num_fingers + obj_dof
    ctrl = MPPI(dynamics=dynamics, running_cost=running_cost, nx=nx, noise_sigma=noise_sigma, 
                num_samples=params['N'], horizon=params['T'], lambda_=params['lambda'], u_min=u_min, u_max=u_max,
                device=params['device'])
    
    validity_flag = True
    if params['task'] == 'peg_alignment':
        contact_list = []

    with torch.no_grad():
        for k in range(params['num_steps']):
            state = env.get_state()['q']
            start = state[0, :4 * num_fingers + obj_dof].to(device=params['device'])

            actual_trajectory.append(state[0, :4 * num_fingers + obj_dof].clone())
            start_time = time.time()

            if params['mode'] == 'hardware':
                robot_state = env.get_processed_robot_state().to(device=params['device'])
                prime_dof_state = torch.cat((robot_state, start[4 * num_fingers:].unsqueeze(0)), dim=-1).repeat(params['N'], 1)
                # prime_dof_state = torch.stack((prime_dof_state, torch.zeros_like(prime_dof_state)), dim=-1)
            else:
                prime_dof_state = env.dof_states.clone()[0].to(device=params['device'])
                prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
            finger_state = start[:4 * num_fingers].clone()

            num_warmup_iters = 4 if k == 0 else 0
            for _ in range(num_warmup_iters):
                if params['include_velocity']:
                    action = ctrl.command(prime_dof_state[0].reshape(-1), shift_nominal_trajectory=False)
                else:
                    action = ctrl.command(start[:4 * num_fingers + obj_dof].unsqueeze(0), shift_nominal_trajectory=False) # this call will modify the environment
            if params['include_velocity']:
                action = ctrl.command(prime_dof_state[0].reshape(-1), shift_nominal_trajectory=True)
            else:
                action = ctrl.command(start[:4 * num_fingers + obj_dof].unsqueeze(0), shift_nominal_trajectory=True) # this call will modify the environment
            solve_time = time.time() - start_time
            if k == 0:
                warmup_time = solve_time
            else:
                duration += solve_time

            action = finger_state + action
            action = action.unsqueeze(0)
            # repeat the primary environment state to all the virtual environments    
            if params['mode'] == 'simulation':    
                env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
                action = action.repeat(params['N'], 1)
            action = action.to(env.device)
            state = env.step(action)
            if params['mode'] == 'hardware':
                sim_viz_env.step(action)
            
            # if params['mode'] == 'simulation':
                # if not env.check_validity(state):
                #     validity_flag = False
            # if k < params['num_steps'] - 1:
            #     ctrl.change_horizon(ctrl.T - 1)

            print(f"solve time: {time.time() - start_time}")
            print(f"current theta: {state['q'][0, -obj_dof:].detach().cpu().numpy()}")
            # add trajectory lines to sim

            
            action_list.append(action)
            obj_state = env.get_state()['q'][0, -obj_dof:].cpu().unsqueeze(0)

            if params['task'] == 'screwdriver_turning':
                distance2goal = euler_diff(obj_state, goal.unsqueeze(0)).detach().cpu().abs().item()
                print(distance2goal)
            elif params['task'] == 'peg_alignment' or params['task'] == 'peg_turning' or params['task'] == 'reorientation':
                peg_goal_pos = goal[:3]
                peg_goal_mat = R.from_euler('XYZ', goal[-3:]).as_matrix()
                peg_mat = R.from_euler('XYZ', obj_state[0, -3:]).as_matrix()
                distance2goal_ori = tf.so3_relative_angle(torch.tensor(peg_mat).unsqueeze(0), \
                torch.tensor(peg_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs().item()
                distance2goal_pos = (obj_state[0, :3] - peg_goal_pos).norm(dim=-1).detach().cpu().item()
                print(f"distance to goal pos: {distance2goal_pos}, ori: {distance2goal_ori}")
                if params['mode'] == 'simulation':
                    if params['task'] == 'peg_alignment':
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
            elif params['task'] == 'valve_turning':
                distance2goal = (obj_state[0] - goal).detach().item()
                print(distance2goal)

            print(validity_flag)

    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()['q']
    state = state[0, :4 * num_fingers + obj_dof]
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    obj_state = actual_trajectory[:, -obj_dof:].cpu()
    ret = {}
    if params['task'] == 'screwdriver_turning':
        distance2goal = euler_diff(obj_state, goal.unsqueeze(0).repeat(obj_state.shape[0],1)).detach().cpu().abs()
        # final_distance_to_goal = torch.min(distance2goal.abs())
        final_distance_to_goal = distance2goal.abs()[-1].item()
        ret['final_distance_to_goal'] = final_distance_to_goal
        print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}')
    elif params['task'] == 'peg_alignment' or params['task'] == 'peg_turning' or params['task'] == 'reorientation':
        final_distance_to_goal_pos = distance2goal_pos
        final_distance_to_goal_ori = distance2goal_ori
        ret['final_distance_to_goal_pos'] = final_distance_to_goal_pos
        ret['final_distance_to_goal_ori'] = final_distance_to_goal_ori
        if params['task'] == 'peg_alignment':
            contact_rate = np.array(contact_list).mean()
            ret['contact_rate'] = contact_rate
    elif params['task'] == 'valve_turning':
        distance2goal = (obj_state - goal).detach().cpu().abs()
        final_distance_to_goal = distance2goal.abs()[-1].item()
        ret['final_distance_to_goal'] = final_distance_to_goal
        print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"])}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy())
    env.reset()
    ret['validity_flag'] = validity_flag
    ret['avg_online_time'] = duration / (params["num_steps"] - 1)
    ret['warmup_time'] = warmup_time
    return ret

if __name__ == "__main__":
    # get config
    from tqdm import tqdm
    task = 'screwdriver_turning'
    # task = 'valve_turning'
    # task = 'peg_alignment'
    # task = 'peg_turning'
    # task =  'reorientation'
    if task == 'screwdriver_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/baselines/config/allegro_screwdriver.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]        
    elif task == 'valve_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_valve.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 0, 1, 0]
    elif task == 'peg_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_peg_turning.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
    elif task == 'peg_alignment':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_peg_alignment.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
    elif task == 'reorientation':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_reorientation.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]

    obj_dof = sum(config['obj_dof_code'])
    config['obj_dof'] = obj_dof
    config['task'] = task

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
        from hardware.hardware_env import HardwareEnv
        # TODO, think about how to read that in simulator
        if task == 'screwdriver_turning':
            default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                        torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                        torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                        torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                        dim=1)
            obj = 'screwdriver'
        elif task == 'peg_alignment':
            default_dof_pos = torch.cat((torch.tensor([[0, 0.7, 0.8, 0.8]]).float(),
                                    torch.tensor([[0, 0.8, 0.7, 0.6]]).float(),
                                    torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.05, 1.1]]).float()),
                                    dim=1)
            obj = 'peg'
        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj=obj,
                          mode='relative',
                          gradual_control=config['gradual_control'],
                          num_repeat=10)
        if task == 'screwdriver_turning':
            from isaac_victor_envs.utils import get_assets_dir
            from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
            root_coor, root_ori = env.obj_reader.get_state()
            root_coor = root_coor / 1000 # convert to meters
            # robot_p = np.array([-0.025, -0.1, 1.33])
            robot_p = np.array([0, -0.095, 1.33])
            root_coor = root_coor + robot_p
            sim_env = AllegroScrewdriverTurningEnv(num_envs=config['controllers']['mppi']['N'], 
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
                                            obj_pose=root_coor,
                                            )
        elif task == 'peg_alignment':
            from isaac_victor_envs.utils import get_assets_dir
            from utils.isaacgym_utils import get_env
            sim_env = get_env(task, img_save_dir, config, num_envs=config['controllers']['mppi']['N'])
        sim, gym, viewer = sim_env.get_sim()
        if task == 'screwdriver_turning':
            assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.obj_pose = sim_env.obj_pose
        if task == 'peg_alignment':
            env.wall_pose = sim_env.wall_pose
            env.wall_dims = sim_env.wall_dims
    else:
        from isaac_victor_envs.utils import get_assets_dir
        from ccai.utils.isaacgym_utils import get_env
        env = get_env(task, img_save_dir, config, num_envs=config['controllers']['mppi']['N'])
        sim, gym, viewer = env.get_sim()


    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read()).to(device=config['device'])


    results = {}

    for i in tqdm(range(config['num_trials'])):
        config['goal'] = torch.tensor(config['goal']).to(device=config['device']).float()
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
            params['goal'] = params['goal'].to(device=params['device'])
            params['chain'] = chain
            object_location = torch.tensor(env.obj_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            ret = do_trial(env, params, fpath, sim_env)
            for key in ret.keys():
                if key not in results.keys():
                    results[key] = []
                results[key].append(ret[key])
        print(results)

    for key in results.keys():
        print(f"{key}: avg: {np.array(results[key]).mean()}, std: {np.array(results[key]).std()}")
    valid_distance2goal = []
    if 'final_distance_to_goal_ori' in results.keys():
        for validity, distance2goal_ori in zip(results['validity_flag'], results['final_distance_to_goal_ori']):
            if validity:
                valid_distance2goal.append(distance2goal_ori)
    elif 'final_distance_to_goal' in results.keys():
        for validity, distance2goal in zip(results['validity_flag'], results['final_distance_to_goal']):
            if validity:
                valid_distance2goal.append(distance2goal)
    if len(valid_distance2goal) == 0:
        print("No valid trials")
    else:
        print(f"valid distance2goal: avg: {np.rad2deg(np.array(valid_distance2goal).mean())} degrees, std: {np.rad2deg(np.array(valid_distance2goal).std())} degrees")
    print(task)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

