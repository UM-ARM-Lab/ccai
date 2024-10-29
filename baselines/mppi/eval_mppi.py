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

from utils.allegro_utils import *
from examples.allegro_valve_roll import AllegroContactProblem, PositionControlConstrainedSVGDMPC
from pytorch_mppi import MPPI
from dynamics_model import DynamicsModel


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos') # this does not really matter, as it will be overwritten


def do_trial(env, params, fpath):
    obj_dof = params['obj_dof']
    goal = params['goal'].cpu()
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
    if params['object_type'] == 'screwdriver':
        obj_joint_dim = 1 # compensate for the screwdriver cap
    else:
        obj_joint_dim = 0
    state = env.get_state()
    action_list = []

    start = state[0, :4 * num_fingers + obj_dof].to(device=params['device'])
  
    pregrasp_problem = AllegroContactProblem(
        dx=4 * num_fingers,
        du=4 * num_fingers,
        start=start[:4 * num_fingers + obj_dof],
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
        fixed_obj=True
    )
    pregrasp_params = params.copy()
    pregrasp_params['N'] = 10
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)
    pregrasp_planner.warmup_iters = 50
    
    
    start = env.get_state()[0, :4 * num_fingers + obj_dof].to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])


    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        action = action.repeat(params['N'], 1)
        env.step(action)
        action_list.append(action)
    if params['task'] == 'peg_alignment':
        desired_table_pose = torch.tensor([0, 0, -1.0, 0, 0, 0, 1]).float().to(env.device)
        env.set_table_pose(env.handles['table'][0], desired_table_pose)
        state = env.get_state()
        state = env.step(state[:, :4 * num_fingers])

    prime_dof_state = env.dof_states.clone()[0]
    prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
    env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
    state = env.get_state()
    start = state[0].reshape(1, 4 * num_fingers + obj_dof).to(device=params['device'])
    

    actual_trajectory = []
    duration = 0

    info_list = []
    dynamics = DynamicsModel(env, num_fingers=len(params['fingers']), include_velocity=params['include_velocity'], obj_joint_dim=obj_joint_dim)
    if config['task'] == 'screwdriver_turning':
        from baselines.mppi.allegro_screwdriver import RunningCost
        from baselines.mppi.allegro_screwdriver import ValidityCheck
        validity_checker = ValidityCheck(params['object_chain'], obj_dof, env.world_trans, env.obj_pose)
    elif config['task'] == 'peg_alignment':
        from baselines.mppi.allegro_peg_alignment import RunningCost
        from baselines.mppi.allegro_peg_alignment import ValidityCheck
        validity_checker = ValidityCheck(obj_dof=obj_dof)
    running_cost = RunningCost(start, params['goal'], include_velocity=params['include_velocity'])
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
    if params['task'] == 'peg_alignment':
        contact_list = []

    with torch.no_grad():
        for k in range(params['num_steps']):
            state = env.get_state()
            start = state[0, :4 * num_fingers + obj_dof].to(device=params['device'])

            actual_trajectory.append(state[0, :4 * num_fingers + obj_dof].clone())
            start_time = time.time()

            prime_dof_state = env.dof_states.clone()[0]
            prime_dof_state = prime_dof_state.unsqueeze(0).repeat(params['N'], 1, 1)
            finger_state = start[:4 * num_fingers].clone()

            if params['include_velocity']:
                action = ctrl.command(prime_dof_state[0].reshape(-1))
            else:
                action = ctrl.command(start[:4 * num_fingers + obj_dof].unsqueeze(0)) # this call will modify the environment
            solve_time = time.time() - start_time
            duration += solve_time

            action = finger_state + action
            action = action.unsqueeze(0).repeat(params['N'], 1)
            # repeat the primary environment state to all the virtual environments        
            env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False)
            state = env.step(action)
            
            if not validity_checker.check_validity(state):
                validity_flag = False
            # if k < params['num_steps'] - 1:
            #     ctrl.change_horizon(ctrl.T - 1)

            print(f"solve time: {time.time() - start_time}")
            print(f"current theta: {state[0, -obj_dof:].detach().cpu().numpy()}")
            # add trajectory lines to sim

            
            action_list.append(action)
            obj_state = env.get_state()[0, -obj_dof:].cpu().unsqueeze(0)

            if params['task'] == 'screwdriver_turning':
                distance2goal = euler_diff(obj_state, goal.unsqueeze(0)).detach().cpu().abs()
                print(distance2goal)
            elif params['task'] == 'peg_alignment':
                peg_goal_pos = goal[:3]
                peg_goal_mat = R.from_euler('xyz', goal[-3:]).as_matrix()
                peg_mat = R.from_euler('xyz', obj_state[0, -3:]).as_matrix()
                distance2goal_ori = tf.so3_relative_angle(torch.tensor(peg_mat).unsqueeze(0), \
                torch.tensor(peg_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()
                distance2goal_pos = (obj_state[0, :3] - peg_goal_pos).norm(dim=-1).detach().cpu()
                print(f"distance to goal pos: {distance2goal_pos}, ori: {distance2goal_ori}")

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

            print(validity_flag)

    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()
    state = state[0, :4 * num_fingers + obj_dof].to(device=params['device'])
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
    elif params['task'] == 'peg_alignment':
        final_distance_to_goal_pos = distance2goal_pos
        final_distance_to_goal_ori = distance2goal_ori
        ret['final_distance_to_goal_pos'] = final_distance_to_goal_pos
        ret['final_distance_to_goal_ori'] = final_distance_to_goal_ori
        contact_rate = np.array(contact_list).mean()
        ret['contact_rate'] = contact_rate
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"])}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy())
    env.reset()
    ret['validity_flag'] = validity_flag
    ret['avg_online_time'] = duration / (params["num_steps"])
    return ret

if __name__ == "__main__":
    # get config
    from tqdm import tqdm
    # task = 'screwdriver_turning'
    task = 'peg_alignment'
    if task == 'screwdriver_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_screwdriver.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]        
        config['num_env_force'] = 1
    elif task == 'valve_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_valve.yaml').read_text())
        config['obj_dof_code'] = [0, 0, 0, 0, 1, 0]
        config['num_env_force'] = 0
    elif task == 'peg_turning':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_peg_turning.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 0
    elif task == 'peg_alignment':
        config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/mppi/config/allegro_peg_alignment.yaml').read_text())
        config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
        config['num_env_force'] = 0
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
        from isaac_victor_envs.utils import get_assets_dir
        from utils.isaacgym_utils import get_env
        env = get_env(task, img_save_dir, config, num_envs=config['controllers']['mppi']['N'])
        sim, gym, viewer = env.get_sim()

    # TODO: fix the object chain
    if task == 'screwdriver_turning':
        asset_object = get_assets_dir() + '/screwdriver/screwdriver.urdf'
        object_chain = pk.build_chain_from_urdf(open(asset_object).read()).to(device=config['device'])
    else:
        object_chain = None

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
            params['object_chain'] = object_chain
            object_location = torch.tensor(env.obj_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            ret = do_trial(env, params, fpath)
            for key in ret.keys():
                if key not in results.keys():
                    results[key] = []
                results[key].append(ret[key])
        print(results)

    for key in results.keys():
        print(f"{key}: avg: {np.array(results[key]).mean()}, std: {np.array(results[key]).std()}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

