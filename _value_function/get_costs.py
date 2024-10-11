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
import matplotlib.pyplot as plt
from utils.allegro_utils import state2ee_pos
from scipy.spatial.transform import Rotation as R
import sys
from get_initial_poses import emailer
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from examples.allegro_valve_roll import PositionControlConstrainedSVGDMPC
from examples.allegro_screwdriver import AllegroScrewdriver

obj_dof = 3
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

def vector_cos(a, b):
    return torch.dot(a.reshape(-1), b.reshape(-1)) / (torch.norm(a.reshape(-1)) * torch.norm(b.reshape(-1)))


fpath = pathlib.Path(f'{CCAI_PATH}/data')
with open(f'{fpath.resolve()}/initial_poses_10k.pkl', 'rb') as file:
    initial_poses  = pkl.load(file)

def do_trial(env, params, fpath, initial_pose_idx = None, sim_viz_env=None, ros_copy_node=None):

    "only turn the screwdriver once"
    screwdriver_goal = params['screwdriver_goal'].cpu()
    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()
    num_fingers = len(params['fingers'])
    action_list = []
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    # sample initial state from dataset
    if initial_pose_idx is None:
        idx = np.random.randint(0, len(initial_poses))
    else:
        idx = initial_pose_idx

    initial_pose = initial_poses[idx]
    env.reset(dof_pos = initial_pose, deterministic=False)

    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    turn_problem_fingers = params['fingers']
    turn_problem_start = start[:4 * num_fingers + obj_dof]
    turn_problem = AllegroScrewdriver(
        start=turn_problem_start,
        goal=params['screwdriver_goal'],
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type=params['object_type'],
        friction_coefficient=params['friction_coefficient'],
        world_trans=env.world_trans,
        fingers=turn_problem_fingers,
        optimize_force=params['optimize_force'],
        force_balance=params['force_balance'],
        collision_checking=params['collision_checking'],
        obj_gravity=params['obj_gravity'],
        contact_region=params['contact_region'],
        static_init=params['static_init'],
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

    num_fingers_to_plan = num_fingers
    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())
        start_time = time.time()
        best_traj, trajectories = turn_planner.step(start[:4 * num_fingers + obj_dof])
        
        if torch.isnan(best_traj).any().item():
            env.reset()
            break
        #debug only
        # turn_problem.save_history(f'{fpath.resolve()}/op_traj.pkl')

        #print(f"solve time: {time.time() - start_time}")
        planned_theta_traj = best_traj[:, 4 * num_fingers_to_plan: 4 * num_fingers_to_plan + obj_dof].detach().cpu().numpy()
        #print(f"current theta: {state['q'][0, -(obj_dof+1): -1].detach().cpu().numpy()}")
        #print(f"planned theta: {planned_theta_traj}")

        # if params['visualize_plan']:
        #     traj_for_viz = best_traj[:, :turn_problem.dx]
        #     traj_for_viz = torch.cat((start[:turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
        #     tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device) # add the joint for the screwdriver cap
        #     traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
        #     # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])
        
        #     viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
        #     img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        #     gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        #     pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        #     pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        #     visualize_trajectory(traj_for_viz, turn_problem.viz_contact_scenes, viz_fpath, turn_problem.fingers, turn_problem.obj_dof+1)
        
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        #print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        if params['optimize_force']:
            # print("planned force")
            # print(action[:, 4 * num_fingers_to_plan:].reshape(num_fingers_to_plan, 3)) # print out the action for debugging
            # print("delta action")
            # print(action[:, :4 * num_fingers_to_plan].reshape(num_fingers_to_plan, 4))
            pass
        # print(action)
        action = action[:, :4 * num_fingers_to_plan]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers].to(env.device) # NOTE: this is required since we define action as delta action
        
        env.step(action)
        action_list.append(action)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        screwdriver_state = env.get_state()['q'][:, -obj_dof-1: -1].cpu()
        screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
            torch.tensor(screwdriver_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()

        # distance2goal = (screwdriver_goal - screwdriver_state)).detach().cpu()
        if torch.isnan(distance2goal).item():
            env.reset()
            break
        #print(distance2goal)
        if torch.isnan(distance2goal).any().item():
            env.reset()
            #print("NAN")
            break
        # info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
        # info_list.append(info)

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

    # with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        # pkl.dump(info_list, f)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # newLabels, newHandles = [], []
    # for handle, label in zip(handles, labels):
    #   if label not in newLabels:
    #     newLabels.append(label)
    #     newHandles.append(handle)
    # fig.tight_layout()
    # fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    # plt.savefig(f'{fpath.resolve()}/traj.png')
    # plt.close()
    # plt.show()

    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    actual_trajectory.append(state.clone()[:4 * num_fingers + obj_dof])
    actual_trajectory = [tensor.to(device=params['device']) for tensor in actual_trajectory]
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    screwdriver_state = actual_trajectory[:, -obj_dof:].cpu()
    screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
    distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
        torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0],1,1), cos_angle=False).detach().cpu()

    final_distance_to_goal = torch.min(distance2goal.abs())
    final_cost = turn_problem._cost(state.reshape(1,-1), start, goal).detach().cpu().item()
    #print("final cost: ", final_cost)

    state = env.get_state()['q']
    final_state = torch.cat((
                    state.clone()[:, :8], 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]), 
                    state.clone()[:, 8:], 
                    ), dim=1).detach().cpu().numpy()
    
    #print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}')
    #print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    # env.reset()
    return final_distance_to_goal.cpu().detach().item(), final_cost, final_state, idx

if __name__ == "__main__":
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
    from tqdm import tqdm
    sim_env = None
    ros_copy_node = None

    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=False,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=True,
                                gravity=True,
                                )
    sim, gym, viewer = env.get_sim()

    state = env.get_state()
    results = {}
    succ_rate = {}

    # set up the kinematic chain
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

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) 

    pose_cost_tuples = []
    final_states = []
    n_poses = len(initial_poses)

    for i in tqdm(range(1000)):
        goal = - 90 / 180 * torch.tensor([0, 0, np.pi])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            succ = False
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()

            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['screwdriver_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location

            idx = i + params['start_idx']
            final_distance_to_goal,final_cost,final_state, initial_pose_index = do_trial(env, params, fpath, idx, sim_env, ros_copy_node)
            pose_cost_tuples.append((initial_poses[initial_pose_index], final_cost))
            final_states.append(final_state)
            
            if final_distance_to_goal < 30 / 180 * np.pi:
                succ = True

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
                succ_rate[controller] = [succ]
            else:
                results[controller].append(final_distance_to_goal)
                succ_rate[controller].append(succ)
        # print(results)
        # print(succ_rate)
    

    print(f"Average final distance to goal: {torch.mean(torch.tensor(results['csvgd']))}, std: {torch.std(torch.tensor(results['csvgd']))}")
    print(f"Success rate: {torch.mean(torch.tensor(succ_rate['csvgd']).float())}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    start_idx = params['start_idx']
    savepath = f'{fpath.resolve()}/value_dataset_odin_{start_idx}.pkl'
    with open(savepath, 'wb') as f:
        pkl.dump(pose_cost_tuples, f)

    savepath_states = f'{fpath.resolve()}/final_states.pkl'
    with open(savepath_states, 'wb') as f:
        pkl.dump(final_states, f)

    print(f'saved to {savepath}')
    emailer().send()