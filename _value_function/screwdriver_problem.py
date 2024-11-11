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
from utils.allegro_utils import state2ee_pos, visualize_trajectory
from scipy.spatial.transform import Rotation as R
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from examples.allegro_valve_roll import PositionControlConstrainedSVGDMPC
from examples.allegro_screwdriver import AllegroScrewdriver, ALlegroScrewdriverContact
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
    img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
    sim_env = None
    ros_copy_node = None
    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=visualize,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=True,
                                gravity=True,
                                randomize_obj_start= True
                                #device = config['device'],
                                )
    sim, gym, viewer = env.get_sim()
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
    
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos_partial = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)

    return config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial

def pregrasp(env, config, chain, deterministic=True, initialization = None, useVFgrads=False, vis_plan = False, iters = 80):
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
    if initialization is None:
        if deterministic is False:
            index_noise_mag = torch.tensor([0.08]*4)
            index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
            middle_ring_noise_mag = torch.tensor([0.15]*4)
            middle_ring_noise = middle_ring_noise_mag * (2 * torch.rand(4) - 1)
            screwdriver_noise = torch.tensor([
            np.random.uniform(-0.05, 0.05),  # Random value between -0.05 and 0.05
            np.random.uniform(-0.05, 0.05),  # Random value between -0.05 and 0.05
            np.random.uniform(0, 2 * np.pi),  # Random value between 0 and 2π
            0.0  
            ])
            
        
        else:
            index_noise = torch.tensor([0., 0., 0., 0.])
            middle_ring_noise = torch.tensor([0., 0., 0., 0.])
            screwdriver_noise = torch.tensor([0., 0., 0., 0.])

        default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + index_noise,
                                    torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + middle_ring_noise,
                                    torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + middle_ring_noise,
                                    torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=sim_device),
                                    torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float().to(device=sim_device) + screwdriver_noise),
                                    dim=1).to(sim_device)
    else:
        default_dof_pos = initialization

    env.reset(dof_pos= default_dof_pos)

    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)
    print("start: ", start)

    screwdriver = start.clone()[-4:-1]
    #print("start screwdriver: ", screwdriver)
    screwdriver = torch.cat((screwdriver, torch.tensor([0]).to(device=device)),dim=0).reshape(1,4)

    if 'index' in params['fingers']:
        contact_fingers = params['fingers']
    else:
        contact_fingers = ['index'] + params['fingers']    

    pregrasp_problem = ALlegroScrewdriverContact(
        dx=4 * num_fingers,
        du=4 * num_fingers,
        start=start[:4 * num_fingers + obj_dof],
        goal=None,
        T=2,
        chain=params['chain'],
        device=device,
        object_asset_pos=env.table_pose,
        object_location=params['object_location'],
        object_type=params['object_type'],
        world_trans=env.world_trans,
        fingers=contact_fingers,
        obj_dof_code=params['obj_dof_code'],
        obj_joint_dim=1,
        fixed_obj=True,
        useVFgrads=useVFgrads,
    )
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
    pregrasp_planner.warmup_iters = iters#500 #50

    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])

    if vis_plan:
        robot_dof = 4 * num_fingers
        traj_for_viz = best_traj[:, :pregrasp_problem.dx]
        tmp = start[robot_dof:].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)  
        fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_0')
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)  
        viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, 
                                pregrasp_problem.fingers, pregrasp_problem.obj_dof+1,)
        


    action = best_traj[-1, :4 * num_fingers]
    action = action.reshape(-1, 4 * num_fingers).to(device=device) 
    solved_pos = torch.cat((
            action.clone()[:, :8], 
            torch.tensor([[0., 0.5, 0.65, 0.65]]).to(device=device), 
            action.clone()[:, 8:], 
            screwdriver.to(device=device)
            #torch.zeros(1,4).to(device=device)
            ), dim=1).to(device)

    env.reset(dof_pos = solved_pos.to(device = sim_device))

    return solved_pos.cpu()

def solve_turn(env, gym, viewer, params, fpath, initial_pose, state2ee_pos_partial, sim_viz_env=None, ros_copy_node=None):

    obj_dof = 3
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')
    fpath = pathlib.Path(f'{CCAI_PATH}/data')

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

    env.reset(dof_pos = initial_pose)

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
        ee = state2ee_pos_partial(start[:4 * num_fingers], turn_problem.ee_names[finger])
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

        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        #print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
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
        info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
        info_list.append(info)

        gym.clear_lines(viewer)
        state = env.get_state()
        start = state['q'][:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
        for finger in params['fingers']:
            ee = state2ee_pos_partial(start[:4 * num_fingers], turn_problem.ee_names[finger])
            finger_traj_history[finger].append(ee.detach().cpu().numpy())
        for finger in params['fingers']:
            traj_history = finger_traj_history[finger]
            temp_for_plot = np.stack(traj_history, axis=0)
            if k >= 2:
                axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')

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
    # np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
    #     d2goal=final_distance_to_goal.cpu().numpy())

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
    
    return final_distance_to_goal.cpu().detach().item(), final_state, full_trajectory #final_cost, 

def do_turn( initial_pose, config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial):

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
    goal = torch.tensor([0, 0, -np.pi/2]) + screwdriver_pose.clone()

    params['screwdriver_goal'] = goal.to(device=params['device'])
    params['chain'] = chain.to(device=params['device'])
    object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
    params['object_location'] = object_location

    final_distance_to_goal, final_pose, full_trajectory = solve_turn(env, gym, viewer, params, fpath, initial_pose, state2ee_pos_partial, sim_env, ros_copy_node)
    
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

if __name__ == "__main__":

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    initial_poses = pkl.load(open(f'{fpath.resolve()}/initial_poses/initial_poses_10k.pkl', 'rb'))
    # show_state(env, initial_poses[0], t=2)
    # show_state(env, initial_poses[1], t=2)
    for i in range(10):
        pregrasp(env, config, chain, deterministic=False, initialization = None, useVFgrads=True)
    # initial_pose, final_pose, succ, full_trajectory = do_turn(initial_poses[0], config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)
    