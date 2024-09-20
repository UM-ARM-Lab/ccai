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
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
import matplotlib.pyplot as plt
from utils.allegro_utils import *
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from allegro_screwdriver import ALlegroScrewdriverContact
from scipy.spatial.transform import Rotation as R
from baselines.planning.ik import IKSolver

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    fingers=['index', 'middle', 'thumb']

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())

    params = config.copy()
    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                #video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=True,
                                )

    sim, gym, viewer = env.get_sim()
    state = env.get_state()
    device = config['sim_device']
    params['device'] = device

    dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float().to(device=device),
                        torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float().to(device=device),
                        torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=device),
                        torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float().to(device=device)),
                        dim=1).to(device)
    dof_pos = torch.cat((dof_pos, torch.zeros((1, 4)).float().to(device=device)),
                                         dim=1).to(device)
    dof_pos = dof_pos.repeat(1, 1)
    partial_default_dof_pos = np.concatenate((dof_pos[:, 0:8], dof_pos[:, 12:16]), axis=1)

    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    params['ee_names'] = ee_names
    params['obj_dof_code'] = [0, 0, 0, 1, 1, 1]
    params['obj_dof'] = np.sum(params['obj_dof_code'])

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    
    chain = pk.build_chain_from_urdf(open(asset).read())
    params['chain'] = chain.to(device=device)
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in fingers]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=fingers, chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
    params['object_location'] = object_location
    params.update(config['controllers']['csvgd'])
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices)

    fk = forward_kinematics(dof_pos[:,:16])

    # list of poses for each of the three fingers, each pose is a Transform3d object
    default_poses = list(fk.values())
    
    world_trans = env.world_trans

    def world_to_robot_frame(point):
        tform = world_trans.inverse()
        return tform.transform_points(point.reshape(1, 3))
    
    def screwdriver_to_world_tform(r=0, p=0):
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(r), -np.sin(r), 0],
            [0, np.sin(r), np.cos(r), 0],
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(p), 0, np.sin(p), 0],
            [0, 1, 0, 0],
            [-np.sin(p), 0, np.cos(p), 0],
            [0, 0, 0, 1]
        ])
        Tz = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1.330],
            [0, 0, 0, 1]
        ])
        T = np.dot(Tz, np.dot(Ry, Rx))
        return T
    
    def screwdriver_to_robot_frame(p, roll = 0, pitch = 0):
        world_to_robot = world_trans.inverse().get_matrix()
        sd_to_world = screwdriver_to_world_tform(roll, pitch)
        p = np.array([p[0], p[1], p[2], 1])
        world = np.dot(sd_to_world, p)
            
        #print(world)
        #print(np.dot(world_to_robot, world)[:,:3])

        #exit()

        return np.dot(world_to_robot, world)[:,:3]
    
    
    cylinder_center_world = torch.tensor([[0, 0, 1.330]])
    cylinder_center = world_to_robot_frame(cylinder_center_world)
    cylinder_radius = 0.02
    cylinder_height = 0.1

    def get_circle_xy(radius0, radius1, theta_0, theta_1):
        theta = np.random.rand(1) * (theta_1 - theta_0) + theta_0
        radius = np.random.rand(1) * (radius1 - radius0) + radius0
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x.item(), y.item()

    def get_index_goal():

        circle_x, circle_y = get_circle_xy(0, cylinder_radius, 0, 2*np.pi)
        z_offset = cylinder_height/2

        goal_screwdriver_frame = torch.tensor([circle_x, circle_y, z_offset]).to(device)
        goal_robot_frame = screwdriver_to_robot_frame(goal_screwdriver_frame).reshape(1,3)

        #goal_robot_frame = default_poses[0].clone().reshape(1,3)
        # goal_robot_frame = np.array([[0.0, 0.0, 0.0]])
        return goal_robot_frame

    def get_middle_goal():
        
        circle_x, circle_y = get_circle_xy(cylinder_radius, cylinder_radius, np.pi-np.pi/4, np.pi+np.pi/4)
        z_offset = float(np.random.rand(1)) * cylinder_height - cylinder_height/2
        # z_offset = 0

        goal_screwdriver_frame = torch.tensor([circle_x, circle_y, z_offset]).to(device)
        goal_robot_frame = screwdriver_to_robot_frame(goal_screwdriver_frame).reshape(1,3)

        #goal_robot_frame = default_poses[1].clone().reshape(1,3)
        # goal_robot_frame = np.array([[0.0, 0.0, 0.0]])
        return goal_robot_frame
    
    
    def get_thumb_goal():

        circle_x, circle_y = get_circle_xy(cylinder_radius, cylinder_radius, -np.pi/4, np.pi/4)
        z_offset = float(np.random.rand(1)) * cylinder_height - cylinder_height/2 
        # z_offset = 0

        goal_screwdriver_frame = torch.tensor([circle_x, circle_y, z_offset]).to(device)
        goal_robot_frame = screwdriver_to_robot_frame(goal_screwdriver_frame).reshape(1,3)

        #goal_robot_frame = default_poses[2].clone().reshape(1,3)
        # goal_robot_frame = np.array([[0.0, 0.0, 0.0]])
        return goal_robot_frame

    initial_poses = []
    contact_points = []
    costs = []
    fpath = pathlib.Path(f'{CCAI_PATH}/data')

    env.reset(dof_pos)
    for i in range(50):
        print("iteration: ", i)

        #goal_poses = default_poses.copy()
        goal_poses = torch.tensor([get_index_goal(), get_middle_goal(), get_thumb_goal()]).reshape(3,3).to(device)

        obj_dof = 3
        num_fingers = len(params['fingers'])
        start = state['q'].reshape(4 * num_fingers + 4).to(device=device)
        if 'index' in params['fingers']:
            contact_fingers = params['fingers']
        else:
            contact_fingers = ['index'] + params['fingers']    
        pregrasp_problem = ALlegroScrewdriverContact(
            dx=4 * num_fingers,
            du=4 * num_fingers,
            start=start[:4 * num_fingers + obj_dof],
            goal=None,
            T=3,
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
            goal_poses=goal_poses,
        )
        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
        pregrasp_planner.warmup_iters = 100#500 #50

        start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)
        best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])

        traj_for_viz = best_traj[:, :pregrasp_problem.dx]
        tmp = start[4 * num_fingers:].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)    
        viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, pregrasp_problem.fingers, pregrasp_problem.obj_dof+1)

        #for x in best_traj[:, :4 * num_fingers]:
        x = best_traj[-1, :4 * num_fingers]

        cost = 0
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) 
        solved_pos = torch.cat((action.clone()[:,:8], 
                torch.tensor([[0., 0.5, 0.65, 0.65]]).to(device=device), 
                action.clone()[:,8:], 
                torch.zeros((1,4)).float().to(device=device)), 
                dim=1).to(device)
        env.reset(solved_pos)
        #env.step(action)
        #action_list.append(action)

        #solved_pos = 0
        #env.reset(solved_pos)
        initial_poses.append(solved_pos.cpu())
        contact_points.append(goal_poses.clone().cpu().numpy())
        costs.append(cost)

        env.gym.write_viewer_image_to_file(env.viewer, f'{fpath.resolve()}/initial_pose_frames.pkl/frame_{i}.png')


    with open(f'{fpath.resolve()}/initial_poses_free.pkl', 'wb') as f:
        pkl.dump(initial_poses, f)
    with open(f'{fpath.resolve()}/contact_points_free.pkl', 'wb') as f:
        pkl.dump(contact_points, f)
    with open(f'{fpath.resolve()}/costs.pkl', 'wb') as f:
        pkl.dump(costs, f)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)