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
from scipy.spatial.transform import Rotation as R
from baselines.planning.ik import IKSolver

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    fingers=['index', 'middle', 'thumb']

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
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
    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in fingers]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=fingers, chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices)

    fk = forward_kinematics(dof_pos[:,:16])

    # list of poses for each of the three fingers, each pose is a Transform3d object
    default_poses = list(fk.values())
    
    world_trans = env.world_trans
    def world_to_robot_frame(point):
        tform = world_trans.inverse()
        return tform.transform_points(point.reshape(1, 3)) - tform.get_matrix()[:, :3, 3]
    
    cylinder_center_world = torch.tensor([[0, 0, -0.02+0.1+0.05]])
    cylinder_center = world_to_robot_frame(cylinder_center_world)
    #cylinder_center = np.array([[0.08439247, 0.05959691, 0.05529295]])
    cylinder_center = np.array([])
    cylinder_radius = 0.02
    cylinder_height = 0.1

    def get_circle_xy(radius, theta_0, theta_1):
        theta = np.random.rand(1) * (theta_1 - theta_0) + theta_0
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x.item(), y.item()

    def get_index_goal(delta = 0):
        goal_index = default_poses[0].clone()

        circle_x, circle_y = get_circle_xy(cylinder_radius, 0, 3.14)
        z_offset = cylinder_height/2

        offset_world_frame = torch.tensor([[circle_x - delta, circle_y, z_offset]]).to(device)
        offset_robot_frame = world_to_robot_frame(offset_world_frame).cpu().numpy()

        pos_index = cylinder_center + offset_robot_frame
        pos_index = torch.tensor(pos_index).to(device)
        goal_index._matrix[:, :3, 3] = pos_index.clone()
        return goal_index

    def get_middle_goal(delta = 0):
        goal_middle = default_poses[1].clone()
        
        z_offset_range = 0.012
        z_offset = float(np.random.rand(1)) * 2*z_offset_range - z_offset_range
        circle_x, circle_y = get_circle_xy(cylinder_radius, 0, 3.14)

        offset_world_frame = torch.tensor([[circle_x - delta, circle_y, z_offset]]).to(device)
        offset_robot_frame = world_to_robot_frame(offset_world_frame).cpu().numpy()

        pos_middle = cylinder_center + offset_robot_frame
        pos_middle = torch.tensor(pos_middle).to(device)
        goal_middle._matrix[:, :3, 3] = pos_middle
        return goal_middle
    
    def get_thumb_goal(delta = 0):
        goal_thumb = default_poses[2].clone()
        z_offset_range = 0.012
        z_offset = float(np.random.rand(1)) * 2*z_offset_range - z_offset_range
        circle_x, circle_y = get_circle_xy(cylinder_radius, 0, 3.14)

        offset_world_frame = torch.tensor([[circle_x - delta, circle_y, z_offset]]).to(device)
        offset_robot_frame = torch.tensor([[0,0,delta]])
        offset_robot_frame = world_to_robot_frame(offset_world_frame).cpu().numpy()
        
        og = default_poses.copy()[2].get_matrix().cpu().numpy()[:, :3, 3]
        #pos_thumb = cylinder_center + offset_robot_frame
        pos_thumb = og + offset_robot_frame
        pos_thumb = torch.tensor(pos_thumb).to(device)
        goal_thumb._matrix[:, :3, 3] = pos_thumb
        return goal_thumb

    lim = torch.tensor(chain.get_joint_limits(), device=device)
    
    ik = IKSolver(chain, fingers, device)
    #sol = ik.do_IK(default_poses, ignore_dims=[3,4,5]).view(1, 12)

    delta_0 = 0.0
    delta_1 = 0.0
    delta_2 = 0.0#15

    # thumb z +- 0.012
    initial_poses = []
    fpath = pathlib.Path(f'{CCAI_PATH}/data')

    #print(default_poses[0].get_matrix().cpu().numpy()[:, :3, 3])

    env.reset(dof_pos)
    for i in range(200):
        print("iteration: ", i)

        #goal_poses = [get_index_goal(delta_0), get_middle_goal(delta_1), get_thumb_goal(delta_2)]
        goal_poses = default_poses.copy()
        goal_poses[2] = get_thumb_goal(delta_2)
        delta_2 = 0#np.random.rand(1).item() * 0.015 - 0.015/2
        #print(delta_1)
        #print(goal_poses[0].get_matrix().cpu().numpy()[:, :3, 3])
        
        sol = ik.do_IK(goal_poses, ignore_dims=[3,4,5], current= partial_default_dof_pos.copy()).view(1, 12)
        solved_pos = torch.cat((sol.clone()[:,:8], 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]).to(device=device), 
                    sol.clone()[:,8:], 
                    torch.zeros((1,4)).float().to(device=device)), 
                    dim=1).to(device)
        env.reset(solved_pos)

        initial_poses.append(solved_pos.cpu())

    with open(f'{fpath.resolve()}/initial_poses.pkl', 'wb') as f:
        pkl.dump(initial_poses, f)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)