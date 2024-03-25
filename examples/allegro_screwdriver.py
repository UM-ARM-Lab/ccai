from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroValveTurningEnv

import numpy as np
import pickle as pkl

import torch
import time
import yaml
import pathlib
from functools import partial

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
# import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos, visualize_trajectory
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

device = 'cuda:0'
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

def axis_angle_to_euler(axis_angle):
    matrix = tf.axis_angle_to_matrix(axis_angle)
    euler = tf.matrix_to_euler_angles(matrix, convention='XYZ')
    return euler

class AllegroIndexPlanner:
    "The index finger is desgie"
    def __init__(self, chain_hand, chain_screwdriver, screwdriver_asset_pose) -> None:
        pass

class AllegroScrewdriverContact(AllegroContactProblem):
    def __init__(self, 
                 dx,
                 du,
                 start, 
                 goal, 
                 T, 
                 chain, 
                 object_location, 
                 object_type,
                 world_trans,
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 obj_dof=obj_dof,
                 obj_ori_rep='axis_angle',
                 device='cuda:0'):
        super(AllegroScrewdriverContact, self).__init__(dx=dx, du=du, start=start, goal=goal, T=T, chain=chain,
                                    object_location=object_location, object_type=object_type,
                                    world_trans=world_trans,object_asset_pos=object_asset_pos,
                                    fingers=fingers, obj_dof=obj_dof, obj_ori_rep=obj_ori_rep, 
                                    obj_joint_dim=1, device=device)
        
        asset_screwdriver = get_assets_dir() + '/screwdriver/screwdriver.urdf'

        chain_screwdriver = pk.build_chain_from_urdf(open(asset_screwdriver).read())
        chain_screwdriver = chain_screwdriver.to(device=device)
        screwdriver_sdf = pv.RobotSDF(chain_screwdriver, path_prefix=get_assets_dir() + '/screwdriver')
        robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models')

        scene_trans = world_trans.inverse().compose(
            pk.Transform3d(device=device).translate(object_asset_pos[0], object_asset_pos[1], object_asset_pos[2]))

        # contact checking
        self.contact_scenes = {}
        for finger in self.fingers:
            self.contact_scenes[finger] = pv.RobotScene(robot_sdf, screwdriver_sdf, scene_trans,
                                                          collision_check_links=[self.ee_names[finger]],
                                                          softmin_temp=1.0e3,
                                                          points_per_link=1000,
                                                          )
            

class AllegroScrewdriver(AllegroValveTurning):
    # def get_constraint_dim(self, T):
    #     self.friction_polytope_k = 4
    #     if self.optimize_force:
    #         self.dg_per_t = self.num_fingers * (1 + 3 + 4) + 3# 1 contact, 3 kinematics, 2 dynamics, 3 force equlibrium
    #     else:
    #         self.dg_per_t = self.num_fingers * (1 + 3) + 3# 1 contact, 3 kinematics, 2 dynamics, 3 force equlibrium
    #     self.dg_constant = 0
    #     self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
    #     self.dz = (self.friction_polytope_k) * self.num_fingers # one friction constraints per finger
    #     self.dh = self.dz * T  # inequality
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
                 obj_ori_rep='axis_angle',
                 optimize_force=False,
                 device='cuda:0', **kwargs):
        self.num_fingers = len(fingers)
        self.obj_dof = 3
        self.optimize_force = optimize_force
        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof=obj_dof, 
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1, optimize_force=optimize_force, device=device)
        self.friction_coefficient = friction_coefficient
        self.grad_axis_angle_to_euler = jacrev(axis_angle_to_euler)

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        smoothness_cost += 50 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        # smoothness_cost += 500 * torch.sum((state[:, -self.obj_dof:-1]) ** 2) # the screwdriver should only rotate in z direction

        goal_cost = torch.sum((100 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((10 * (state[:, -1] - goal) ** 2), dim=0)

        return smoothness_cost + action_cost + goal_cost
    
                
    @combine_finger_constraints
    def _valve_kinematics_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):
        """
            Computes on the kinematics of the valve and the finger being consistant
        """
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        contact_jacobian = ret_scene.get('contact_jacobian', None)
        contact_loc = ret_scene.get('closest_pt_world', None)
        d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
        dJ_dq = ret_scene.get('dJ_dq', None)

        # approximate q dot and theta dot
        dq = (x[:, 1:, :4 * self.num_fingers] - x[:, :-1, :4 * self.num_fingers])
        # assume axis angle representation
        dtheta = (x[:, 1:, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] - x[:, :-1, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof])
        valve_omega = dtheta
        # compute robot contact point velocity
        contact_point_v = (contact_jacobian[:, :-1] @ dq.reshape(N, T, 4 * self.num_fingers, 1)).squeeze(-1)  # should be N x T x 3

        # compute valve contact point velocity
        valve_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        contact_point_r_valve = contact_loc.reshape(N, T + 1, 3) - valve_robot_frame.reshape(1, 1, 3)
        valve_omega_robot_frame = self.world_trans.inverse().transform_normals(valve_omega)
        object_contact_point_v = torch.cross(valve_omega_robot_frame, contact_point_r_valve[:, :-1])

        # kinematics constraint, should be 3-dimensional
        # we actually ended up computing T+1 contact constraints, but start state is fixed so we throw that away
        g = (contact_point_v - object_contact_point_v).reshape(N, -1)

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_range_minus = torch.arange(T - 1, device=x.device)
            T_range_plus = torch.arange(1, T, device=x.device)

            # Compute gradient w.r.t q
            dcontact_v_dq = (dJ_dq[:, 1:] @ dq.reshape(N, T, 1, 4 * self.num_fingers, 1)).squeeze(-1) - contact_jacobian[:, 1:]
            tmp = torch.cross(d_contact_loc_dq[:, 1:], valve_omega.reshape(N, T, 3, 1), dim=2)  # N x T x 3 x 8
            dg_dq = dcontact_v_dq - tmp

            # Compute gradient w.r.t valve angle
            d_omega_dtheta_0 = torch.stack((torch.ones((N, T)),
                                          torch.zeros((N, T)),
                                          torch.zeros((N, T))), dim=-1).to(self.device)  # N x T x 3
            d_omega_dtheta_1 = torch.stack((torch.zeros((N, T)),
                                          torch.ones((N, T)),
                                          torch.zeros((N, T))), dim=-1).to(self.device) 
            d_omega_dtheta_2 = torch.stack((torch.zeros((N, T)),
                                          torch.zeros((N, T)),
                                          torch.ones((N, T))), dim=-1).to(self.device) 
            dg_dtheta_0 = torch.cross(self.world_trans.inverse().transform_normals(d_omega_dtheta_0), contact_point_r_valve[:, :-1], dim=-1)
            dg_dtheta_1 = torch.cross(self.world_trans.inverse().transform_normals(d_omega_dtheta_1), contact_point_r_valve[:, :-1], dim=-1)
            dg_dtheta_2 = torch.cross(self.world_trans.inverse().transform_normals(d_omega_dtheta_2), contact_point_r_valve[:, :-1], dim=-1)

            dg_dtheta = torch.stack((dg_dtheta_0, dg_dtheta_1, dg_dtheta_2), dim=-1)
            
            # assemble gradients into a single (sparse) tensor
            grad_g = torch.zeros((N, T, T, 3, self.dx + self.du), device=x.device)
            grad_g[:, T_range_plus, T_range_minus, :, :4 * self.num_fingers] = dg_dq[:, 1:]
            grad_g[:, T_range_plus, T_range_minus, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dg_dtheta[:, 1:]
            grad_g[:, T_range, T_range, :, :4 * self.num_fingers] = contact_jacobian[:, :-1]
            grad_g[:, T_range, T_range, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = -dg_dtheta
            grad_g = grad_g.permute(0, 1, 3, 2, 4).reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None
    
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the valve once"
    num_fingers = len(params['fingers'])
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    if params['controller'] == 'csvgd':
        # index finger is used for stability
        if 'index' in params['fingers']:
            contact_fingers = params['fingers']
        else:
            contact_fingers = ['index'] + params['fingers']        
        pregrasp_problem = AllegroScrewdriverContact(
            dx=4 * num_fingers + obj_dof,
            du=4 * num_fingers,
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=4,
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            fingers=contact_fingers,
            obj_dof=obj_dof
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)

        turn_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            friction_coefficient=params['friction_coefficient'],
            world_trans=env.world_trans,
            fingers=params['fingers'],
            obj_dof=obj_dof,
            optimize_force=params['optimize_force']
        )

    else:
        raise ValueError('Invalid controller')
    
    
    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers + obj_dof])

    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        env.step(action)
        if params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, 4 * num_fingers)[0], params['fingers']))

    turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)

    actual_trajectory = []
    duration = 0

    # debug: plot the thumb traj
    fig = plt.figure()
    axes = [fig.add_subplot(int(f'1{num_fingers}{i+1}'), projection='3d') for i in range(num_fingers)]
    for i, ax in enumerate(axes):
        axes[i].set_title(params['fingers'][i])
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('x', labelpad=20)
        axes[i].set_ylabel('y', labelpad=20)
        axes[i].set_zlabel('z', labelpad=20)
        axes[i].set_xlim3d(0.8, 0.87)
        axes[i].set_ylim3d(0.52, 0.58)
        axes[i].set_zlim3d(1.36, 1.46)
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []
    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    for finger in params['fingers']:
        ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())
        start_time = time.time()
        best_traj, trajectories = turn_planner.step(start[:4 * num_fingers + obj_dof])

        print(f"solve time: {time.time() - start_time}")
        # TODO: transform the representation for theta
        planned_theta_traj = best_traj[:, 4 * num_fingers: 4 * num_fingers + obj_dof].detach().cpu().numpy()
        print(f"current theta: {state['q'][0, -1].detach().cpu().numpy()}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        if params['mode'] == 'hardware':
            add_trajectories_hardware(trajectories, best_traj, axes)
        else:
            add_trajectories(trajectories, best_traj, axes)

        if params['visualize_plan']:
            traj_for_viz = best_traj[:, :turn_problem.dx]
            traj_for_viz = torch.cat((start[:turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device) # add the joint for the screwdriver cap
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])
        
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, turn_problem.contact_scenes['index'], viz_fpath, turn_problem.fingers, turn_problem.obj_dof+1)
        
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        print("--------------------------------------")
        # print(f'Equality constraint violation: {torch.norm(equality_constr)}')
        # print(f'Inequality constraint violation: {torch.norm(inequality_constr)}')

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        print(action)
        action = action[:, :4 * num_fingers]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers] # NOTE: this is required since we define action as delta action
        # action = best_traj[0, :8]
        # action[:, 4:] = 0
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        # action = x[:, :4 * num_fingers].to(device=env.device)
        env.step(action)
        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        # print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
        #                                                         compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - object_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - object_location[0].unsqueeze(0))**2)
        distance2goal = (params['valve_goal'].cpu() - env.get_state()['q'][:, -obj_dof-1: -1].cpu()).detach().cpu()
        print(distance2goal)
        info_list.append({
                        #   'distance': distance, 
                          'distance2goal': distance2goal, 
                        })

        gym.clear_lines(viewer)
        # for debugging
        state = env.get_state()
        start = state['q'][:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
        for finger in params['fingers']:
            ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
            finger_traj_history[finger].append(ee.detach().cpu().numpy())
        for i, ax in enumerate(axes):
            finger = params['fingers'][i]
            traj_history = finger_traj_history[finger]
            temp_for_plot = np.stack(traj_history, axis=0)
            if k >= 2:
                axes[i].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    fig.tight_layout()
    fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    # plt.savefig(f'{fpath.resolve()}/traj.png')
    # plt.close()
    plt.show()



    env.reset()
    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + 1).to(device=params['device'])
    actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + 1)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = (actual_trajectory[:, -1] - params['valve_goal']).abs()

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


def add_trajectories(trajectories, best_traj, axes=None):
    M = len(trajectories)
    fingers = config['fingers']
    if M > 0:
        initial_state = env.get_state()['q']
        num_fingers = (initial_state.shape[1] - obj_dof - 1) // 4
        initial_state = initial_state[:, :4 * num_fingers]
        all_state = torch.cat((initial_state, best_traj[:-1, :4 * num_fingers]), dim=0)
        desired_state = all_state + best_traj[:, 4 * num_fingers + obj_dof: 4 * num_fingers + obj_dof + 4 * num_fingers]
        
        desired_best_traj_ee = [state2ee_pos(desired_state, ee_names[finger]) for finger in fingers]
        best_traj_ee = [state2ee_pos(best_traj[:, :4 * num_fingers], ee_names[finger]) for finger in fingers]
        state_colors = np.array([0, 0, 1]).astype(np.float32)
        force_colors = np.array([0, 1, 1]).astype(np.float32)
        
        for e in env.envs:
            T = best_traj.shape[0]
            for t in range(T):
                for i, finger in enumerate(fingers):
                    if t == 0:
                        initial_ee = state2ee_pos(initial_state, ee_names[finger])
                        state_traj = torch.stack((initial_ee, best_traj_ee[i][0]), dim=0).cpu().numpy()
                        action_traj = torch.stack((initial_ee, desired_best_traj_ee[i][0]), dim=0).cpu().numpy()
                        axes[i].plot3D(state_traj[:, 0], state_traj[:, 1], state_traj[:, 2], 'blue', label='desired next state')
                        axes[i].plot3D(action_traj[:, 0], action_traj[:, 1], action_traj[:, 2], 'green', label='raw commanded position')
                    else:
                        state_traj = torch.stack((best_traj_ee[i][t - 1, :3], best_traj_ee[i][t, :3]), dim=0).cpu().numpy()
                        action_traj = torch.stack((best_traj_ee[i][t - 1, :3], desired_best_traj_ee[i][t, :3]), dim=0).cpu().numpy()
                    
                    state_traj = state_traj.reshape(2, 3)
                    action_traj = action_traj.reshape(2, 3)
                
                    gym.add_lines(viewer, e, 1, state_traj, state_colors)
                    gym.add_lines(viewer, e, 1, action_traj, force_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

def add_trajectories_hardware(trajectories, best_traj, axes=None):
    M = len(trajectories)
    fingers = config['fingers']
    if M > 0:
        initial_state = env.get_state()['q']
        num_fingers = initial_state.shape[1] // 4
        initial_state = initial_state[:, :4 * num_fingers]
        all_state = torch.cat((initial_state, best_traj[:-1, :4 * num_fingers]), dim=0)
        desired_state = all_state + best_traj[:, 4 * num_fingers + obj_dof: 4 * num_fingers + obj_dof + 4 * num_fingers]

        desired_best_traj_ee = [state2ee_pos(desired_state, ee_names[finger]) for finger in fingers]
        best_traj_ee = [state2ee_pos(best_traj[:, :4 * num_fingers], ee_names[finger]) for finger in fingers]

        for i, finger in enumerate(fingers):
            initial_ee = state2ee_pos(initial_state, ee_names[finger])
            state_traj = torch.stack((initial_ee, best_traj_ee[i][0]), dim=0).cpu().numpy()
            action_traj = torch.stack((initial_ee, desired_best_traj_ee[i][0]), dim=0).cpu().numpy()
            axes[i].plot3D(action_traj[:, 0], action_traj[:, 1], action_traj[:, 2], 'green', label='raw commanded position')

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
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
        env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
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
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'

    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    # full_to_partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])


    for i in tqdm(range(config['num_trials'])):
        goal = - 0.5 * torch.tensor([0, 0, np.pi])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
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
            object_location = torch.tensor([0, 0, 1.205]).to(device) # TODO: confirm if this is the correct location
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
