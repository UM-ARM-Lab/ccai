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
from torch.func import vmap, jacrev, hessian, jacfwd

import matplotlib.pyplot as plt
from utils.allegro_utils import *
from examples.allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R

class AblationAllegroValveTurning(AllegroValveTurning):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        self.dg_per_t = self.num_fingers * (1 + 4) + wrench_dim
        self.dg_constant = 0
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = (self.friction_polytope_k) * self.num_fingers # one friction constraints per finger
        self.dz += self.num_fingers # min force constraint
        self.dh = self.dz * T  # inequality
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 contact_obj_frame,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 optimize_force=False,
                 obj_dof_code=[0, 0, 0, 0, 1, 0],
                 du=None,
                 obj_joint_dim=0, 
                 device='cuda:0', 
                 **kwargs):
        super(AblationAllegroValveTurning, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=obj_dof_code, 
                                                 obj_joint_dim=obj_joint_dim, optimize_force=optimize_force, 
                                                 collision_checking=False,
                                                 contact_region=False, du=du, device=device)
        self.friction_coefficient = friction_coefficient
        self.grad_contact_point_constr = jacrev(self._contact_point_constr, argnums=(0))

        self.contact_obj_frame = {}
        for i, finger in enumerate(self.fingers):
            self.contact_obj_frame[finger] = contact_obj_frame[i]

    def _contact_point_constr(self, x, desired_ee_obj_frame, finger_name):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector
        NT = x.shape[0]

        obj_state = x[:, -self.obj_dof:]
        obj_ori = obj_state[:, -self.obj_rotational_dim:]
        if self.obj_rotational_dim == 3:
            obj_ori_3d = obj_ori
        elif self.obj_rotational_dim == 1:
            obj_ori_3d = torch.zeros((NT, 3), device=self.device)
            obj_ori_3d[:, 1] = obj_ori[:, 0]
            assert self.obj_dof_code[4] == 1
        if self.obj_translational_dim > 0:
            obj_pos = obj_state[:, :self.obj_translational_dim].float().to(self.device) + torch.tensor(self.object_asset_pos).float().to(self.device).unsqueeze(0)
        else:
            obj_pos = torch.tensor(self.object_asset_pos).float().to(self.device)
        obj_mat = tf.euler_angles_to_matrix(obj_ori_3d, convention='XYZ')
        obj_quat = tf.matrix_to_quaternion(obj_mat)
        obj_trans = tf.Transform3d(pos=obj_pos, 
                                    rot=obj_quat.float().to(self.device), device=self.device)

        # desired_ee_world_frame = self.contact_obj_frame[finger_name]
        desired_ee_world_frame = obj_trans.compose(desired_ee_obj_frame)
        desired_ee_robot_frame = self.world_trans.inverse().compose(desired_ee_world_frame).get_matrix()
        desired_ee_pos = desired_ee_robot_frame[:, :3, 3]

        ee_pos = self.forward_kinematics(x)
        ee_pos = ee_pos[self.ee_names[finger_name]].get_matrix()
        ee_pos = ee_pos[:, :3, 3]
        error = torch.norm(ee_pos - desired_ee_pos, dim=1)

        return error

    @all_finger_constraints
    def _contact_point_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]

        contact_obj_frame = self.contact_obj_frame[finger_name]
        

        # compute constraint value
        g = self._contact_point_constr(x.reshape(-1, self.dx),
                                 contact_obj_frame,
                                 finger_name=finger_name).reshape(N, T, -1)
        dg = 1
        g = g.reshape(N, -1)

        # compute the gradient
        if compute_grads:
            dg_dx = self.grad_contact_point_constr(x.reshape(-1, self.dx),
                                 contact_obj_frame,
                                 finger_name)
            NTrange = torch.arange(N * T, device=self.device)
            dg_dx = dg_dx[NTrange, NTrange, :].reshape(N, T, dg, self.dx)


            grad_g = torch.zeros(N, dg, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            grad_g[:, :, T_range, T_range, :self.dx] = dg_dx.transpose(1, 2)
            grad_g = grad_g.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g, grad_g, hess_g

        return g, grad_g, None
    
    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_point_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess)
        if self.optimize_force:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints_w_force(
                xu=xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)
        else:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints(
                xu=xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)
        
        if verbose:
            print(f"max contact constraint: {torch.max(torch.abs(g_contact))}")
            # print(f"max dynamics constraint: {torch.max(torch.abs(g_dynamics))}")
            print(f"max force equilibrium constraint: {torch.max(torch.abs(g_equil))}")
            result_dict = {}
            result_dict['contact'] = torch.max(torch.abs(g_contact)).item()
            result_dict['force'] = torch.max(torch.abs(g_equil)).item()
            result_dict['contact_mean'] = torch.mean(torch.abs(g_contact)).item()
            result_dict['force_mean'] = torch.mean(torch.abs(g_equil)).item()

            return result_dict
        g_contact = torch.cat((
                                g_contact, 
                            #    g_dynamics,
                               g_equil,
                            #    g_friction,
                               ), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((
                                        grad_g_contact, 
                                        # grad_g_dynamics,
                                        grad_g_equil,
                                        # grad_g_friction,
                                        ), dim=1)
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((
                                        hess_g_contact, 
                                        # hess_g_dynamics,
                                        hess_g_equil,
                                        # hess_g_friction,
                                        ), dim=1)

        return g_contact, grad_g_contact, hess_g_contact
    
  
    
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the screwdriver once"
    screwdriver_goal = params['screwdriver_goal'].cpu()
    screwdriver_goal_mat = R.from_euler('XYZ', screwdriver_goal).as_matrix()
    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state.reshape(4 * num_fingers + 4).to(device=params['device'])
    # start = torch.cat((state.reshape(10), torch.zeros(1).to(state.device))).to(device=params['device'])
    contact_fingers = params['fingers']
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
        fixed_obj=True,
    )

    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
    pregrasp_planner.warmup_iters = 50
    
    
    start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])

    if params['visualize_plan']:
        traj_for_viz = best_traj[:, :pregrasp_problem.dx]
        tmp = start[4 * num_fingers:].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)    
        viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, pregrasp_problem.fingers, pregrasp_problem.obj_dof+1)


    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        if params['mode'] == 'hardware':
            set_state = env.get_state()['all_state'].to(device=env.device)
            set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
            sim_viz_env.set_pose(set_state)
            sim_viz_env.step(action)
        env.step(action)
        action_list.append(action)
        if params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, 4 * num_fingers)[0], params['fingers']))

    state = env.get_state()
    start = state.reshape(4 * num_fingers + 4).to(device=params['device'])
    if params['exclude_index']:
            turn_problem_fingers = copy.copy(params['fingers'])
            turn_problem_fingers.remove('index')
            turn_problem_start = start[4:4 * num_fingers + obj_dof]
    else:
        turn_problem_fingers = params['fingers']
        turn_problem_start = start[:4 * num_fingers + obj_dof]

    fk_dict = forward_kinematics(partial_to_full_state(start[:4*num_fingers], fingers=params['fingers']))
    fks = [fk_dict[finger] for finger in fk_dict.keys()]
    fk_world = [env.world_trans.to(fk.device).compose(fk) for fk in fks]
    screwdriver_state = start[-(obj_dof+1):-1]
    screwdriver_quat = R.from_euler('XYZ', screwdriver_state.detach().cpu().numpy()).as_quat()
    screwdriver_trans = tf.Transform3d(pos=torch.tensor(env.table_pose).float().to(params['device']), 
                                rot=torch.tensor(
                                    [screwdriver_quat[3], screwdriver_quat[0], screwdriver_quat[1], screwdriver_quat[2]],
                                    device=params['device']).float(), device=params['device'])
    fk_screwdriver_frame = [screwdriver_trans.inverse().compose(fk) for fk in fk_world]
    turn_problem = AblationAllegroValveTurning(
        start=turn_problem_start,
        goal=params['screwdriver_goal'],
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.table_pose,
        contact_obj_frame=fk_screwdriver_frame,
        object_location=params['object_location'],
        object_type=params['object_type'],
        friction_coefficient=params['friction_coefficient'],
        world_trans=env.world_trans,
        fingers=turn_problem_fingers,
        optimize_force=params['optimize_force'],
        force_balance=False,
    )
    turn_problem.ee_names = ee_names
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
    validity_flag = True
    warmup_time = 0

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state.reshape(4 * num_fingers + 4).to(device=params['device'])

        actual_trajectory.append(state[:, :4 * num_fingers + obj_dof].squeeze(0).clone())
        start_time = time.time()
        best_traj, trajectories = turn_planner.step(start[:4 * num_fingers + obj_dof])
        
        #debug only
        # turn_problem.save_history(f'{fpath.resolve()}/op_traj.pkl')
        solve_time = time.time() - start_time
        print(f"solve time: {solve_time}")
        if k == 0:
            warmup_time = solve_time
        else:
            duration += solve_time
        planned_theta_traj = best_traj[:, 4 * num_fingers_to_plan: 4 * num_fingers_to_plan + obj_dof].detach().cpu().numpy()
        print(f"current theta: {state[0, -(obj_dof+1): -1].detach().cpu().numpy()}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        # if k < params['num_steps'] - 1:
        #     if params['mode'] == 'hardware':
        #         pass # debug TODO: fix it
        #         # add_trajectories_hardware(trajectories, best_traj, axes, env, config=params, state2ee_pos_func=state2ee_pos)
        #     else:
        #         add_trajectories(trajectories, best_traj, axes, env, sim=sim, gym=gym, viewer=viewer,
        #                         config=params, state2ee_pos_func=state2ee_pos)

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
            visualize_trajectory(traj_for_viz, turn_problem.viz_contact_scenes, viz_fpath, turn_problem.fingers, turn_problem.obj_dof+1)
        
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        if params['optimize_force']:
            print("planned force")
            print(action[:, 4 * num_fingers_to_plan:].reshape(num_fingers_to_plan + 1, 3)) # print out the action for debugging
            print("delta action")
            print(action[:, :4 * num_fingers_to_plan].reshape(num_fingers_to_plan, 4))
        # print(action)
        action = action[:, :4 * num_fingers_to_plan]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers].to(action.device) # NOTE: this is required since we define action as delta action
        if params['mode'] == 'hardware':
            set_state = env.get_state()['all_state'].to(device=env.device)
            set_state = torch.cat((set_state, torch.zeros(1).float().to(env.device)), dim=0)
            sim_viz_env.set_pose(set_state)
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        env.step(action)
        action_list.append(action)
        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        screwdriver_state = env.get_state()['q'][:, -obj_dof-1: -1].cpu()
        screwdriver_mat = R.from_euler('XYZ', screwdriver_state).as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
            torch.tensor(screwdriver_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()
        
        screwdriver_top_pos = get_screwdriver_top_in_world(screwdriver_state[0], turn_problem.object_chain, env.world_trans, turn_problem.object_asset_pos)
        screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
        distance2nominal = np.linalg.norm(screwdriver_top_pos - nominal_screwdriver_top)
        if distance2nominal > 0.02:
            validity_flag = False
        # distance2goal = (screwdriver_goal - screwdriver_state)).detach().cpu()
        print(distance2goal, validity_flag)

        # distance2goal = (screwdriver_goal - screwdriver_state)).detach().cpu()
        print(distance2goal)
        info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal, 'validity_flag': validity_flag, 'distance2nominal': distance2nominal}}
        info_list.append(info)

        gym.clear_lines(viewer)
        state = env.get_state()
        start = state[:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
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
    # action_list = torch.concat(action_list, dim=0)
    # with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
    #     pkl.dump(action_list, f)
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



    state = env.get_state()
    state = state.reshape(4 * num_fingers + obj_dof + 1)
    actual_trajectory.append(state.clone()[:4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    screwdriver_state = actual_trajectory[:, -obj_dof:].cpu()
    screwdriver_mat = R.from_euler('XYZ', screwdriver_state).as_matrix()
    distance2goal = tf.so3_relative_angle(torch.tensor(screwdriver_mat), \
        torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0],1,1), cos_angle=False).detach().cpu()

    # final_distance_to_goal = torch.min(distance2goal.abs())
    final_distance_to_goal = distance2goal.abs()[-1]

    print(f'Controller: {params["controller"]} Final distance to goal: {final_distance_to_goal}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal)
    env.reset()
    ret = {'warmup_time': warmup_time,
    'final_distance_to_goal': final_distance_to_goal, 
    'validity_flag': validity_flag,
    'avg_online_time': duration / (params["num_steps"] - 1)}
    return ret

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/ablation/config/allegro_screwdriver.yaml').read_text())
    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
        from hardware.hardware_env import HardwareEnv
        # TODO, think about how to read that in simulator
        # default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
        #                             torch.tensor([[0., 0.5, 0.7, 0.7]]).float(),
        #                             torch.tensor([[0., 0.5, 0.0, 0.7]]).float(),
        #                             torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float()),
        #                             dim=1)
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
                          gradual_control=True,
                          num_repeat=10)
        root_coor, root_ori = env.obj_reader.get_state()
        root_coor = root_coor / 1000 # convert to meters
        # robot_p = np.array([-0.025, -0.1, 1.33])
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
                                 table_pose=root_coor,
                                 )
        sim, gym, viewer = sim_env.get_sim()
        assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.table_pose = sim_env.table_pose
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
                                    gradual_control=True,
                                    )
        sim, gym, viewer = env.get_sim()
    if config['mode'] == 'hardware_copy':
        from hardware.hardware_env import RosNode
        ros_copy_node = RosNode()
        

    

    


    state = env.get_state()
    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass

    results = {}
    validity_list = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    # ee_names = {
    #         'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
    #         'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
    #         'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
    #         'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
    #         }
    ee_names = {
            'index': 'hitosashi_ee',
            'middle': 'naka_ee',
            'ring': 'kusuri_ee',
            'thumb': 'oya_ee',
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
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])

    for controller in config['controllers'].keys():
        results[controller] = {}
        results[controller]['warmup_time'] = []
        results[controller]['dist2goal'] = []
        results[controller]['validity_flag'] = []
        results[controller]['avg_online_time'] = []
    for i in tqdm(range(config['num_trials'])):
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
            ret = do_trial(env, params, fpath, sim_env, ros_copy_node)
            results[controller]['warmup_time'].append(ret['warmup_time'])
            results[controller]['dist2goal'].append(ret['final_distance_to_goal'])
            results[controller]['validity_flag'].append(ret['validity_flag'])
            results[controller]['avg_online_time'].append(ret['avg_online_time'])
        print(results)

    for key in results[controller].keys():
        print(f"{controller} {key}: avg: {np.array(results[controller][key]).mean()}, std: {np.array(results[controller][key]).std()}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


