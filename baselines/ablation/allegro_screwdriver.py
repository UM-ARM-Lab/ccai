import numpy as np

import torch

from utils.allegro_utils import *
from baselines.ablation.allegro_valve_turning import AblationAllegroValveTurning
from scipy.spatial.transform import Rotation as R


class AblationAllegroScrewdriver(AblationAllegroValveTurning):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        if self.screwdriver_force_balance:
            wrench_dim += 2
        self.dg_per_t = self.num_fingers * (1 + 4) + wrench_dim
        self.dg_constant = 0
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = (self.friction_polytope_k) * self.num_fingers + 1 # one friction constraints per finger
        self.dz += self.num_fingers # min force constraint
        # self.dz = 0 # DEBUG ONLY
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
                 force_balance=False,
                 device='cuda:0', 
                 obj_gravity=False,
                 **kwargs):
        self.num_fingers = len(fingers)
        self.obj_dof_code = [0, 0, 0, 1, 1, 1]
        self.optimize_force = optimize_force
        self.obj_mass = 0.1
        du = (4 + 3) * self.num_fingers + 3 
        super(AblationAllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=self.obj_dof_code, 
                                                 obj_joint_dim=1, optimize_force=optimize_force, 
                                                 screwdriver_force_balance=force_balance,
                                                 collision_checking=False, obj_gravity=obj_gravity,
                                                 contact_region=False, du=du, contact_obj_frame=contact_obj_frame, device=device)
        self.env_force = True
        max_f = torch.ones(3) * 10 # add environment force
        min_f = torch.ones(3) * -10
        self.x_max = torch.cat((self.x_max, max_f))
        self.x_min = torch.cat((self.x_min, min_f))
        self.nominal_screwdriver_top = np.array([0, 0, 1.405]) # in the world frame

    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
        force = 1.5 * torch.randn(N, self.T, 3 * self.num_fingers + 3, device=self.device)
        force[:, :, :3] = force[:, :, :3] * 0.01 # NOTE: scale down the index finger force, might not apply to situations other than screwdriver
        force[:, :, -3:] = force[:, :, -3:] * 0.01
        u = torch.cat((u, force), dim=-1)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :4 * self.num_fingers] + u[:, t, :4 * self.num_fingers]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (4 * self.num_fingers + self.obj_dof):
            if self.obj_dof == 6:
                current_obj_position = self.start[4 * self.num_fingers: 4 * self.num_fingers + self.obj_translational_dim]
                current_obj_orientation = self.start[4 * self.num_fingers + self.obj_translational_dim:4 * self.num_fingers + self.obj_dof]
                current_obj_R = R.from_euler('XYZ', current_obj_orientation.cpu().numpy())
                goal_obj_R = R.from_euler('XYZ', self.goal[self.obj_translational_dim:self.obj_translational_dim + self.obj_rotational_dim].cpu().numpy())
                key_times = [0, self.T]
                times = np.linspace(0, self.T, self.T + 1)
                slerp = Slerp(key_times, R.concatenate([current_obj_R, goal_obj_R]))
                interp_rots = slerp(times)
                interp_rots = interp_rots.as_euler('XYZ')[1:]
                # current_obj_orientation = tf.euler_angles_to_matrix(current_obj_orientation, convention='XYZ')
                # current_obj_orientation = tf.matrix_to_rotation_6d(current_obj_orientation)

                theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
                theta = np.concatenate((theta_position, interp_rots), axis=-1)
                theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
            else:
                theta = np.linspace(self.start[-self.obj_dof:].cpu().numpy(), self.goal.cpu().numpy(), self.T + 1)[1:]
                theta = torch.tensor(theta, device=self.device, dtype=torch.float32)

                # repeat the current state
                # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((self.T, 1))
            theta = theta.unsqueeze(0).repeat((N,1,1))

            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        action_cost = 0
        # action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        smoothness_cost += 50 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 10000 * torch.sum((state[:, -self.obj_dof:-1]) ** 2) # the screwdriver should only rotate in z direction

        goal_cost = torch.sum((500 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # add a running cost
        goal_cost += torch.sum((1 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))

        return smoothness_cost + action_cost + goal_cost + upright_cost

    def _force_equlibrium_constraints_w_force(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        q = x[:, :-1, :self.num_fingers * 4]
        next_q = x[:, 1:, :self.num_fingers * 4]
        next_env_q = x[:, 1:, self.num_fingers * 4:self.num_fingers * 4 + self.obj_dof]
        u = xu[:, :, self.dx: self.dx + 4 * self.num_fingers]
        force = xu[:, :, self.dx + 4 * self.num_fingers: self.dx + 4 * self.num_fingers + 3 * self.num_fingers + 3]
        force_list = force.reshape((force.shape[0], force.shape[1], self.num_fingers + 1, 3))
        # contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1].reshape(-1, 3, 4 * self.num_fingers)\
        #                      for finger_name in self.fingers]
        contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, 1:].reshape(-1, 3, 4 * self.num_fingers)\
                             for finger_name in self.fingers]
        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=xu.device)
        contact_point_list = [self.data[finger_name]['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3) for finger_name in self.fingers]
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=xu.device)

        g = self.force_equlibrium_constr(q.reshape(-1, 4 * self.num_fingers), 
                                         u.reshape(-1, 4 * self.num_fingers), 
                                         next_q.reshape(-1, 4 * self.num_fingers), 
                                         force_list.reshape(-1, self.num_fingers + 1, 3),
                                         contact_jac_list,
                                         contact_point_list,
                                         next_env_q.reshape(-1, self.obj_dof)).reshape(N, T, -1)
        # print(g.abs().max().detach().cpu().item(), g.abs().mean().detach().cpu().item())
        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact, dg_dnext_env_q = self.grad_force_equlibrium_constr(q.reshape(-1, 4 * self.num_fingers), 
                                                                                  u.reshape(-1, 4 * self.num_fingers), 
                                                                                  next_q.reshape(-1, 4 * self.num_fingers), 
                                                                                  force_list.reshape(-1, self.num_fingers + 1, 3),
                                                                                  contact_jac_list,
                                                                                  contact_point_list,
                                                                                  next_env_q.reshape(-1, self.obj_dof))
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], self.num_fingers * 3 + 3)
            
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            dg_dq = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_fingers) 
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], 4 * self.num_fingers) 
            for i, finger_name in enumerate(self.fingers):
                # NOTE: assume fingers have joints independent of each other
                # djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, :-1] # jacobian is the contact jacobian
                # dg_dq = dg_dq + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, 1:]
                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dnext_q.reshape(N, T, -1, 4 * self.num_fingers)
                
                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1]
                dg_dq = dg_dq + dg_dcontact[:, : ,i].reshape(N, T, g.shape[2], 3) @ d_contact_loc_dq 
            grad_g[:, :, T_plus, T_minus, :4 * self.num_fingers] = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_fingers)[:, 1:].transpose(1, 2)  # first q is the start
            dg_du = torch.cat((dg_du, dg_dforce), dim=-1)  # check the dim
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_range, T_range, :4 * self.num_fingers] = dg_dnext_q.reshape(N, T, -1, 4 * self.num_fingers).transpose(1, 2)
            if self.obj_gravity:
                grad_g[:, :, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dg_dnext_env_q.reshape(N, T, -1, self.obj_dof).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None
        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), hess
        else:
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), None
    
    def _env_force_constraint(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]
        u = xu[:, :, self.dx:]
        env_force = xu[:, :, -3:]
        env_force_z = env_force[:, :, 2]
        h = -env_force_z
        if compute_grads:
            grad_h = torch.zeros(N, 1, T, T, d, device=self.device)
            grad_h[:, :, torch.arange(T), torch.arange(T), -3:] = -1
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None
    
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
    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        h, grad_h, hess_h = self._friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        # h_vel, grad_h_vel, hess_h_vel = self._friction_vel_constraint(
        #     xu=xu.reshape(-1, T, self.dx + self.du),
        #     compute_grads=compute_grads,
        #     compute_hess=compute_hess)
        h_env, grad_h_env, hess_h_env = self._env_force_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        h_force, grad_h_force, hess_h_force = self._min_force_constraints(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            # print(f"max step size constraint: {torch.max(h_step_size)}")
            # print(f"max singularity constraint: {torch.max(h_sin)}")
            print(f"max env force constraint: {torch.max(h_env)}")
            print(f"max min force constraint: {torch.max(h_force)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            result_dict['friction_mean'] = torch.mean(h).item()
            result_dict['env_force'] = torch.max(h_env).item()
            result_dict['env_force_mean'] = torch.mean(h_env).item()
            result_dict['min_force'] = torch.max(h_force).item()
            # result_dict['singularity'] = torch.max(h_sin).item()
            return result_dict

        # h = torch.cat((h,
        #             #    h_step_size,
        #                h_sin), dim=1)
        h = torch.cat((h, h_env, h_force), dim=1)
        if compute_grads:
            grad_h = grad_h.reshape(N, -1, self.T * (self.dx + self.du))
            grad_h = torch.cat((grad_h, grad_h_env, grad_h_force), dim=1)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None
    
    def check_validity(self, state):
        screwdriver_state = state[-self.obj_dof:]
        screwdriver_top_pos = get_screwdriver_top_in_world(screwdriver_state, self.object_chain, self.world_trans, self.object_asset_pos)
        screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
        distance2nominal = np.linalg.norm(screwdriver_top_pos - self.nominal_screwdriver_top)
        if distance2nominal > 0.02:
            validity_flag = False
        else:
            validity_flag = True
        return validity_flag
  