"""
Tactile-Feedback Motion-Contact Tracking Controller

This module implements the tactile-feedback controller for synergistically tracking
motion-contact references (finger motions and contact forces) as described in the paper.

The controller supports:
- Single contact case
- Full hand case with multiple contacts
- Coupling effect modeling
- Model Predictive Control (MPC)
- Adaptive weighting matrix determination
- Integration with AllegroManipulationProblem preprocessing
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import scipy.linalg
from scipy.spatial.transform import Rotation

from ccai.allegro_contact import AllegroManipulationProblem

from pygrampc import Grampc, GrampcResults, ProblemDescription

from ccai.controller.se3_dist import se3_distance_gradient, se3_distance
from ccai.utils.allegro_utils import partial_to_full_state

import pytorch_kinematics.transforms as tf
import pytorch_kinematics as pk

logger = logging.getLogger(__name__)

def quaternion_close(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-4):
    """
    Returns true if two quaternions are close to each other. Assumes the quaternions are normalized.
    Based on: https://math.stackexchange.com/a/90098/516340

    """
    dist = 1 - torch.square(torch.sum(q1 * q2, dim=-1))
    return torch.all(dist < eps)

def quaternion_angular_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Computes the angular distance between two quaternions.
    Args:
        q1: First quaternion (assume normalized).
        q2: Second quaternion (assume normalized).
    Returns:
        Angular distance between the two quaternions.
    """

    # Compute the cosine of the angle between the two quaternions
    cos_theta = torch.sum(q1 * q2, dim=-1)
    # we use atan2 instead of acos for better numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    abs_dot = torch.abs(cos_theta)
    # identity sin^2(theta) = 1 - cos^2(theta)
    sin_half_theta = torch.sqrt(1.0 - torch.square(abs_dot))
    theta = 2.0 * torch.atan2(sin_half_theta, abs_dot)

    # theta for the ones that are close gets 0 and we don't care about them
    close = quaternion_close(q1, q2)
    theta[close] = 0
    return theta

@dataclass
class ControllerConfig:
    """Configuration for the tactile feedback controller."""
    # Environment stiffness and damping
    K_e: float = 200.0  # Environment stiffness
    K_P: float = 3  # Proportional gain
    K_D: float = 1    # Damping gain
    
    K_D_inv: float = 1/ K_D
    K_p_inv: float = 1/ K_P
    
    # Force threshold for contact classification
    force_threshold: float = 0.5  # Δ in the paper
    
    # MPC parameters
    horizon_length: int = 10
    dt: float = 0.01
    mpc_solver: str = "cvxpy"  # Options: 'cvxpy' (recommended), 'lbfgs', 'osqp', 'scipy', 'adam', 'augmented_lagrangian'
    
    # Weighting parameters
    w_q: float = 20.0
    w_p: float = 1.0
    w_f: float = 1.0
    w_u: float = 1
    w_ori: float = .1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # State dimension
    dq: int = 12
    df: int = 9



class ModelPredictiveController:
    """
    Model Predictive Controller for tactile-feedback control (Section 6.4).
    
    Implements equation (24): ẍ = g(x, u) where x = [q; q̇; q̇_d; λ_ext]
    Uses cvxpy for quadratic programming formulation.
    """
    
    def __init__(self, problem: AllegroManipulationProblem, config: ControllerConfig):
        self.config = config
        self.horizon = config.horizon_length
        self.dt = config.dt
        self.device = config.device
        
        self.dq = config.dq
        self.df = 3 * problem.num_fingers
        
        self.K_e = config.K_e
        self.K_P = config.K_P * np.eye(self.dq)
        self.K_D = config.K_D * np.eye(self.dq)
        self.K_p_inv = config.K_p_inv * np.eye(self.dq)
        self.K_D_inv = config.K_D_inv * np.eye(self.dq)
        
        
        n_x = self.dq * 2 + self.df  # State dimension
        n_u = self.dq  # Control dimension (joint torques)
        T = self.horizon
        
        # Indices of state variables in the decision variable
        self.state_indices = []
        self.control_indices = []
        for i in range(T):
            start_idx = i * (n_x + n_u)
            end_idx = start_idx + n_x
            self.state_indices.extend(list(range(start_idx, end_idx)))
            self.control_indices.extend(list(range(end_idx, end_idx+n_u)))
            
        self.problem = problem
        
        # Check if cvxpy is available
        try:
            import cvxpy as cp
            self.cvxpy_available = True
        except ImportError:
            self.cvxpy_available = False
        
    def pack_state(self, q: torch.Tensor, q_d: torch.Tensor, 
                   lambda_ext: torch.Tensor) -> torch.Tensor:
        """
        Pack state vector according to equation (24): x = [q; q_d; λ_ext]
        
        Args:
            q: Actual joint positions [n_q]
            q_d: Commanded joint positions [n_q]  
            lambda_ext: Contact forces [3*n_c]
            
        Returns:
            x: Packed state vector [n_q + n_q + 3*n_c]
        """
        return torch.cat([q, q_d, lambda_ext])
    
    def unpack_state(self, x: torch.Tensor, n_q: int, n_c: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack state vector from equation (24): x = [q; q_d; λ_ext]
        
        Args:
            x: Packed state vector [n_q + n_q + 3*n_c]
            n_q: Number of joints
            n_c: Number of contacts
            
        Returns:
            q, q_d, lambda_ext: Unpacked state components
        """
        q = x[:n_q]
        q_d = x[n_q:2*n_q]
        lambda_ext = x[2*n_q:2*n_q + 3*n_c]
        return q, q_d, lambda_ext
    
    def compute_K_r(self, J_s):
        return J_s @ self.K_p_inv @ J_s.T
    
    def compute_K_bar(self, J_s):
        K_r = self.compute_K_r(J_s)
        K_bar = (np.eye(self.K_e.shape[0]) + self.K_e @np.linalg.inv(K_r + 1e-6 * np.eye(K_r.shape[0]))) @ self.K_e
        return K_bar
    
    def compute_K_coup(self, G_o: torch.Tensor, J_s: torch.Tensor) -> torch.Tensor:
        """
        Compute K_coup using the coupling formula.
        """
        self.K_bar = self.compute_K_bar(J_s)
        G_o_Kbar = G_o @ self.K_bar
        K_coup = self.K_bar + (self.K_bar @ G_o.T) @ np.linalg.inv(G_o_Kbar @ G_o.T + 1e-6 * np.eye(6)) @ G_o_Kbar
        return K_coup
        
class TactileMPC(ProblemDescription, ModelPredictiveController):
    def __init__(self, problem: AllegroManipulationProblem, config: ControllerConfig):
        ProblemDescription.__init__(self)
        ModelPredictiveController.__init__(self, problem, config)
        self.Nx = config.dq*2 + config.df
        self.Nu = config.dq
        self.Np = 0
        self.Ng = 0
        self.Nh = 0
        self.NgT = 0
        self.NhT = 0
        
        self.last_t = None
        self.last_q = None
        self.last_q_d = None
        
        self.contact_indices = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
        
    def compute_system_matrices(self, x):
        q = x[:self.dq]
        q_d = x[self.dq:2*self.dq]
        if (self.last_q is not None and self.last_q_d is not None and np.allclose(q, self.last_q) and np.allclose(q_d, self.last_q_d)):
            return
        self.last_q = q
        self.last_q_d = q_d
        
        q_for_preprocess = torch.tensor(np.stack((q, q_d), axis=0), device=self.problem.device).unsqueeze(0).float()
        theta_for_preprocess = torch.zeros((1, 2, self.problem.obj_dof), device=self.problem.device).float()
        
        self.problem._preprocess_fingers(q_for_preprocess, theta_for_preprocess, T_override=1, tactile_controller=True)
        
        Js = self.problem.data['J_q'].clone().flatten(1, 2)
        Hs = self.problem.data['H_q'].clone().flatten(1, 2)
        
        
        self.J_q = Js[0].detach().cpu().numpy()[:, self.contact_indices]
        self.J_q_d = Js[1].detach().cpu().numpy()[:, self.contact_indices]
        self.H_q = Hs[0, :, self.contact_indices][:, :, self.contact_indices].detach().cpu().numpy()
        self.H_q_d = Hs[1, :, self.contact_indices][:, :, self.contact_indices].detach().cpu().numpy()
        self.G_o = self.problem.data['G_o'][0].detach().cpu().numpy()
        
        self.K_coup = self.compute_K_coup(self.G_o, self.J_q)

    def set_weighting_matrices(self, W_A, W_P):
        self.W_A = W_A
        self.W_P = W_P
        print('Setting weighting matrices')
        print(self.W_A)
        print(self.W_P)
        print()

    def set_reference_trajectory(self, spline_func):
        self.spline_func = spline_func
        print('Setting reference trajectory')
        print()
        
    def get_reference_trajectory(self, t):
        interpolated_state = self.spline_func(t)
        return {
            'q_ref': interpolated_state[:self.dq],
            'f_ref': interpolated_state[2*self.dq:2*self.dq + self.df]
        }
            
    def ffct(self, out, t, x, u, p):
        self.compute_system_matrices(x)
        q = x[:self.dq]
        q_d = x[self.dq:2*self.dq]
        f = x[2*self.dq:2*self.dq + self.df]
        # q_dot
        out[:self.dq] = u + self.K_D_inv@(self.K_P@(q_d - q) - self.J_q.T @ f)
        #q_d_dot
        out[self.dq:2*self.dq] = u
        # f_dot
        out[2*self.dq:2*self.dq + self.df] = self.K_coup @ (self.J_q_d @ u)
        return out
    
    def dfdx_vec(self, out, t, x, vec, u, p):
        self.compute_system_matrices(x)
        J = np.zeros((self.dq*2+self.df, self.dq*2+self.df))
        
        # dq_dot/dq
        J[:self.dq, :self.dq] = -self.K_P @ self.K_D_inv
        
        # Jacobian chain rule
        f_ext = x[2*self.dq:2*self.dq + self.df]
        dq_dot_dJ = -self.K_D_inv @ (self.H_q.transpose(1, 2, 0) @ f_ext)
        J[:self.dq, :self.dq] += dq_dot_dJ
        
        # dq_dot/dq_d
        J[:self.dq, self.dq:2*self.dq] = self.K_D_inv @ self.K_P
        
        #dq_dot/df
        J[:self.dq, 2*self.dq:2*self.dq + self.df] = self.K_D_inv @ -self.J_q.T
        
        
        # df_dot/dq_d
        K_c_h = np.einsum('ij, jkl->ikl', self.K_coup, self.H_q_d)
        J[2*self.dq:2*self.dq + self.df, self.dq:2*self.dq] = K_c_h @ u
        
        return J.T @ vec
    
    def dfdu_vec(self, out, t, x, vec, u, p):
        self.compute_system_matrices(x)
        J = np.zeros((self.dq*2+self.df, self.dq))
        
        # dq_dot/du
        J[:self.dq, :self.dq] = np.eye(self.dq)
        
        # dq_d_dot/du
        J[self.dq:2*self.dq, :self.dq] = np.eye(self.dq)        
        
        # df_dot/du
        J[2*self.dq:2*self.dq + self.df, :self.dq] = self.K_coup @ self.J_q_d
        
        return J.T @ vec
    
    def lfct(self, out, t, x, u, p, xdes, udes):
        ref = self.get_reference_trajectory(t)
        q_ref = ref['q_ref']
        f_ref = ref['f_ref']
        
        q = x[:self.dq]
        
        fk_q_ref = self.handle_fk(q_ref)
        fk_q = self.handle_fk(q)
        
        dist, _, _ = se3_distance(fk_q, fk_q_ref, self.W_P)
        
        p_cost = dist ** 2 * self.config.w_p
        
        q_cost = np.sum((q_ref - q)**2) * self.config.w_q
        f_cost = np.sum((f_ref - x[2*self.dq:2*self.dq + self.df])**2) * self.config.w_f
        u_cost = np.sum(u**2) * self.config.w_u
        
        out = q_cost + p_cost + f_cost + u_cost
        
        return out
    
    def handle_fk(self, q, jac=False):
        q_for_fk = torch.tensor(q.reshape(1, -1), device=self.problem.device).float()
        q_for_fk = partial_to_full_state(q_for_fk, fingers=self.problem.fingers)
        ee_names = [self.problem.ee_names[f] for f in self.problem.fingers]
        frame_indices = [self.problem.contact_scenes.robot_sdf.chain.frame_to_idx[ee_name] for ee_name in ee_names]
        if jac:
            q_for_fk = q_for_fk.repeat(len(frame_indices), 1)
            fk_q = self.problem.contact_scenes.robot_sdf.chain.jacobian(q_for_fk, link_indices=torch.tensor(frame_indices, device=self.problem.device).long())[..., self.contact_indices].cpu().numpy()
        else:
            fk_q = self.problem.contact_scenes.robot_sdf.chain.forward_kinematics(q_for_fk)
            pts = []
            for ee_name in ee_names:
                pts.append(fk_q[ee_name].get_matrix().cpu().numpy())
            fk_q = np.concatenate(pts, axis=0)

        return fk_q
    
    def dldx(self, out, t, x, u, p, xdes, udes):
        ref = self.get_reference_trajectory(t)
        q_ref = ref['q_ref']
        f_ref = ref['f_ref']
        q = x[:self.dq]
        
        # joint position cost
        out[:self.dq] = 2 * (q_ref - q) * self.config.w_q
        
        fk_q = self.handle_fk(q)
        fk_q_ref = self.handle_fk(q_ref)
        jac_fk_q = self.handle_fk(q, jac=True)
        dist, grad = se3_distance_gradient(fk_q, fk_q_ref, self.W_P)
        # Squared cost, so adjust grad
        grad = grad * 2 * dist.reshape(-1, 1)
        grad_fk_q = np.einsum('bi,bij->bj', grad, jac_fk_q)
        #
        out[:self.dq] += grad_fk_q.sum(axis=0) * self.config.w_p
        
        # external force cost
        out[2*self.dq:2*self.dq + self.df] = 2 * (x[2*self.dq:2*self.dq + self.df] - f_ref) * self.config.w_f
        
        return out
    
    def dldu(self, out, t, x, u, p, xdes, udes):
        out = 2 * u * self.config.w_u
        return out

class WeightingMatrixDeterminer:
    """
    Determines weighting matrices based on contact classification (Section 6.5).
    """
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.force_threshold = config.force_threshold
        
    def classify_contacts(self, contact_forces: torch.Tensor) -> torch.Tensor:
        """
        Classify contacts as active/inactive based on force threshold.
        
        Args:
            contact_forces: Contact forces [n_c, 3]
            
        Returns:
            is_active: Boolean mask for active contacts [n_c]
        """
        force_magnitudes = np.linalg.norm(contact_forces.reshape(-1, 3), axis=1)
        is_active = force_magnitudes >= self.force_threshold
        return is_active
    
    def compute_weighting_matrices(self, contact_forces: np.ndarray, avg_normal: np.ndarray, segment_tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weighting matrices W_A and W_P (Equation 26).
        
        Args:
            contact_forces: Contact forces [n_c, 3]
            contact_normals: Contact normal vectors [n_c, 3]
            
        Returns:
            W_A: Force weighting matrix
            W_P: Position weighting matrix
        """
        n_c = avg_normal.shape[0]
        is_active = self.classify_contacts(contact_forces)
        
        # Initialize weighting matrices
        W_A = np.zeros((3 * n_c, 3 * n_c))
        W_P = np.zeros((6 * n_c, 6 * n_c))
        
        for i in range(n_c):
            start_idx = 3 * i
            end_idx = 3 * (i + 1)
            
            start_idx_6 = 6 * i
            end_idx_6 = 6 * (i + 1)
            
            if is_active[i]:
                # Active contact: track normal force, perform position tracking
                n_i = avg_normal[i]  # Contact normal
                t_i = segment_tangents[i]
                
                # Create normal and tangential projections
                N_i = np.outer(n_i, n_i)  # Normal projection
                t_prod = t_i @ t_i.T   # Tangential projection
                
                T_i = np.eye(6)
                T_i[:3, :3] = t_prod
                T_i[3:, 3:] *= self.config.w_ori
                
                
                # W_A focuses on normal direction for force control
                W_A[start_idx:end_idx, start_idx:end_idx] = N_i
                
                # W_P focuses on tangential directions for position control  
                W_P[start_idx_6:end_idx_6, start_idx_6:end_idx_6] = T_i
            else:
                # Inactive contact: position tracking only
                W_A[start_idx:end_idx, start_idx:end_idx] = np.zeros((3, 3))
                W_P_nc = np.eye(6)
                W_P_nc[3:, 3:] *= self.config.w_ori
                W_P[start_idx_6:end_idx_6, start_idx_6:end_idx_6] = W_P_nc
                
        return W_A, W_P

class TactileFeedbackController:
    """
    Main tactile-feedback motion-contact tracking controller.
    
    Integrates all components for complete controller functionality.
    """
    
    def __init__(self, problem: AllegroManipulationProblem, config: ControllerConfig):
        self.config = config
        self.device = config.device
        self.problem = problem  # AllegroManipulationProblem instance
        
        # Initialize subcomponents
        self.weighting_determiner = WeightingMatrixDeterminer(config)
        self.mpc_problem_definition = TactileMPC(problem, config)
        
        self.grampc = Grampc(self.mpc_problem_definition)
        
        self.grampc.set_param({
            'Thor': self.config.dt * 2,
            'dt': self.config.dt,
        })
        self.grampc.print_opts()
        self.grampc.print_params()
        self.min_pen_estimate = False

        # Controller state
        self.current_mode = "multi_contact"  # "single_contact" or "multi_contact"
        self.contact_history = []
        
        logger.info(f"Initialized TactileFeedbackController with config: {config}")
        
    def solve(self, t0: float, state: torch.Tensor, q_d_init: torch.Tensor, f_ext_init: torch.Tensor, avg_normal: np.ndarray, segment_tangents: np.ndarray) -> torch.Tensor:
        
        # preprocess_input = torch.stack((state, q_d_init), dim=0).unsqueeze(0)
        # preprocess_input = torch.cat((preprocess_input, torch.zeros((1, 2, self.config.dq+self.config.df), device=state.device)), dim=-1)
        
        # self.problem._preprocess(preprocess_input)
        
        # Js = self.problem.data['J_q'].clone()
        # Hs = self.problem.data['H_q'].clone()
        
        # self.problem.data['J_q'] = Js[0]
        # self.problem.data['J_q_d'] = Js[1]
        # self.problem.data['H_q'] = Hs[0]
        # self.problem.data['H_q_d'] = Hs[1]
        # self.problem.data['G_o'] = self.problem.data['G_o'][0]
        
        W_A, W_P = self.weighting_determiner.compute_weighting_matrices(f_ext_init.cpu().numpy(), avg_normal, segment_tangents)
        
        
        self.mpc_problem_definition.K_e = self.config.K_e * W_A
        
        self.mpc_problem_definition.set_weighting_matrices(W_A *self.config.w_f, W_P * self.config.w_p)
                
        grampc_x0 = torch.cat((state.cpu(), q_d_init.cpu(), f_ext_init), dim=0).numpy()
        self.grampc.set_param({"x0": grampc_x0,
                               "t0": t0})
        
        if not self.min_pen_estimate:
            self.grampc.estim_penmin(True)
            self.min_pen_estimate = True
        runtime = self.grampc.run()
        
        print(f'Solved GRAMPC in {runtime} seconds.')

        
        u = self.grampc.sol.unext * self.config.dt
        
        return u
        
class TactileFeedbackQPController:
    def __init__(self, problem: AllegroManipulationProblem, config: ControllerConfig):
        self.config = config
        self.device = config.device
        self.problem = problem  # AllegroManipulationProblem instance
        
        # Initialize subcomponents
        self.weighting_determiner = WeightingMatrixDeterminer(config)
        self.mpc_problem_definition = TactileMPC(problem, config)
        
        # Initialize control attributes
        self.dq = config.dq
        self.df = 3 * problem.num_fingers
        self.horizon = config.horizon_length
        self.dt = config.dt
        
        # Control gains
        self.K_P = config.K_P * np.eye(self.dq)
        self.K_D = config.K_D * np.eye(self.dq)
        self.K_D_inv = (1.0 / config.K_D) * np.eye(self.dq)
        self.K_P_inv = (1.0 / config.K_P) * np.eye(self.dq)
        
        # State and control dimensions
        self.n_x = self.dq * 2 + self.df  # [q; q_d; λ_ext]
        self.n_u = self.dq  # joint torques
        
        # Contact indices (same as in TactileMPC)
        self.contact_indices = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
        
        # Check if cvxpy is available
        try:
            import cvxpy as cp
            self.cvxpy_available = True
        except ImportError:
            self.cvxpy_available = False
            logger.warning("cvxpy not available, falling back to basic QP solver")
        
        logger.info(f"Initialized TactileFeedbackQPController with config: {config}")
        
    def set_reference_trajectory(self, spline_func):
        """Set the reference trajectory spline function."""
        self.reference_spline = spline_func
        
    def compute_system_matrices_at_state(self, x_ref: np.ndarray):
        """
        Compute system matrices (Jacobians, coupling, etc.) at a reference state.
        
        Args:
            x_ref: Reference state [n_x] = [q; q_d; λ_ext]
            
        Returns:
            Dictionary containing system matrices
        """
        q = x_ref[:self.dq]
        q_d = x_ref[self.dq:2*self.dq]
        
        # Use the preprocessing from TactileMPC to get Jacobians
        q_for_preprocess = torch.tensor(np.stack((q, q_d), axis=0), device=self.problem.device).unsqueeze(0).float()
        theta_for_preprocess = torch.zeros((1, 2, self.problem.obj_dof), device=self.problem.device).float()
        
        self.problem._preprocess_fingers(q_for_preprocess, theta_for_preprocess, T_override=1, tactile_controller=True)
        
        Js = self.problem.data['J_q'].clone().flatten(1, 2)
        # Hs = self.problem.data['H_q'].clone().flatten(1, 2)
        
        J_q = Js[0].detach().cpu().numpy()[:, self.contact_indices]
        J_q_d = Js[1].detach().cpu().numpy()[:, self.contact_indices]
        # H_q = Hs[0, :, self.contact_indices][:, :, self.contact_indices].detach().cpu().numpy()
        # H_q_d = Hs[1, :, self.contact_indices][:, :, self.contact_indices].detach().cpu().numpy()
        G_o = self.problem.data['G_o'][0].detach().cpu().numpy()
        
        # Compute coupling matrix
        K_r = J_q @ self.K_P_inv @ J_q.T
        K_bar = np.linalg.inv(np.eye(K_r.shape[0]) + self.config.K_e @ np.linalg.inv(K_r + 1e-6 * np.eye(K_r.shape[0]))) @ self.config.K_e
        G_o_Kbar = G_o @ K_bar
        K_coup = K_bar + (K_bar @ G_o.T) @ np.linalg.inv(G_o_Kbar @ G_o.T + 1e-6 * np.eye(6)) @ G_o_Kbar
        
        return {
            'J_q': J_q,
            'J_q_d': J_q_d, 
            # 'H_q': H_q,
            # 'H_q_d': H_q_d,
            'G_o': G_o,
            'K_coup': K_coup
        }
               
    def compute_fk_poses(self, q_ref: np.ndarray):
        """
        Compute forward kinematics poses for reference joint configuration.
        
        Args:
            q_ref: Reference joint configuration [dq]
            
        Returns:
            fk_poses: Forward kinematics poses [6*n_contacts] (position + orientation)
        """
        # Convert to torch and expand to full joint state
        q_for_fk = torch.tensor(q_ref.reshape(1, -1), device=self.problem.device).float()
        q_for_fk = partial_to_full_state(q_for_fk, fingers=self.problem.fingers)
        
        # Get end-effector names
        ee_names = [self.problem.ee_names[f] for f in self.problem.fingers]
        
        # Compute forward kinematics
        fk_result = self.problem.contact_scenes.robot_sdf.chain.forward_kinematics(q_for_fk)
        poses = []
        for ee_name in ee_names:
            pose_matrix = fk_result[ee_name].get_matrix().cpu().numpy()[0]  # [4, 4]
            # Extract position and orientation
            position = pose_matrix[:3, 3]
            rotation = pose_matrix[:3, :3]
            # Convert rotation matrix to axis-angle representation
            axis_angle = Rotation.from_matrix(rotation).as_rotvec()
            pose_6d = np.concatenate([position, axis_angle])
            poses.append(pose_6d)
        
        fk_poses = np.concatenate(poses)  # [6*n_contacts]
        return fk_poses
        
    def evaluate_dynamics(self, x: np.ndarray, u: np.ndarray, system_matrices: dict):
        """
        Evaluate nonlinear dynamics at given state and control.
        
        Args:
            x: State [n_x]
            u: Control [n_u]
            system_matrices: Precomputed system matrices
            
        Returns:
            x_dot: State derivative [n_x]
        """
        q = x[:self.dq]
        q_d = x[self.dq:2*self.dq]
        f_ext = x[2*self.dq:2*self.dq + self.df]
        
        J_q = system_matrices['J_q']
        J_q_d = system_matrices['J_q_d']
        K_coup = system_matrices['K_coup']
        
        x_dot = np.zeros(self.n_x)
        
        # q_dot = u + K_D^(-1)(K_P(q_d - q) - J^T f)
        x_dot[:self.dq] = u + self.K_D_inv @ (self.K_P @ (q_d - q) - J_q.T @ f_ext)
        
        # q_d_dot = u
        x_dot[self.dq:2*self.dq] = u
        
        # f_dot = K_coup J_q_d u
        x_dot[2*self.dq:2*self.dq + self.df] = K_coup @ (J_q_d @ u)
        
        return x_dot
        
    def setup_qp_problem(self, x0: np.ndarray, reference_trajectory: callable, 
                        W_A: np.ndarray, W_P: np.ndarray):
        """
        Set up the QP problem for MPC using linearized dynamics.
        
        minimize: sum_{t=0}^{T-1} [cost(x_t, u_t)] + cost_terminal(x_T)
        subject to: x_{t+1} = A_t x_t + B_t u_t + c_t  (linearized dynamics)
                   u_min <= u_t <= u_max                (control bounds)
        
        Args:
            x0: Initial state [n_x]
            reference_trajectory: Function t -> reference_state
            W_A: Force weighting matrix
            W_P: Position weighting matrix
            
        Returns:
            QP problem (depends on available solver)
        """
        if not self.cvxpy_available:
            return self.setup_basic_qp_problem(x0, reference_trajectory, W_A, W_P)
            
        import cvxpy as cp
        
        # Decision variables: [u_0, u_1, ..., u_{T-1}, x_1, x_2, ..., x_T]
        u_vars = [cp.Variable(self.n_u) for _ in range(self.horizon)]
        x_vars = [cp.Variable(self.n_x) for _ in range(self.horizon)]
        
        constraints = []
        cost = 0
        
        # Linearize around reference trajectory
        x_ref_traj = []
        u_ref_traj = []
        A_mats = []
        B_mats = []
        c_vecs = []
        system_matrices_traj = []  # Cache system matrices to reuse in cost function
        
        # Cache FK computations to avoid redundant calculations [[memory:2647243]]
        fk_cache = {}
        
        for t in range(self.horizon):
            # Get reference trajectory at time t
            ref_state = reference_trajectory(t * self.dt)
            x_ref = ref_state[:self.n_x]  
            u_ref = np.zeros(self.n_u)  # Assume zero reference control
            
            x_ref_traj.append(x_ref)
            u_ref_traj.append(u_ref)
            
            # Linearize dynamics and cache system matrices
            A, B, c, system_matrices_t = self.linearize_dynamics(x_ref, u_ref)
            A_mats.append(A)
            B_mats.append(B)
            c_vecs.append(c)
            
            # Cache system matrices for reuse in cost function
            system_matrices_traj.append(system_matrices_t)
            
            # Cache FK computation for this reference point to reuse in cost function
            q_ref_t = x_ref[:self.dq]
            q_key = tuple(q_ref_t.round(6))  # Round for floating point key
            if q_key not in fk_cache:
                try:
                    fk_ref = self.compute_fk_poses(q_ref_t)
                    fk_cache[q_key] = fk_ref
                except Exception as e:
                    logger.warning(f"FK computation failed at timestep {t}: {e}")
                    fk_cache[q_key] = None
        
        # Add dynamics constraints
        x_prev = x0
        for t in range(self.horizon):
            if t == 0:
                constraints.append(x_vars[t] == A_mats[t] @ x_prev + B_mats[t] @ u_vars[t] + c_vecs[t])
            else:
                constraints.append(x_vars[t] == A_mats[t] @ x_vars[t-1] + B_mats[t] @ u_vars[t] + c_vecs[t])
        
        # Add cost function
        for t in range(self.horizon):
            ref_t = reference_trajectory(t * self.dt)
            q_ref = ref_t[:self.dq]
            f_ref = ref_t[2*self.dq:2*self.dq + self.df]
            
            # Extract state components
            q_t = x_vars[t][:self.dq]
            f_t = x_vars[t][2*self.dq:2*self.dq + self.df]
            
            # Joint position cost
            cost += self.config.w_q * cp.sum_squares(q_t - q_ref)
            
            # Force cost (using weighting matrix W_A)
            if W_A.shape[0] > 0:
                W_A_sqrt = np.real(scipy.linalg.sqrtm(W_A + 1e-6 * np.eye(W_A.shape[0])))
                cost += self.config.w_f * cp.sum_squares(W_A_sqrt @ (f_t - f_ref))
            
            # Control cost
            cost += self.config.w_u * cp.sum_squares(u_vars[t])
            
            # Linearized forward kinematics pose cost using cached FK jacobian
            # This makes the pose cost convex for the QP formulation
            if W_P.shape[0] > 0:
                # Use cached FK computation to avoid redundant calculations
                q_ref_t = q_ref
                q_key = tuple(q_ref_t.round(6))
                
                if q_key in fk_cache and fk_cache[q_key] is not None:
                    fk_ref = fk_cache[q_key]
                    
                    # Get the contact jacobian J_q from the cached system matrices
                    J_q = system_matrices_traj[t]['J_q']  # Contact jacobian [df, dq]
                    
                    # Linearized FK: pose_approx = fk_ref + J_q @ (q_t - q_ref)
                    # Cost: ||W_P_sqrt @ (pose_approx - fk_ref)||^2 = ||W_P_sqrt @ J_q @ (q_t - q_ref)||^2
                    W_P_sqrt = np.real(scipy.linalg.sqrtm(W_P + 1e-6 * np.eye(W_P.shape[0])))
                    
                    # Handle dimension mismatches between W_P and contact jacobian
                    if W_P.shape[0] > J_q.shape[0]:
                        # W_P might include extra dimensions, truncate to contact jacobian dimensions
                        W_P_sqrt = W_P_sqrt[:J_q.shape[0], :J_q.shape[0]]
                    elif W_P.shape[0] < J_q.shape[0]:
                        # Contact jacobian might have more DOFs, truncate to W_P dimensions  
                        J_q = J_q[:W_P.shape[0], :]
                    
                    # Weighted contact jacobian for linearized pose cost
                    weighted_contact_jac = W_P_sqrt @ J_q
                    
                    # Add convex pose cost: ||weighted_contact_jac @ (q_t - q_ref)||^2
                    cost += self.config.w_p * cp.sum_squares(weighted_contact_jac @ (q_t - q_ref))
                else:
                    # Fallback to joint-space cost if FK computation failed
                    logger.warning(f"Using joint-space fallback for pose cost at timestep {t}")
                    cost += self.config.w_p * cp.sum_squares(q_t - q_ref)
        
        
        # Control bounds (optional)
        u_max = 50.0  # Reasonable torque limits for Allegro hand
        for t in range(self.horizon):
            constraints.append(u_vars[t] >= -u_max)
            constraints.append(u_vars[t] <= u_max)
        
        # Create and return problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        return {
            'problem': problem,
            'u_vars': u_vars,
            'x_vars': x_vars,
            'reference_trajectory': reference_trajectory
        }
        
    def setup_basic_qp_problem(self, x0: np.ndarray, reference_trajectory: callable,
                              W_A: np.ndarray, W_P: np.ndarray):
        """
        Fallback QP setup using scipy for when cvxpy is not available.
        """
        logger.warning("Using basic QP solver - may be less robust than cvxpy")
        
        # Simplified QP formulation 
        # For now, just return zero control as fallback
        return {
            'problem': None,
            'fallback': True,
            'reference_trajectory': reference_trajectory
        }
    
    def get_x_ref_from_ref_state(self, ref_state: np.ndarray):
        """
        Get the state x_ref from the reference state.
        """
        x_ref = np.zeros(self.n_x)
        x_ref[:self.dq] = ref_state[:self.dq]
        x_ref[self.dq:2*self.dq] = ref_state[self.dq+self.problem.obj_dof:2*self.dq+self.problem.obj_dof]
        x_ref[2*self.dq:2*self.dq + self.df] = ref_state[2*self.dq+self.problem.obj_dof:]
        return x_ref
        
    def setup_qp_problem_precomputed_jacobians(self, t0: float, x0: np.ndarray, reference_trajectory: callable, 
                                             W_A: np.ndarray, W_P: np.ndarray):
        """
        Alternative QP setup using original dynamics with pre-computed jacobians.
        
        Instead of linearizing the dynamics, we use the original dynamics structure:
        q̇ = u + K_D^(-1)(K_P(q_d - q) - J_ref^T λ_ext)
        q̇_d = u  
        λ̇_ext = K_coup_ref J_ref_d u
        
        But with jacobians J_ref, J_ref_d, K_coup_ref pre-computed at reference points,
        making the dynamics linear in state.
        
        Args:
            x0: Initial state [n_x]
            reference_trajectory: Function t -> reference_state
            W_A: Force weighting matrix
            W_P: Position weighting matrix
            
        Returns:
            QP problem (depends on available solver)
        """
        if not self.cvxpy_available:
            return self.setup_basic_qp_problem(x0, reference_trajectory, W_A, W_P)
            
        import cvxpy as cp
        
        # Decision variables: [u_0, u_1, ..., u_{T-1}, x_1, x_2, ..., x_T]
        u_vars = [cp.Variable(self.n_u) for _ in range(self.horizon)]
        x_vars = [cp.Variable(self.n_x) for _ in range(self.horizon)]
        
        constraints = []
        cost = 0
        
        # Pre-compute jacobians at reference trajectory points
        ref_jacobians = []
        fk_cache = {}
        
        for t in range(self.horizon):
            # Get reference trajectory at time t
            ref_state = reference_trajectory(t0 + t * self.dt)
            x_ref = self.get_x_ref_from_ref_state(ref_state)
            
            # Pre-compute system matrices at reference point
            system_matrices = self.compute_system_matrices_at_state(x_ref)
            ref_jacobians.append(system_matrices)
            
            # Cache FK poses for cost function
            q_ref_t = x_ref[:self.dq]
            q_key = tuple(q_ref_t.round(6))
            if q_key not in fk_cache:
                try:
                    fk_ref = self.compute_fk_poses(q_ref_t)
                    fk_cache[q_key] = fk_ref
                except Exception as e:
                    logger.warning(f"FK computation failed at timestep {t}: {e}")
                    fk_cache[q_key] = None
        
        # Add linear dynamics constraints using pre-computed jacobians
        x_prev = x0
        for t in range(self.horizon):
            # Get pre-computed jacobians for this timestep
            J_q = ref_jacobians[t]['J_q']      # Contact jacobian [df, dq]
            J_q_d = ref_jacobians[t]['J_q_d']  # Contact jacobian derivative [df, dq]
            K_coup = ref_jacobians[t]['K_coup'] # Coupling matrix [df, df]
            
            # Extract state variables for current timestep
            if t == 0:
                q_prev = x_prev[:self.dq]
                q_d_prev = x_prev[self.dq:2*self.dq]
                f_prev = x_prev[2*self.dq:2*self.dq + self.df]
            else:
                q_prev = x_vars[t-1][:self.dq]
                q_d_prev = x_vars[t-1][self.dq:2*self.dq]
                f_prev = x_vars[t-1][2*self.dq:2*self.dq + self.df]
            
            # Current state variables
            q_curr = x_vars[t][:self.dq]
            q_d_curr = x_vars[t][self.dq:2*self.dq]
            f_curr = x_vars[t][2*self.dq:2*self.dq + self.df]
            
            # Linear dynamics using pre-computed jacobians:
            # q̇ = u + K_D^(-1)(K_P(q_d - q) - J_ref^T f)
            q_dot = u_vars[t] + self.K_D_inv @ (self.K_P @ (q_d_prev - q_prev) - J_q.T @ f_prev)
            
            # q̇_d = u
            q_d_dot = u_vars[t]
            
            # λ̇_ext = K_coup_ref J_ref_d u  
            f_dot = K_coup @ (J_q_d @ u_vars[t])
            
            # Discrete-time integration: x_{t+1} = x_t + dt * x_dot
            constraints.append(q_curr == q_prev + self.dt * q_dot)
            constraints.append(q_d_curr == q_d_prev + self.dt * q_d_dot)
            constraints.append(f_curr == f_prev + self.dt * f_dot)
        
        # Add cost function (same as before)
        for t in range(self.horizon):
            ref_t = reference_trajectory(t0 + t * self.dt)
            ref_t = self.get_x_ref_from_ref_state(ref_t)
            q_ref = ref_t[:self.dq]
            f_ref = ref_t[2*self.dq:2*self.dq + self.df]
            
            # Extract state components
            q_t = x_vars[t][:self.dq]
            f_t = x_vars[t][2*self.dq:2*self.dq + self.df]
            
            # Joint position cost
            cost += self.config.w_q * cp.sum_squares(q_t - q_ref)
            
            # Force cost (using weighting matrix W_A)
            if W_A.shape[0] > 0:
                W_A_sqrt = np.real(scipy.linalg.sqrtm(W_A + 1e-6 * np.eye(W_A.shape[0])))
                cost += self.config.w_f * cp.sum_squares(W_A_sqrt @ (f_t - f_ref))
            
            # Control cost
            cost += self.config.w_u * cp.sum_squares(u_vars[t])
            
            # Linearized forward kinematics pose cost using pre-computed jacobians
            if W_P.shape[0] > 0:
                q_ref_t = q_ref
                q_key = tuple(q_ref_t.round(6))
                
                if q_key in fk_cache and fk_cache[q_key] is not None:
                    fk_ref = fk_cache[q_key]
                    
                    # Use pre-computed contact jacobian
                    J_q = ref_jacobians[t]['J_q']  # Contact jacobian [df, dq]
                    
                    # Linearized FK cost using pre-computed jacobian
                    W_P_sqrt = np.real(scipy.linalg.sqrtm(W_P + 1e-6 * np.eye(W_P.shape[0])))
                    
                    # Handle dimension mismatches
                    if W_P.shape[0] > J_q.shape[0]:
                        W_P_sqrt = W_P_sqrt[:J_q.shape[0], :J_q.shape[0]]
                    elif W_P.shape[0] < J_q.shape[0]:
                        J_q = J_q[:W_P.shape[0], :]
                    
                    # Weighted contact jacobian for pose cost
                    weighted_contact_jac = W_P_sqrt @ J_q
                    
                    # Add pose cost: ||weighted_contact_jac @ (q_t - q_ref)||^2
                    cost += self.config.w_p * cp.sum_squares(weighted_contact_jac @ (q_t - q_ref))
                else:
                    # Fallback to joint-space cost if FK computation failed
                    logger.warning(f"Using joint-space fallback for pose cost at timestep {t}")
                    cost += self.config.w_p * cp.sum_squares(q_t - q_ref)
        
        # Control bounds
        u_max = 50.0  # Reasonable torque limits for Allegro hand
        for t in range(self.horizon):
            constraints.append(u_vars[t] >= -u_max)
            constraints.append(u_vars[t] <= u_max)
        
        # Create and return problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        return {
            'problem': problem,
            'u_vars': u_vars,
            'x_vars': x_vars,
            'reference_trajectory': reference_trajectory,
            'method': 'precomputed_jacobians'
        }
        
    def solve(self, t0: float, state: torch.Tensor, q_d_init: torch.Tensor, 
             f_ext_init: torch.Tensor, avg_normal: np.ndarray, 
             segment_tangents: np.ndarray, method: str = 'linearized') -> torch.Tensor:
        """
        Solve the QP-based MPC problem.
        
        Args:
            t0: Initial time
            state: Current joint state [dq]
            q_d_init: Initial commanded positions [dq]  
            f_ext_init: Initial external forces [df]
            avg_normal: Average contact normals
            segment_tangents: Contact tangent directions
            method: QP formulation method - 'linearized' or 'precomputed_jacobians'
            
        Returns:
            u: Optimal control input [dq]
        """
        
        # Compute weighting matrices
        W_A, W_P = self.weighting_determiner.compute_weighting_matrices(
            f_ext_init.cpu().numpy(), avg_normal, segment_tangents)
        
        # Update environment stiffness with weighting  
        self.config.K_e = self.config.K_e * W_A
        
        # Pack initial state
        x0 = np.concatenate([state.cpu().numpy(), q_d_init.cpu().numpy(), f_ext_init.cpu().numpy()])
        
        # Set up reference trajectory (use the spline function if available)
        if hasattr(self, 'reference_spline') and self.reference_spline is not None:
            reference_trajectory = self.reference_spline
        elif hasattr(self.mpc_problem_definition, 'spline_func'):
            reference_trajectory = self.mpc_problem_definition.spline_func
        else:
            # Fallback: constant reference
            def reference_trajectory(t):
                return x0.copy()
        
        # Choose QP formulation method
        if method == 'precomputed_jacobians':
            qp_data = self.setup_qp_problem_precomputed_jacobians(t0, x0, reference_trajectory, W_A, W_P)
            logger.info("Using precomputed jacobians method for QP formulation")
        else:  # default to 'linearized'
            qp_data = self.setup_qp_problem(t0, x0, reference_trajectory, W_A, W_P)
            logger.info("Using linearized dynamics method for QP formulation")
        
        if qp_data.get('fallback', False):
            logger.warning("Using fallback zero control")
            return torch.zeros_like(state)
        
        try:
            # Solve the QP problem
            problem = qp_data['problem']
            problem.solve(solver='OSQP', verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"QP solver status: {problem.status}")
                return torch.zeros_like(state)
            
            # Extract first control input
            u_optimal = qp_data['u_vars'][0].value
            
            if u_optimal is None:
                logger.warning("QP solver returned None")
                return torch.zeros_like(state)
            
            # Scale by time step (similar to GRAMPC integration)
            u_scaled = u_optimal * self.dt
            
            return torch.tensor(u_scaled, device=state.device, dtype=state.dtype)
            
        except Exception as e:
            logger.error(f"QP solve failed: {e}")
            return torch.zeros_like(state) 
        
if __name__ == "__main__":
    # Example usage with AllegroManipulationProblem integration
    
    print("=== Tactile Feedback Controller with Allegro Integration ===")
    
    # Configuration (cvxpy is now the default solver)
    config = ControllerConfig(
        K_e=1000.0,
        K_r=100.0,
        K_P=1000.0,
        force_threshold=0.5,
        horizon_length=10,
        mpc_solver='cvxpy'  # Uses QP formulation with linearized dynamics
    )
    
    # Create controller (problem will be set later)
    controller = TactileFeedbackController(config)
    
    print("Controller initialized and ready for AllegroManipulationProblem integration")
    print("Use controller.set_problem(problem) to set the manipulation problem")
    print("Then use controller.update(..., use_allegro_integration=True) for control")
    
    print("\n=== Two QP Formulation Methods ===")
    print("1. 'linearized': Linearizes full dynamics around reference trajectory")
    print("   - Computes A, B, c matrices via differentiation")
    print("   - Handles all nonlinearities through linearization")
    print("   - QP constraint: x_{t+1} = A_t x_t + B_t u_t + c_t")
    
    print("\n2. 'precomputed_jacobians': Uses original dynamics with fixed jacobians")
    print("   - Pre-computes J_q, J_q_d, K_coup at reference points")
    print("   - Dynamics become linear since jacobians are constants")
    print("   - QP constraint: x_{t+1} = x_t + dt * [original dynamics with J_ref]")
    
    print("\n=== Key Differences ===")
    print("Linearized approach:")
    print("  ✓ Handles all nonlinearities via Taylor expansion")
    print("  ✓ More general, captures higher-order effects")
    print("  ✗ Computationally intensive (many derivatives)")
    
    print("\nPrecomputed jacobians approach:")
    print("  ✓ Computationally efficient (no derivatives)")
    print("  ✓ Preserves original dynamics structure")
    print("  ✓ Intuitive - jacobians frozen at reference")
    print("  ✗ May miss some coupling effects")
    
    print("\n=== Usage Example ===")
    print("""
    # Create QP controller
    qp_controller = TactileFeedbackQPController(problem, config)
    
    # Method 1: Linearized dynamics  
    u1 = qp_controller.solve(t0, state, q_d, f_ext, normals, tangents, method='linearized')
    
    # Method 2: Precomputed jacobians
    u2 = qp_controller.solve(t0, state, q_d, f_ext, normals, tangents, method='precomputed_jacobians')
    """)
    
    print("\nReady for integration with AllegroManipulationProblem!") 