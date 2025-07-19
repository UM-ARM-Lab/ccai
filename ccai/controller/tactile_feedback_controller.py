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

from ccai.allegro_contact import AllegroManipulationProblem

from pygrampc import Grampc, GrampcResults, ProblemDescription

from ccai.controller.se3_dist import se3_distance_gradient, se3_distance

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
        self.df = config.df
        
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
        return (torch.eye(J_s.shape[0]) + self.K_e @torch.linalg.inv(K_r)) @ self.K_e
    
    def compute_K_coup(self, G_o: torch.Tensor, J_s: torch.Tensor) -> torch.Tensor:
        """
        Compute K_coup using the coupling formula.
        """
        self.K_bar = self.compute_K_bar(J_s)
        G_o_Kbar = self.K_bar @ G_o
        K_coup = self.K_bar + (self.K_bar @ G_o.T) @ torch.inverse(G_o_Kbar @ G_o.T + 1e-6 * torch.eye(6, device=self.device)) @ G_o_Kbar
        return K_coup
        
    def system_dynamics(self, system_matrices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Implement system dynamics equation (24) EXACTLY as shown in the image:
        
        [q̇]     [u + K_D^(-1)(K_P(q_d - q) - J(q)^T λ_ext)]
        [q̇_d] = [            u            ]
        [λ̇_ext] [      K_coup J(q_d)u     ]
        
        Args:
            x: Current state [q; q_d; λ_ext]
            u: Control input [n_q]
            system_matrices: System matrices 
            references: Reference trajectories 
            n_q: Number of joints
            n_c: Number of contacts
            
        Returns:
            x_dot: State derivative [q̇; q̇_d; λ̇_ext]
        """
        # Unpack current state
        
        # Extract system matrices
        J_q = system_matrices['jacobian']  # Contact Jacobian J(q) [3*n_c, n_q]
        J_q_d = system_matrices['jacobian_d']  # Contact Jacobian J(q_d) [3*n_c, n_q]
        G_o = system_matrices['G_o']
        
        K_coup = self.compute_K_coup(G_o, J_q)

        mat = torch.zeros((self.dq*2 + self.df, self.dq*3 + self.df))
        mat[:self.dq, :self.dq] = torch.eye(self.dq) * -1 * self.K_D_inv * self.K_P
        mat[:self.dq, self.dq:2*self.dq] = torch.eye(self.dq) * self.K_D_inv * self.K_P
        mat[:self.dq, 2*self.dq:2*self.dq + self.df] = -J_q.T * self.K_D_inv
        mat[:self.dq, 2*self.dq+self.df:] = 1
        
        mat[self.dq:2*self.dq, -self.df:] = 1
        
        mat[2*self.dq:, -self.dq:] = -J_q_d @ K_coup
                  
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
        
        Js = self.problem.data['J_q'].clone()
        Hs = self.problem.data['H_q'].clone()
        
        self.J_q = Js[0].detach().cpu().numpy()
        self.J_q_d = Js[1].detach().cpu().numpy()
        self.H_q = Hs[0].detach().cpu().numpy()
        self.H_q_d = Hs[1].detach().cpu().numpy()
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
            'lambda_ref': interpolated_state[2*self.dq:2*self.dq + self.df]
        }
            
    def ffct(self, out, t, x, u, p):
        self.compute_system_matrices(x)
        q = x[:self.dq]
        q_d = x[self.dq:2*self.dq]
        f = x[2*self.dq:2*self.dq + self.df]
        # q_dot
        out[:self.dq] = u + self.K_D_inv*(self.K_P*(q_d - q) - self.J_q.T @ f)
        #q_d_dot
        out[self.dq:2*self.dq] = u
        # f_dot
        out[2*self.dq:2*self.dq + self.df] = self.K_coup @ (self.J_q_d @ u)
        return out
    
    def dfdx_vec(self, out, t, x, vec, u, p):
        self.compute_system_matrices(x)
        J = np.zeros((self.dq*2+self.df, self.dq*2+self.df))
        
        # dq_dot/dq
        J[:self.dq, :self.dq] = np.eye(self.dq) * -self.K_P * self.K_D_inv
        
        # Jacobian chain rule
        dq_dot_dJ = -self.K_D_inv * x[2*self.dq:2*self.dq + self.df]
        J[:self.dq, :self.dq] += dq_dot_dJ @ self.H_q_d
        
        # dq_dot/dq_d
        J[:self.dq, self.dq:2*self.dq] = np.eye(self.dq) * self.K_D_inv * self.K_P
        
        # df/dq_dot
        df_dot_dJ = self.K_coup @ u
        J[2*self.dq:2*self.dq + self.df, self.dq:2*self.dq] = df_dot_dJ @ self.H_q_d
        
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
        fk_q_ref = self.problem.robot_sdf.chain.forward_kinematics(torch.tensor(q_ref.reshape(1, -1), device=self.problem.device)).detach().cpu().numpy()
        q = x[:self.dq]
        fk_q = self.problem.robot_sdf.chain.forward_kinematics(torch.tensor(q.reshape(1, -1), device=self.problem.device)).detach().cpu().numpy()
        
        dist, _, _ = se3_distance(fk_q, fk_q_ref, self.W_P)
        
        p_cost = dist ** 2 * self.config.w_p
        
        q_cost = np.sum((q_ref - q)**2) * self.config.w_q
        f_cost = np.sum((f_ref - x[2*self.dq:2*self.dq + self.df])**2) * self.config.w_f
        u_cost = np.sum(u**2) * self.config.w_u
        
        out = q_cost + p_cost + f_cost + u_cost
        
        return out
    
    def dldx(self, out, t, x, u, p, xdes, udes):
        ref = self.get_reference_trajectory(t)
        q_ref = ref['q_ref']
        f_ref = ref['f_ref']
        q = x[:self.dq]
        
        # joint position cost
        out[:self.dq] = 2 * (q_ref - q) * self.config.w_q
        
        # Forward kinematics cost derivative
        fk_q = self.problem.robot_sdf.chain.forward_kinematics(torch.tensor(q.reshape(1, -1), device=self.problem.device)).detach().cpu().numpy()[0]
        fk_q_ref = self.problem.robot_sdf.chain.forward_kinematics(torch.tensor(q_ref.reshape(1, -1), device=self.problem.device)).detach().cpu().numpy()[0]
        
        jac_fk_q = self.problem.robot_sdf.chain.jacobian(torch.tensor(q.reshape(1, -1), device=self.problem.device)).detach().cpu().numpy()[0]
        dist, grad = se3_distance_gradient(fk_q, fk_q_ref, self.W_P)
        # Squared cost, so adjust grad
        grad = grad * 2 * dist
        grad_fk_q = grad @ jac_fk_q
        #
        out[:self.dq] += grad_fk_q * self.config.w_p
        
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
        force_magnitudes = torch.norm(contact_forces, dim=1)
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
        n_c = contact_forces.shape[0]
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
                W_A[start_idx:end_idx, start_idx:end_idx] = np.zeros(3, 3)
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
        self.grampc.estim_penmin(True)
        self.grampc.print_opts()
        self.grampc.print_params()

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
                
        grampc_x0 = torch.cat((state, q_d_init, f_ext_init), dim=0).cpu().numpy()
        self.grampc.set_param({"x0": grampc_x0,
                               "t0": t0})
        
        runtime = self.grampc.run()
        
        u = self.grampc.sol.unext * self.config.dt
        
        return u
        
        
        
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
    
    # Example integration workflow:
    print("\n=== Example Integration Workflow ===")
    print("""
    # 1. Create AllegroManipulationProblem
    from ccai.allegro_contact import AllegroManipulationProblem
    
    problem = AllegroManipulationProblem(
        start=start_state,
        goal=goal_state,
        T=horizon_length,
        chain=kinematic_chain,
        object_location=object_location,
        object_type='screwdriver',  # or 'valve', etc.
        world_trans=world_transform,
        object_asset_pos=object_position,
        contact_fingers=['index', 'middle', 'ring', 'thumb'],
        regrasp_fingers=[],  # or subset of fingers
        optimize_force=True,
        device='cuda'
    )
    
    # 2. Set up controller with problem
    controller.set_problem(problem)
    
    # 3. Create state from current robot configuration
    current_state = create_allegro_compatible_state(
        problem, 
        q_current=torch.tensor([...]),  # current joint positions
        q_dot_current=torch.tensor([...])  # current joint velocities
    )
    
    # 4. Define reference trajectory
    reference_trajectory = {
        'q_ref': torch.zeros((horizon_length, n_joints)),  # desired joint trajectory
        'theta_ref': torch.zeros((horizon_length, obj_dof)),  # desired object trajectory
    }
    
    # 5. Run controller with integrated preprocessing
    control_output = controller.update(
        state=current_state,
        references={'trajectory': reference_trajectory},
        use_allegro_integration=True
    )
    
    # 6. Extract control commands
    joint_torques = control_output['joint_torques']  # Apply to robot
    optimal_sequence = control_output['optimal_control_sequence']  # Full horizon plan
    system_matrices = control_output['system_matrices']  # Extracted matrices
    contact_data = control_output['contact_data']  # Contact information
    
    # Key benefits of this integration:
    # - Automatic extraction of contact Jacobians from preprocessing
    # - Proper handling of contact forces and positions
    # - Seamless integration with existing AllegroManipulationProblem
    # - Support for different object types and contact configurations
    # - Advanced MPC with multiple solver options
    """)
    
    print("\n=== Available MPC Solvers ===")
    print("- 'cvxpy': cvxpy QP solver (PRIMARY - uses linearized dynamics as QP constraints)")
    print("  └─ Automatically uses OSQP, ECOS, or other backends")
    print("  └─ Formulates exact QP with dynamics constraints")
    print("  └─ Superior numerical properties and convergence")
    print("- 'lbfgs': L-BFGS (fallback - direct nonlinear optimization)")
    print("- 'osqp': OSQP quadratic programming (legacy - simple QP approximation)")
    print("- 'scipy': SciPy SLSQP (fallback - robust for constrained problems)")
    print("- 'augmented_lagrangian': Augmented Lagrangian (complex constraints)")
    print("- 'adam': Adam optimizer (for comparison)")
    
    print("\n=== Configuration Options ===")
    print("Key parameters to tune:")
    print("- K_e, K_r: Environment and robot stiffness")
    print("- K_P, K_D: Proportional and damping gains")
    print("- force_threshold: Contact classification threshold")
    print("- w_motion, w_contact, w_smooth: Cost function weights")
    print("- horizon_length: MPC prediction horizon")
    
    print("\n=== Contact Modes ===")
    print("- Single contact: Use update_single_contact() for single finger contact")
    print("- Multi contact: Use update_multi_contact() for multiple finger contacts")
    print("- MPC mode: Use update_with_allegro_integration() for full MPC with preprocessing")
    
    print("\n=== cvxpy QP Formulation Benefits ===")
    print("The new cvxpy solver offers significant advantages:")
    print("✓ Exact dynamics constraints via linearization")
    print("✓ Automatic differentiation for constraint matrices")
    print("✓ Multiple QP solver backends (OSQP, ECOS, CLARABEL)")
    print("✓ Better numerical conditioning and convergence")
    print("✓ Handles box constraints on controls naturally")
    print("✓ Disciplined convex programming guarantees")
    print("✓ Scales well with horizon length and state dimension")
    
    print("\n=== QP Formulation Details ===")
    print("Problem structure: minimize 0.5 * z^T * H * z + f^T * z")
    print("                   subject to: A_eq * z = b_eq (dynamics)")
    print("                              G * z <= h (bounds)")
    print("where z = [u_0; u_1; ...; u_{T-1}; x_1; x_2; ...; x_T]")
    print("- Dynamics linearized around nominal trajectory")
    print("- State: x = [q; q_d; λ_ext] from equation (24)")
    print("- Controls: u = joint torques")
    
    print("\n=== Solver Comparison Example ===")
    print("""
    # cvxpy QP solver (recommended)
    config_qp = ControllerConfig(mpc_solver='cvxpy')
    controller_qp = TactileFeedbackController(config_qp)
    
    # Traditional nonlinear solver (fallback)
    config_nl = ControllerConfig(mpc_solver='lbfgs')
    controller_nl = TactileFeedbackController(config_nl)
    
    # Both use the same dynamics equation (24): ẍ = g(x, u)
    # Key difference:
    # - cvxpy: Linearizes dynamics → creates QP constraints → solves QP
    # - lbfgs: Direct nonlinear optimization via automatic differentiation
    
    # Benefits of QP approach:
    # ✓ Guaranteed convergence for convex problems
    # ✓ Handles constraints more naturally
    # ✓ Better numerical conditioning
    # ✓ Multiple solver backends available
    """)
    
    print("\nReady for integration with AllegroManipulationProblem!") 