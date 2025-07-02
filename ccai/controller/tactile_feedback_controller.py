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

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Configuration for the tactile feedback controller."""
    # Environment stiffness and damping
    K_e: float = 1000.0  # Environment stiffness
    K_r: float = 100.0   # Robot stiffness
    K_P: float = 1000.0  # Proportional gain
    K_D: float = 50.0    # Damping gain
    
    # Force threshold for contact classification
    force_threshold: float = 0.5  # Δ in the paper
    
    # MPC parameters
    horizon_length: int = 10
    dt: float = 0.01
    mpc_solver: str = "lbfgs"  # Options: 'lbfgs', 'osqp', 'scipy', 'adam', 'augmented_lagrangian'
    
    # Weighting parameters
    w_motion: float = 1.0
    w_contact: float = 1.0
    w_smooth: float = 0.1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ForceMotionModel:
    """
    Force-Motion Model for tactile-feedback control.
    
    Implements equations (15)-(26) from the paper.
    """
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.K_e = config.K_e
        self.K_r = config.K_r
        self.K_P = config.K_P
        self.K_D = config.K_D
        self.K_s = (1 + self.K_e / self.K_r) * self.K_e
        self.K_bar = self.K_s
        
    def single_contact_force(self, p_d: torch.Tensor, p_c: torch.Tensor) -> torch.Tensor:
        """
        Compute contact force for single contact case (Equation 15).
        
        Args:
            p_d: Desired contact point position [3]
            p_c: Current contact point position [3]
            
        Returns:
            λ_ext: Contact force [3]
        """
        dp_c_d = p_d - p_c
        
        # λ_ext = K_e * dp_c,d = (I + K_e * K_r^(-1))^(-1) * K_e * dp_c,d
        I = torch.eye(3, device=p_d.device)
        K_e_Kr_inv = self.K_e / self.K_r * I
        inv_term = torch.inverse(I + K_e_Kr_inv)
        lambda_ext = inv_term @ (self.K_e * dp_c_d)
        
        return lambda_ext
    
    def multi_contact_force(self, P_d: torch.Tensor, P_c: torch.Tensor, 
                           K_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute contact forces for multiple contacts (Equation 17).
        
        Args:
            P_d: Desired contact positions [3*n_c]
            P_c: Current contact positions [3*n_c]
            K_matrix: Stiffness matrix [3*n_c, 3*n_c]
            
        Returns:
            A_ext: Contact forces [3*n_c]
        """
        A_ext = K_matrix @ (P_d - P_c)
        return A_ext
    
    def joint_motion_single(self, q_d: torch.Tensor, q: torch.Tensor, 
                           q_d_dot: torch.Tensor, J: torch.Tensor, 
                           lambda_ext: torch.Tensor) -> torch.Tensor:
        """
        Compute joint acceleration for single contact (Equation 16).
        
        Args:
            q_d: Desired joint positions
            q: Current joint positions
            q_d_dot: Desired joint velocities
            J: Jacobian matrix
            lambda_ext: Contact force
            
        Returns:
            q_ddot: Joint accelerations
        """
        K_P_inv = torch.eye(len(q), device=q.device) / self.K_P
        q_ddot = q_d_dot + K_P_inv @ (self.K_P * (q_d - q) - J.T @ lambda_ext)
        return q_ddot
    
    def joint_motion_multi(self, q_d: torch.Tensor, q: torch.Tensor, 
                          q_d_dot: torch.Tensor, J: torch.Tensor, 
                          A_ext: torch.Tensor) -> torch.Tensor:
        """
        Compute joint acceleration for multiple contacts (Equation 18).
        
        Args:
            q_d: Desired joint positions
            q: Current joint positions  
            q_d_dot: Desired joint velocities
            J: Jacobian matrix
            A_ext: Contact forces
            
        Returns:
            q_ddot: Joint accelerations
        """
        K_P_inv = torch.eye(len(q), device=q.device) / self.K_P
        q_ddot = q_d_dot + K_P_inv @ (self.K_P * (q_d - q) - J.T @ A_ext)
        return q_ddot


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
    
    def compute_weighting_matrices(self, contact_forces: torch.Tensor, 
                                 contact_normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        W_A = torch.zeros((3 * n_c, 3 * n_c), device=contact_forces.device)
        W_P = torch.zeros((3 * n_c, 3 * n_c), device=contact_forces.device)
        
        for i in range(n_c):
            start_idx = 3 * i
            end_idx = 3 * (i + 1)
            
            if is_active[i]:
                # Active contact: track normal force, perform position tracking
                n_i = contact_normals[i]  # Contact normal
                
                # Create normal and tangential projections
                N_i = torch.outer(n_i, n_i)  # Normal projection
                T_i = torch.eye(3, device=contact_forces.device) - N_i  # Tangential projection
                
                # W_A focuses on normal direction for force control
                W_A[start_idx:end_idx, start_idx:end_idx] = N_i
                
                # W_P focuses on tangential directions for position control  
                W_P[start_idx:end_idx, start_idx:end_idx] = T_i
            else:
                # Inactive contact: position tracking only
                W_A[start_idx:end_idx, start_idx:end_idx] = torch.zeros(3, 3, device=contact_forces.device)
                W_P[start_idx:end_idx, start_idx:end_idx] = torch.eye(3, device=contact_forces.device)
                
        return W_A, W_P


class ModelPredictiveController:
    """
    Model Predictive Controller for tactile-feedback control (Section 6.4).
    
    Implements equation (24): ẍ = g(x, u) where x = [q; q̇; q̇_d; λ_ext]
    """
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.horizon = config.horizon_length
        self.dt = config.dt
        self.device = config.device
        
        # Initialize force motion model for dynamics
        self.force_motion_model = ForceMotionModel(config)
        
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
        
    def system_dynamics(self, x: torch.Tensor, u: torch.Tensor, 
                       system_matrices: Dict[str, torch.Tensor],
                       references: Dict[str, torch.Tensor],
                       n_q: int, n_c: int) -> torch.Tensor:
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
        q, q_d, lambda_ext = self.unpack_state(x, n_q, n_c)
        
        # Extract system matrices
        J_q = system_matrices['jacobian']  # Contact Jacobian J(q) [3*n_c, n_q]
        K_P = system_matrices.get('K_P', self.config.K_P * torch.eye(n_q, device=self.device))  # Proportional gain matrix
        K_D = system_matrices.get('K_D', self.config.K_D * torch.eye(n_q, device=self.device))  # Damping gain matrix
        
        # Handle K_coup computation for AllegroManipulationProblem
        if 'K_coup' in system_matrices:
            K_coup = system_matrices['K_coup']
        else:
            # Compute K_coup using the coupling formula
            G_o = system_matrices.get('G_o', J_q)  # Use Jacobian as default G_o
            K_bar = self.config.K_e * torch.eye(3 * n_c, device=self.device)
            
            if n_c > 0 and G_o.numel() > 0:
                try:
                    G_o_Kbar = K_bar @ G_o  # [3*n_c, n_q]
                    middle_term = G_o_Kbar @ G_o.T  # [3*n_c, 3*n_c]
                    middle_inv = torch.inverse(middle_term + 1e-6 * torch.eye(3*n_c, device=self.device))
                    K_coup = K_bar + G_o_Kbar.T @ middle_inv @ G_o_Kbar
                except:
                    K_coup = K_bar
            else:
                K_coup = K_bar
        
        # 1. q̇ = u + K_D^(-1)(K_P(q_d - q) - J(q)^T λ_ext)
        try:
            K_D_inv = torch.inverse(K_D)
        except:
            K_D_inv = torch.pinverse(K_D)
            
        if n_c > 0 and J_q.numel() > 0:
            J_q_T_lambda = J_q.T @ lambda_ext.reshape(-1)
        else:
            J_q_T_lambda = torch.zeros(n_q, device=self.device)
            
        q_dot = u + K_D_inv @ (K_P @ (q_d - q) - J_q_T_lambda)
        
        # 2. q̇_d = u
        q_d_dot = u
        
        # 3. λ̇_ext = K_coup J(q_d)u
        # Need J(q_d) - the Jacobian evaluated at q_d instead of q
        J_q_d = system_matrices.get('J_q_d', J_q)  # Default to J(q) if J(q_d) not provided
        
        if n_c > 0 and J_q_d.numel() > 0:
            # For AllegroManipulationProblem integration, we may need to handle varying contact states
            if hasattr(J_q_d, 'shape') and len(J_q_d.shape) > 2:
                J_q_d = J_q_d.reshape(3*n_c, n_q)  # Ensure correct shape
            lambda_ext_dot = K_coup @ (J_q_d @ u)
        else:
            lambda_ext_dot = torch.zeros(3*n_c, device=self.device)
        
        # Pack state derivative: [q̇; q̇_d; λ̇_ext]
        x_dot = torch.cat([q_dot, q_d_dot, lambda_ext_dot])
        
        return x_dot
    
    def integrate_dynamics(self, x: torch.Tensor, u: torch.Tensor,
                          system_matrices: Dict[str, torch.Tensor],
                          references: Dict[str, torch.Tensor],
                          n_q: int, n_c: int) -> torch.Tensor:
        """
        Integrate system dynamics using Euler integration.
        
        Args:
            x: Current state
            u: Control input
            system_matrices: System matrices
            references: Reference values
            n_q: Number of joints
            n_c: Number of contacts
            
        Returns:
            x_next: Next state
        """
        x_dot = self.system_dynamics(x, u, system_matrices, references, n_q, n_c)
        x_next = x + self.dt * x_dot
        return x_next
        
    def solve_mpc(self, current_state: Dict[str, torch.Tensor], 
                  reference_trajectory: Dict[str, torch.Tensor],
                  contact_references: Dict[str, torch.Tensor],
                  system_matrices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Solve MPC optimization problem using equation (24) dynamics.
        
        Args:
            current_state: Current robot state (q, q_dot, contact_positions, etc.)
            reference_trajectory: Desired motion trajectory over horizon
            contact_references: Desired contact forces over horizon
            system_matrices: System dynamics matrices (mass, Jacobians, etc.)
            
        Returns:
            optimal_control: Optimal joint torque sequence [horizon, n_joints]
        """
        # Extract state components
        q = current_state['q']
        q_dot = current_state['q_dot']
        contact_forces = current_state['contact_forces']
        contact_positions = current_state['contact_positions']
        
        # Get dimensions
        n_q = len(q)
        n_c = contact_forces.shape[0] if len(contact_forces.shape) > 1 else len(contact_forces) // 3
        
        # Extract references over horizon
        q_ref = reference_trajectory.get('q_ref', torch.zeros((self.horizon, n_q), device=self.device))
        q_dot_ref = reference_trajectory.get('q_dot_ref', torch.zeros((self.horizon, n_q), device=self.device))
        lambda_ref = contact_references.get('lambda_ref', torch.zeros((self.horizon, 3*n_c), device=self.device))
        
        # Initialize current state according to equation (24)
        q_d_current = reference_trajectory.get('q_ref', torch.zeros_like(q))
        if len(q_d_current.shape) > 1:
            q_d_current = q_d_current[0]  # Take first time step
            
        lambda_ext_current = contact_forces.flatten() if len(contact_forces.shape) > 1 else contact_forces
        
        # Pack initial state: x = [q; q_d; λ_ext]
        x_current = self.pack_state(q, q_d_current, lambda_ext_current)
        
        # Decision variables: joint torque sequence over horizon  
        control_dim = n_q
        u = torch.zeros((self.horizon, control_dim), device=self.device, requires_grad=True)
        
        # Choose solver based on configuration
        solver_type = getattr(self.config, 'mpc_solver', 'lbfgs')
        
        if solver_type == 'osqp':
            return self._solve_with_osqp(x_current, q_ref, q_dot_ref, lambda_ref, 
                                       system_matrices, contact_references, contact_positions, n_q, n_c)
        elif solver_type == 'scipy':
            return self._solve_with_scipy(x_current, q_ref, q_dot_ref, lambda_ref,
                                        system_matrices, contact_references, contact_positions, n_q, n_c)
        elif solver_type == 'augmented_lagrangian':
            return self._solve_with_augmented_lagrangian(x_current, q_ref, q_dot_ref, lambda_ref,
                                                       system_matrices, contact_references, contact_positions, n_q, n_c)
        else:  # 'lbfgs' or 'adam'
            return self._solve_with_pytorch(x_current, q_ref, q_dot_ref, lambda_ref,
                                          system_matrices, contact_references, contact_positions, 
                                          n_q, n_c, solver_type)
    
    def _solve_with_pytorch(self, x_current, q_ref, q_dot_ref, lambda_ref, system_matrices, 
                           contact_references, contact_positions, n_q, n_c, solver_type='lbfgs'):
        """Solve MPC using PyTorch optimizers (L-BFGS or Adam)."""
        # Decision variables: joint torque sequence over horizon
        control_dim = n_q
        u = torch.zeros((self.horizon, control_dim), device=self.device, requires_grad=True)
        
        def closure():
            # Forward simulation using equation (24) dynamics
            x_pred = torch.zeros((self.horizon + 1, len(x_current)), device=self.device)
            x_pred[0] = x_current
            
            total_cost = 0.0
            
            for t in range(self.horizon):
                # Get references for current time step using helper method
                current_refs = self._extract_allegro_references(
                    {'q_ref': q_ref, **contact_references}, t
                )
                # Add current contact positions
                current_refs['P_c'] = contact_positions
                
                # Integrate dynamics using equation (24): ẍ = g(x, u)
                x_pred[t + 1] = self.integrate_dynamics(
                    x_pred[t], u[t], system_matrices, current_refs, n_q, n_c
                )
                
                # Unpack predicted state
                q_pred, q_d_pred, lambda_pred = self.unpack_state(x_pred[t + 1], n_q, n_c)
                
                # Compute cost terms
                # Motion tracking cost (actual vs reference position)
                motion_cost = torch.norm(q_pred - q_ref[t])**2
                
                # Command tracking cost (commanded vs reference position)
                command_cost = torch.norm(q_d_pred - q_ref[t])**2
                
                # Contact force tracking cost  
                contact_cost = torch.norm(lambda_pred - lambda_ref[t])**2
                
                # Control smoothness cost
                control_cost = torch.norm(u[t])**2
                
                # Total cost
                total_cost += (self.config.w_motion * (motion_cost + command_cost) + 
                             self.config.w_contact * contact_cost +
                             self.config.w_smooth * control_cost)
            
            return total_cost
        
        # Choose optimizer
        if solver_type == 'lbfgs':
            optimizer = torch.optim.LBFGS([u], lr=1.0, max_iter=20, 
                                        tolerance_grad=1e-6, tolerance_change=1e-8)
            max_iterations = 10
        else:  # adam
            optimizer = torch.optim.Adam([u], lr=0.01)
            max_iterations = 100
        
        # Optimization loop
        for iteration in range(max_iterations):
            if solver_type == 'lbfgs':
                def closure_with_grad():
                    optimizer.zero_grad()
                    loss = closure()
                    loss.backward()
                    return loss
                
                optimizer.step(closure_with_grad)
                
                # Check convergence
                with torch.no_grad():
                    current_loss = closure()
                    if current_loss.item() < 1e-4:
                        break
            else:  # adam
                optimizer.zero_grad()
                loss = closure()
                loss.backward()
                optimizer.step()
                
                if loss.item() < 1e-4:
                    break
                    
        return u.detach()
    
    def _solve_with_osqp(self, x_current, q_ref, q_dot_ref, lambda_ref, system_matrices,
                        contact_references, contact_positions, n_q, n_c):
        """Solve MPC using OSQP for quadratic approximation."""
        try:
            import osqp
            import scipy.sparse as sp
        except ImportError:
            print("OSQP not available, falling back to L-BFGS")
            return self._solve_with_pytorch(x_current, q_ref, q_dot_ref, lambda_ref,
                                          system_matrices, contact_references, contact_positions, n_q, n_c)
        
        # For QP formulation, we linearize the dynamics around current trajectory
        # This is a simplified QP approximation - in practice would need more sophisticated linearization
        
        control_dim = n_q
        decision_vars = self.horizon * control_dim
        
        # Quadratic cost matrix (regularization)
        P = sp.eye(decision_vars) * self.config.w_smooth
        
        # Linear cost (tracking error linearization)
        q = torch.zeros(decision_vars, device='cpu')  # OSQP requires CPU
        
        # Simple box constraints on control torques
        u_max = 100.0  # Maximum joint torque [Nm]
        l = -u_max * torch.ones(decision_vars)
        u_bound = u_max * torch.ones(decision_vars)
        
        # No dynamics constraints in this simplified version
        A = sp.csc_matrix((0, decision_vars))
        l_eq = torch.tensor([])
        u_eq = torch.tensor([])
        
        # Setup OSQP
        prob = osqp.OSQP()
        prob.setup(P=P, q=q.numpy(), A=A, l=l_eq.numpy(), u=u_eq.numpy(),
                  l_box=l.numpy(), u_box=u_bound.numpy(), verbose=False)
        
        # Solve
        res = prob.solve()
        
        if res.info.status == 'solved':
            u_solution = torch.from_numpy(res.x).reshape(self.horizon, control_dim).to(self.device)
            return u_solution
        else:
            print(f"OSQP failed with status: {res.info.status}, falling back to L-BFGS")
            return self._solve_with_pytorch(x_current, q_ref, q_dot_ref, lambda_ref,
                                          system_matrices, contact_references, contact_positions, n_q, n_c)
    
    def _solve_with_scipy(self, x_current, q_ref, q_dot_ref, lambda_ref, system_matrices,
                         contact_references, contact_positions, n_q, n_c):
        """Solve MPC using scipy.optimize (SLSQP)."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            print("SciPy not available, falling back to L-BFGS")
            return self._solve_with_pytorch(x_current, q_ref, q_dot_ref, lambda_ref,
                                          system_matrices, contact_references, contact_positions, n_q, n_c)
        
        control_dim = n_q
        decision_vars = self.horizon * control_dim
        
        # Initial guess
        u0 = torch.zeros(decision_vars, device='cpu').numpy()
        
        def objective(u_flat):
            u_tensor = torch.from_numpy(u_flat).reshape(self.horizon, control_dim).to(self.device)
            
            # Forward simulation
            x_pred = torch.zeros((self.horizon + 1, len(x_current)), device=self.device)
            x_pred[0] = x_current
            
            total_cost = 0.0
            
            for t in range(self.horizon):
                # Get references for current time step using helper method
                current_refs = self._extract_allegro_references(
                    {'q_ref': q_ref, **contact_references}, t
                )
                current_refs['P_c'] = contact_positions
                
                x_pred[t + 1] = self.integrate_dynamics(
                    x_pred[t], u_tensor[t], system_matrices, current_refs, n_q, n_c
                )
                
                q_pred, q_d_pred, lambda_pred = self.unpack_state(x_pred[t + 1], n_q, n_c)
                
                motion_cost = torch.norm(q_pred - q_ref[t])**2
                command_cost = torch.norm(q_d_pred - q_ref[t])**2
                contact_cost = torch.norm(lambda_pred - lambda_ref[t])**2
                control_cost = torch.norm(u_tensor[t])**2
                
                total_cost += (self.config.w_motion * (motion_cost + command_cost) + 
                             self.config.w_contact * contact_cost +
                             self.config.w_smooth * control_cost)
            
            return total_cost.detach().cpu().numpy()
        
        def jacobian(u_flat):
            u_tensor = torch.from_numpy(u_flat).reshape(self.horizon, control_dim).to(self.device)
            u_tensor.requires_grad_(True)
            
            # For gradient, we need to recompute without detach
            x_pred = torch.zeros((self.horizon + 1, len(x_current)), device=self.device)
            x_pred[0] = x_current
            
            total_cost = 0.0
            
            for t in range(self.horizon):
                # Get references for current time step using helper method
                current_refs = self._extract_allegro_references(
                    {'q_ref': q_ref, **contact_references}, t
                )
                current_refs['P_c'] = contact_positions
                
                x_pred[t + 1] = self.integrate_dynamics(
                    x_pred[t], u_tensor[t], system_matrices, current_refs, n_q, n_c
                )
                
                q_pred, q_d_pred, lambda_pred = self.unpack_state(x_pred[t + 1], n_q, n_c)
                
                motion_cost = torch.norm(q_pred - q_ref[t])**2
                command_cost = torch.norm(q_d_pred - q_ref[t])**2
                contact_cost = torch.norm(lambda_pred - lambda_ref[t])**2
                control_cost = torch.norm(u_tensor[t])**2
                
                total_cost += (self.config.w_motion * (motion_cost + command_cost) + 
                             self.config.w_contact * contact_cost +
                             self.config.w_smooth * control_cost)
            
            grad = torch.autograd.grad(total_cost, u_tensor)[0]
            return grad.flatten().detach().cpu().numpy()
        
        # Box constraints on joint torques
        u_max = 100.0  # Maximum joint torque [Nm]
        bounds = [(-u_max, u_max) for _ in range(decision_vars)]
        
        # Solve
        result = minimize(objective, u0, method='SLSQP', jac=jacobian, bounds=bounds,
                         options={'ftol': 1e-6, 'maxiter': 50})
        
        if result.success:
            u_solution = torch.from_numpy(result.x).reshape(self.horizon, control_dim).to(self.device)
            return u_solution
        else:
            print(f"SciPy optimization failed: {result.message}, falling back to L-BFGS")
            return self._solve_with_pytorch(x_current, q_ref, q_dot_ref, lambda_ref,
                                          system_matrices, contact_references, contact_positions, n_q, n_c)
    
    def _solve_with_augmented_lagrangian(self, x_current, q_ref, q_dot_ref, lambda_ref, system_matrices,
                                       contact_references, contact_positions, n_q, n_c):
        """Solve MPC using Augmented Lagrangian method."""
        
        control_dim = n_q
        decision_vars = self.horizon * control_dim
        
        # Initial guess for control sequence
        u_flat = torch.zeros(decision_vars, device=self.device, requires_grad=True)
        
        # Augmented Lagrangian parameters
        rho = 10.0  # Penalty parameter
        rho_max = 1e6
        rho_update_factor = 10.0
        tau = 0.1  # Tolerance reduction factor
        
        # Initialize Lagrange multipliers for equality constraints
        # Each time step has 2 constraints: contact force limit + joint position limit
        n_constraints_per_timestep = 2
        n_eq_constraints = self.horizon * n_constraints_per_timestep
        lambda_eq = torch.zeros(n_eq_constraints, device=self.device)
        
        # Convergence parameters
        max_outer_iter = 20
        tolerance = 1e-6
        
        for outer_iter in range(max_outer_iter):
            # Tolerance for inner problem
            inner_tol = max(tolerance, tau ** outer_iter)
            
            # Define augmented Lagrangian function
            def augmented_lagrangian(u_var):
                u_tensor = u_var.reshape(self.horizon, control_dim)
                
                # Forward simulation to compute objective and constraints
                x_pred = torch.zeros((self.horizon + 1, len(x_current)), device=self.device)
                x_pred[0] = x_current
                
                total_cost = 0.0
                constraint_violations = []
                
                for t in range(self.horizon):
                    # Get references for current time step
                    current_refs = {
                        'q_d': q_ref[t] if t < q_ref.shape[0] else q_ref[-1],
                        'P_d': contact_references.get('P_d', contact_positions),
                        'P_c': contact_positions,
                        'G_o': system_matrices.get('G_o', torch.eye(n_q, device=self.device))
                    }
                    
                    # Predicted next state using dynamics
                    x_pred_next = self.integrate_dynamics(
                        x_pred[t], u_tensor[t], system_matrices, current_refs, n_q, n_c
                    )
                    
                    # Actual next state (what we want to achieve)
                    x_pred[t + 1] = x_pred_next
                    
                    # Unpack predicted state
                    q_pred, q_d_pred, lambda_pred = self.unpack_state(x_pred[t + 1], n_q, n_c)
                    
                    # Objective function terms
                    motion_cost = torch.norm(q_pred - q_ref[t])**2
                    command_cost = torch.norm(q_d_pred - q_ref[t])**2
                    contact_cost = torch.norm(lambda_pred - lambda_ref[t])**2
                    control_cost = torch.norm(u_tensor[t])**2
                    
                    total_cost += (self.config.w_motion * (motion_cost + command_cost) + 
                                 self.config.w_contact * contact_cost +
                                 self.config.w_smooth * control_cost)
                    
                    # Equality constraints: dynamics violations
                    # Since we're using the dynamics directly, we can add other meaningful constraints
                    # For example, contact force magnitude constraints or state bounds
                    
                    # Add constraint that predicted contact forces should be physically reasonable
                    contact_force_constraint = torch.clamp(torch.norm(lambda_pred) - 100.0, min=0.0)
                    
                    # Add constraint that joint positions stay within reasonable bounds
                    joint_limit_constraint = torch.clamp(torch.norm(q_pred) - 10.0, min=0.0)
                    
                    # Pack constraints for this time step
                    time_constraints = torch.tensor([contact_force_constraint, joint_limit_constraint], 
                                                  device=self.device)
                    constraint_violations.append(time_constraints)
                
                # Stack all constraint violations
                all_constraints = torch.cat(constraint_violations)
                
                # Augmented Lagrangian terms
                # λ^T c(u) + (ρ/2) ||c(u)||^2
                lagrangian_term = torch.dot(lambda_eq, all_constraints)
                penalty_term = (rho / 2) * torch.norm(all_constraints)**2
                
                return total_cost + lagrangian_term + penalty_term
            
            # Solve inner optimization problem using L-BFGS
            optimizer = torch.optim.LBFGS([u_flat], lr=1.0, max_iter=20,
                                        tolerance_grad=inner_tol, tolerance_change=inner_tol)
            
            def closure():
                optimizer.zero_grad()
                loss = augmented_lagrangian(u_flat)
                loss.backward()
                return loss
            
            # Inner optimization loop
            for inner_iter in range(10):
                optimizer.step(closure)
                
                # Check inner convergence
                with torch.no_grad():
                    current_loss = augmented_lagrangian(u_flat)
                    if current_loss.item() < inner_tol:
                        break
            
            # Compute constraint violations at current solution
            with torch.no_grad():
                u_tensor = u_flat.reshape(self.horizon, control_dim)
                
                # Forward simulation to get constraint violations
                x_pred = torch.zeros((self.horizon + 1, len(x_current)), device=self.device)
                x_pred[0] = x_current
                
                constraint_violations = []
                
                for t in range(self.horizon):
                    current_refs = {
                        'q_d': q_ref[t] if t < q_ref.shape[0] else q_ref[-1],
                        'P_d': contact_references.get('P_d', contact_positions),
                        'P_c': contact_positions,
                        'G_o': system_matrices.get('G_o', torch.eye(n_q, device=self.device))
                    }
                    
                    x_pred[t + 1] = self.integrate_dynamics(
                        x_pred[t], u_tensor[t], system_matrices, current_refs, n_q, n_c
                    )
                    
                    # Unpack predicted state for constraint evaluation
                    q_pred, q_d_pred, lambda_pred = self.unpack_state(x_pred[t + 1], n_q, n_c)
                    
                    # Compute same constraint violations as in objective
                    contact_force_constraint = torch.clamp(torch.norm(lambda_pred) - 100.0, min=0.0)
                    joint_limit_constraint = torch.clamp(torch.norm(q_pred) - 10.0, min=0.0)
                    
                    time_constraints = torch.tensor([contact_force_constraint, joint_limit_constraint], 
                                                  device=self.device)
                    constraint_violations.append(time_constraints)
                
                all_constraints = torch.cat(constraint_violations)
                constraint_norm = torch.norm(all_constraints).item()
                
                # Check convergence
                if constraint_norm < tolerance:
                    print(f"Augmented Lagrangian converged in {outer_iter + 1} iterations")
                    break
                
                # Update Lagrange multipliers
                lambda_eq = lambda_eq + rho * all_constraints
                
                # Update penalty parameter if constraints are not reducing
                if constraint_norm > 0.1 * tolerance:
                    rho = min(rho * rho_update_factor, rho_max)
        
        # Return optimal control sequence
        return u_flat.reshape(self.horizon, control_dim).detach()

    def _extract_allegro_references(self, reference_trajectory: Dict[str, torch.Tensor], 
                                   t: int) -> Dict[str, torch.Tensor]:
        """
        Extract references for current time step, compatible with AllegroManipulationProblem.
        
        Args:
            reference_trajectory: Reference trajectory dictionary
            t: Current time step
            
        Returns:
            current_refs: References for current time step
        """
        current_refs = {}
        
        # Extract joint position references
        if 'q_ref' in reference_trajectory:
            q_ref = reference_trajectory['q_ref']
            if len(q_ref.shape) > 1:
                current_refs['q_d'] = q_ref[t] if t < q_ref.shape[0] else q_ref[-1]
            else:
                current_refs['q_d'] = q_ref
        
        # Extract contact position references
        if 'P_d' in reference_trajectory:
            current_refs['P_d'] = reference_trajectory['P_d']
        elif 'contact_positions' in reference_trajectory:
            current_refs['P_d'] = reference_trajectory['contact_positions']
        
        # Extract object state references if available
        if 'theta_ref' in reference_trajectory:
            theta_ref = reference_trajectory['theta_ref']
            if len(theta_ref.shape) > 1:
                current_refs['theta_d'] = theta_ref[t] if t < theta_ref.shape[0] else theta_ref[-1]
            else:
                current_refs['theta_d'] = theta_ref
        
        return current_refs


class AllegroContactDataExtractor:
    """
    Extracts system matrices and contact data from AllegroManipulationProblem preprocessing.
    """
    
    def __init__(self, problem, config: ControllerConfig):
        self.problem = problem
        self.config = config
        self.device = config.device
        
    def extract_system_matrices(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract system matrices from preprocessed contact data.
        
        Args:
            state: Current robot state containing q, q_dot, contact data
            
        Returns:
            system_matrices: Dictionary containing matrices for dynamics equation (24)
        """
        q = state['q']
        n_q = len(q)
        n_contacts = len(self.problem.contact_fingers) if hasattr(self.problem, 'contact_fingers') else 0
        
        # Extract contact Jacobians from preprocessed data
        contact_jacobians = []
        if hasattr(self.problem, 'data') and n_contacts > 0:
            for finger in self.problem.contact_fingers:
                if finger in self.problem.data:
                    # Get contact Jacobian from preprocessing - shape [N, T+1, 3, 16]
                    jac = self.problem.data[finger]['contact_jacobian']
                    if len(jac.shape) == 4:  # [N, T+1, 3, 16]
                        jac = jac[0, -1]  # Take current timestep [3, 16]
                    elif len(jac.shape) == 3:  # [3, 16] already
                        pass
                    else:
                        jac = jac.reshape(3, -1)  # Ensure [3, 16]
                    
                    # Extract relevant joint indices for this finger
                    if hasattr(self.problem, 'joint_index') and finger in self.problem.joint_index:
                        joint_indices = self.problem.joint_index[finger]
                        jac_reduced = jac[:, joint_indices]  # [3, 4] for this finger
                        contact_jacobians.append(jac_reduced)
        
        # Combine Jacobians into overall contact Jacobian
        if contact_jacobians:
            jacobian = torch.block_diag(*contact_jacobians)  # [3*n_c, 4*n_c]
            # Pad to full size if needed
            if jacobian.shape[1] < n_q:
                padding = torch.zeros(jacobian.shape[0], n_q - jacobian.shape[1], device=self.device)
                jacobian = torch.cat([jacobian, padding], dim=1)
        else:
            jacobian = torch.zeros(3 * n_contacts, n_q, device=self.device)
            
        # Create gain matrices
        K_P = self.config.K_P * torch.eye(n_q, device=self.device)
        K_D = self.config.K_D * torch.eye(n_q, device=self.device)
        
        # Create G_o matrix (grasp matrix)
        G_o = jacobian  # Use contact Jacobian as grasp matrix
        
        # Create coupling matrix K_coup for equation (24)
        # K_coup = K_bar + K_bar * G_o^T * (K_bar * G_o * G_o^T)^(-1) * G_o * K_bar
        K_bar = self.config.K_e * torch.eye(3 * n_contacts, device=self.device)
        if n_contacts > 0 and G_o.numel() > 0:
            try:
                G_o_Kbar = K_bar @ G_o  # [3*n_c, n_q]
                middle_term = G_o_Kbar @ G_o.T  # [3*n_c, 3*n_c]
                middle_inv = torch.inverse(middle_term + 1e-6 * torch.eye(3*n_contacts, device=self.device))
                K_coup = K_bar + G_o_Kbar.T @ middle_inv @ G_o_Kbar
            except:
                K_coup = K_bar
        else:
            K_coup = K_bar
            
        # Create stiffness matrix
        stiffness_matrix = torch.eye(3 * n_contacts, device=self.device) * self.config.K_e
        
        # J(q_d) - Jacobian evaluated at desired positions (default to current)
        J_q_d = jacobian.clone()
        
        system_matrices = {
            'jacobian': jacobian,
            'K_P': K_P,
            'K_D': K_D,
            'G_o': G_o,
            'K_coup': K_coup,
            'J_q_d': J_q_d,
            'stiffness_matrix': stiffness_matrix
        }
        
        return system_matrices
    
    def extract_contact_positions(self) -> Dict[str, torch.Tensor]:
        """
        Extract current contact positions from preprocessed data.
        
        Returns:
            contact_data: Dictionary containing contact positions and normals
        """
        contact_positions = []
        contact_normals = []
        
        if hasattr(self.problem, 'data') and hasattr(self.problem, 'contact_fingers'):
            for finger in self.problem.contact_fingers:
                if finger in self.problem.data:
                    # Get closest point on object surface
                    if 'closest_pt_world' in self.problem.data[finger]:
                        pos = self.problem.data[finger]['closest_pt_world']
                        if len(pos.shape) > 1:
                            pos = pos[0, -1]  # Take current timestep
                        contact_positions.append(pos)
                    
                    # Get contact normal
                    if 'contact_normal' in self.problem.data[finger]:
                        normal = self.problem.data[finger]['contact_normal']
                        if len(normal.shape) > 1:
                            normal = normal[0, -1]  # Take current timestep
                        contact_normals.append(normal)
        
        if contact_positions:
            P_c = torch.cat(contact_positions, dim=0)  # [3*n_c]
            normals = torch.stack(contact_normals, dim=0) if contact_normals else torch.zeros_like(contact_positions[0]).unsqueeze(0).repeat(len(contact_positions), 1)  # [n_c, 3]
        else:
            P_c = torch.zeros(0, device=self.device)
            normals = torch.zeros(0, 3, device=self.device)
            
        return {
            'P_c': P_c,
            'contact_normals': normals
        }


class TactileFeedbackController:
    """
    Main tactile-feedback motion-contact tracking controller.
    
    Integrates all components for complete controller functionality.
    """
    
    def __init__(self, config: ControllerConfig, problem=None):
        self.config = config
        self.device = config.device
        self.problem = problem  # AllegroManipulationProblem instance
        
        # Initialize subcomponents
        self.force_motion_model = ForceMotionModel(config)
        self.weighting_determiner = WeightingMatrixDeterminer(config)
        self.mpc_controller = ModelPredictiveController(config)
        
        # Initialize contact data extractor if problem is provided
        if self.problem is not None:
            self.contact_extractor = AllegroContactDataExtractor(problem, config)
        else:
            self.contact_extractor = None
        
        # Controller state
        self.current_mode = "multi_contact"  # "single_contact" or "multi_contact"
        self.contact_history = []
        
        logger.info(f"Initialized TactileFeedbackController with config: {config}")
    
    def set_problem(self, problem):
        """Set the AllegroManipulationProblem instance."""
        self.problem = problem
        self.contact_extractor = AllegroContactDataExtractor(problem, self.config)
    
    def preprocess_with_problem(self, state: Dict[str, torch.Tensor]):
        """
        Use AllegroManipulationProblem preprocessing to populate contact data.
        
        Args:
            state: Current robot state with q, q_dot, contact positions, etc.
        """
        if self.problem is None:
            raise ValueError("No AllegroManipulationProblem instance set. Call set_problem() first.")
        
        # Extract joint positions and object state
        q = state['q']
        if 'theta' in state:
            theta = state['theta']
        else:
            # Extract object DOF from state if available
            if hasattr(self.problem, 'obj_dof') and len(q) > self.problem.robot_dof:
                theta = q[self.problem.robot_dof:self.problem.robot_dof + self.problem.obj_dof]
                q = q[:self.problem.robot_dof]
            else:
                theta = torch.zeros(getattr(self.problem, 'obj_dof', 1), device=self.device)
        
        # Reshape for preprocessing (expects [N, T, dim])
        if len(q.shape) == 1:
            q = q.unsqueeze(0).unsqueeze(0)  # [1, 1, n_joints]
        if len(theta.shape) == 1:
            theta = theta.unsqueeze(0).unsqueeze(0)  # [1, 1, obj_dof]
            
        # Call the problem's preprocessing
        self.problem._preprocess_fingers(q, theta)
    
    def update_with_allegro_integration(self, state: Dict[str, torch.Tensor], 
                                      reference_trajectory: Dict[str, torch.Tensor],
                                      contact_references: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Update controller using AllegroManipulationProblem preprocessing and MPC.
        
        Args:
            state: Current robot state
            reference_trajectory: Desired trajectory over horizon
            contact_references: Desired contact forces over horizon (optional)
            
        Returns:
            control_output: Control commands
        """
        # Use problem preprocessing to populate contact data
        self.preprocess_with_problem(state)
        
        # Extract system matrices from preprocessed data
        if self.contact_extractor is None:
            raise ValueError("No contact extractor available. Set problem first.")
            
        system_matrices = self.contact_extractor.extract_system_matrices(state)
        contact_data = self.contact_extractor.extract_contact_positions()
        
        # Update contact references with current contact positions
        if contact_references is None:
            contact_references = {}
        contact_references.update(contact_data)
        
        # Solve MPC optimization using equation (24) dynamics
        optimal_control = self.mpc_controller.solve_mpc(
            state, reference_trajectory, contact_references, system_matrices
        )
        
        # Extract first control action
        control_action = optimal_control[0]  # First time step
        
        return {
            'joint_torques': control_action,
            'optimal_control_sequence': optimal_control,
            'mpc_horizon': self.config.horizon_length,
            'system_matrices': system_matrices,
            'contact_data': contact_data
        }
    
    def update_with_mpc(self, state: Dict[str, torch.Tensor], 
                       reference_trajectory: Dict[str, torch.Tensor],
                       contact_references: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update controller using Model Predictive Control.
        
        Args:
            state: Current robot state
            reference_trajectory: Desired trajectory over horizon
            contact_references: Desired contact forces over horizon
            
        Returns:
            control_output: Control commands
        """
        # Prepare system matrices for equation (24)
        q = state['q']
        n_contacts = len(state['contact_positions']) // 3 if 'contact_positions' in state else 0
        system_matrices = {
            'jacobian': state.get('jacobian', torch.zeros(3*n_contacts, len(q), device=self.device)),  # J(q)
            'K_P': state.get('K_P', self.config.K_P * torch.eye(len(q), device=self.device)),  # Proportional gain matrix
            'K_D': state.get('K_D', self.config.K_D * torch.eye(len(q), device=self.device)),  # Damping gain matrix
            'G_o': state.get('G_o', torch.eye(len(q), device=self.device)),  # G_o matrix for coupling
            'K_coup': state.get('K_coup', torch.eye(3*n_contacts, device=self.device)),  # Coupling matrix [3*n_c, 3*n_c]
            'J_q_d': state.get('J_q_d', state.get('jacobian', torch.zeros(3*n_contacts, len(q), device=self.device))),  # J(q_d) - Jacobian evaluated at q_d (defaults to J(q))
            'stiffness_matrix': state.get('stiffness_matrix', 
                                        torch.eye(3*n_contacts, device=self.device))
        }
        
        # Solve MPC optimization
        optimal_control = self.mpc_controller.solve_mpc(
            state, reference_trajectory, contact_references, system_matrices
        )
        
        # Extract first control action
        control_action = optimal_control[0]  # First time step
        
        return {
            'joint_torques': control_action,
            'optimal_control_sequence': optimal_control,
            'mpc_horizon': self.config.horizon_length
        }
    
    def set_mode(self, mode: str):
        """Set controller mode (single_contact or multi_contact)."""
        if mode not in ["single_contact", "multi_contact"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.current_mode = mode
        logger.info(f"Controller mode set to: {mode}")
    
    def update_single_contact(self, state: Dict[str, torch.Tensor], 
                            references: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update controller for single contact case.
        
        Args:
            state: Current robot state
            references: Desired references
            
        Returns:
            control_output: Control commands
        """
        # Extract state
        q = state['q']
        p_c = state['contact_position']  # Current contact position [3]
        J = state['jacobian']  # Contact Jacobian
        
        # Extract references
        p_d = references['desired_contact_position']  # [3]
        q_d = references['desired_joint_positions']  # [n_joints]
        q_d_dot = references['desired_joint_velocities']  # [n_joints]
        
        # Compute contact force using force-motion model
        lambda_ext = self.force_motion_model.single_contact_force(p_d, p_c)
        
        # Compute joint accelerations
        q_ddot = self.force_motion_model.joint_motion_single(
            q_d, q, q_d_dot, J, lambda_ext
        )
        
        # Convert to control commands (torques)
        # τ = M(q) * q_ddot + C(q, q_dot) * q_dot + g(q)
        # Simplified: assuming unit mass matrix
        tau = q_ddot  # In practice, would include full dynamics
        
        return {
            'joint_torques': tau,
            'predicted_contact_force': lambda_ext,
            'joint_accelerations': q_ddot
        }
    
    def update_multi_contact(self, state: Dict[str, torch.Tensor], 
                           references: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update controller for multi-contact case.
        
        Args:
            state: Current robot state
            references: Desired references
            
        Returns:
            control_output: Control commands
        """
        # Extract state
        q = state['q']
        q_dot = state['q_dot']
        P_c = state['contact_positions']  # Current contact positions [n_c*3]
        contact_forces = state['contact_forces']  # [n_c, 3]
        contact_normals = state['contact_normals']  # [n_c, 3]
        J = state['jacobian']  # Contact Jacobian
        K_matrix = state.get('stiffness_matrix', torch.eye(len(P_c), device=self.device))
        
        # Extract references  
        P_d = references['desired_contact_positions']  # [n_c*3]
        q_d = references['desired_joint_positions']  # [n_joints]
        q_d_dot = references['desired_joint_velocities']  # [n_joints]
        
        # Compute contact forces using force-motion model
        A_ext = self.force_motion_model.multi_contact_force(P_d, P_c, K_matrix)
        
        # Determine weighting matrices based on contact classification
        W_A, W_P = self.weighting_determiner.compute_weighting_matrices(
            contact_forces, contact_normals
        )
        
        # Apply weighting to contact forces and positions
        weighted_forces = W_A @ A_ext
        weighted_positions = W_P @ (P_d - P_c)
        
        # Compute joint accelerations
        q_ddot = self.force_motion_model.joint_motion_multi(
            q_d, q, q_d_dot, J, weighted_forces
        )
        
        # Convert to control commands
        tau = q_ddot  # Simplified
        
        return {
            'joint_torques': tau,
            'predicted_contact_forces': A_ext,
            'weighted_contact_forces': weighted_forces,
            'weighted_position_errors': weighted_positions,
            'weighting_matrix_force': W_A,
            'weighting_matrix_position': W_P,
            'joint_accelerations': q_ddot
        }
    
    def update(self, state: Dict[str, torch.Tensor], 
              references: Dict[str, torch.Tensor],
              use_mpc: bool = False,
              use_allegro_integration: bool = True) -> Dict[str, torch.Tensor]:
        """
        Main update function for the controller.
        
        Args:
            state: Current robot state
            references: Desired references (can be single step or trajectory)
            use_mpc: Whether to use MPC (requires trajectory references)
            use_allegro_integration: Whether to use AllegroManipulationProblem integration
            
        Returns:
            control_output: Control commands and auxiliary information
        """
        # Convert numpy arrays to tensors if needed
        state = self._ensure_tensors(state)
        references = self._ensure_tensors(references)
        
        if use_allegro_integration and self.problem is not None:
            # Use integrated approach with AllegroManipulationProblem
            reference_trajectory = references.get('trajectory', references)
            contact_references = references.get('contact_trajectory', {})
            return self.update_with_allegro_integration(state, reference_trajectory, contact_references)
            
        elif use_mpc:
            # MPC mode requires trajectory references
            reference_trajectory = references.get('trajectory', references)
            contact_references = references.get('contact_trajectory', references)
            return self.update_with_mpc(state, reference_trajectory, contact_references)
        
        elif self.current_mode == "single_contact":
            return self.update_single_contact(state, references)
        
        elif self.current_mode == "multi_contact":
            return self.update_multi_contact(state, references)
        
        else:
            raise ValueError(f"Invalid controller mode: {self.current_mode}")
    
    def _ensure_tensors(self, data_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        tensor_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                tensor_dict[key] = torch.from_numpy(value).float().to(self.device)
            elif isinstance(value, torch.Tensor):
                tensor_dict[key] = value.to(self.device)
            else:
                tensor_dict[key] = value
        return tensor_dict
    
    def get_controller_info(self) -> Dict[str, any]:
        """Get information about current controller state."""
        return {
            'mode': self.current_mode,
            'config': self.config,
            'device': self.device,
            'mpc_horizon': self.config.horizon_length,
            'force_threshold': self.config.force_threshold
        }


# Utility functions for common operations
def install_optimization_packages():
    """
    Install additional optimization packages for enhanced MPC performance.
    
    This function helps install recommended packages for different solvers:
    - OSQP: Already available (fast QP solver)
    - SciPy: Already available (robust optimization)
    - IPOPT: For industrial-grade nonlinear optimization
    - CasADi: For advanced MPC with automatic differentiation
    """
    try:
        import subprocess
        import sys
        
        print("Installing additional optimization packages...")
        
        # IPOPT (requires cyipopt)
        try:
            import cyipopt
            print("✓ IPOPT already available")
        except ImportError:
            print("Installing IPOPT (cyipopt)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cyipopt"])
            
        # CasADi for advanced MPC
        try:
            import casadi
            print("✓ CasADi already available")
        except ImportError:
            print("Installing CasADi...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "casadi"])
            
        print("✓ All optimization packages installed successfully!")
        
    except Exception as e:
        print(f"Warning: Could not install additional packages: {e}")
        print("You can manually install with:")
        print("  pip install cyipopt casadi")


def create_stiffness_matrix(contact_stiffnesses: List[float], device: str = "cpu") -> torch.Tensor:
    """
    Create block diagonal stiffness matrix for multiple contacts.
    
    Args:
        contact_stiffnesses: List of stiffness values for each contact
        device: PyTorch device
        
    Returns:
        K_matrix: Block diagonal stiffness matrix [3*n_c, 3*n_c]
    """
    n_contacts = len(contact_stiffnesses)
    K_matrix = torch.zeros((3 * n_contacts, 3 * n_contacts), device=device)
    
    for i, stiffness in enumerate(contact_stiffnesses):
        start_idx = 3 * i
        end_idx = 3 * (i + 1)
        K_matrix[start_idx:end_idx, start_idx:end_idx] = stiffness * torch.eye(3, device=device)
    
    return K_matrix


def compute_contact_jacobian(joint_positions: torch.Tensor, 
                           contact_points: torch.Tensor,
                           robot_model) -> torch.Tensor:
    """
    Compute contact Jacobian matrix.
    
    This is a placeholder - in practice would use robot-specific kinematics.
    
    Args:
        joint_positions: Current joint positions
        contact_points: Contact point positions
        robot_model: Robot kinematic model
        
    Returns:
        J: Contact Jacobian matrix
    """
    # Placeholder implementation
    n_joints = len(joint_positions)
    n_contacts = len(contact_points)
    J = torch.randn((3 * n_contacts, n_joints), device=joint_positions.device)
    return J


def create_allegro_compatible_state(problem, q_current: torch.Tensor, 
                                  q_dot_current: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Create a state dictionary compatible with both TactileFeedbackController and AllegroManipulationProblem.
    
    Args:
        problem: AllegroManipulationProblem instance
        q_current: Current joint positions
        q_dot_current: Current joint velocities (optional)
        
    Returns:
        state: State dictionary with all necessary fields
    """
    device = q_current.device
    
    if q_dot_current is None:
        q_dot_current = torch.zeros_like(q_current)
    
    # Split joint positions into robot and object parts
    if hasattr(problem, 'robot_dof'):
        robot_dof = problem.robot_dof
        q_robot = q_current[:robot_dof]
        if len(q_current) > robot_dof:
            theta = q_current[robot_dof:robot_dof + getattr(problem, 'obj_dof', 1)]
        else:
            theta = torch.zeros(getattr(problem, 'obj_dof', 1), device=device)
    else:
        q_robot = q_current
        theta = torch.zeros(1, device=device)
    
    # Initialize contact data (will be populated by preprocessing)
    n_contacts = len(getattr(problem, 'contact_fingers', []))
    
    state = {
        'q': q_current,
        'q_dot': q_dot_current,
        'q_robot': q_robot,
        'theta': theta,
        'contact_positions': torch.zeros(3 * n_contacts, device=device),
        'contact_forces': torch.zeros(n_contacts, 3, device=device),
        'contact_normals': torch.zeros(n_contacts, 3, device=device),
        'jacobian': torch.zeros(3 * n_contacts, len(q_robot), device=device),
    }
    
    return state


if __name__ == "__main__":
    # Example usage with AllegroManipulationProblem integration
    
    print("=== Tactile Feedback Controller with Allegro Integration ===")
    
    # Configuration
    config = ControllerConfig(
        K_e=1000.0,
        K_r=100.0,
        K_P=1000.0,
        force_threshold=0.5,
        horizon_length=10,
        mpc_solver='lbfgs'
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
    print("- 'lbfgs': L-BFGS (recommended for most applications)")
    print("- 'osqp': OSQP quadratic programming (fast for real-time)")
    print("- 'scipy': SciPy SLSQP (robust for constrained problems)")
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
    
    print("\nReady for integration with AllegroManipulationProblem!") 