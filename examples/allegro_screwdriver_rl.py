from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
from ccai.utils.allegro_utils import state2ee_pos, partial_to_full_state
import pytorch_kinematics as pk

import os
import sys
import pathlib
import yaml
import torch
import numpy as np
import time
from functools import partial
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

from allegro_screwdriver import AllegroScrewdriver
import copy
from torch_cg import cg_batch

torch.autograd.set_detect_anomaly(True)

sys.path.append('..')

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

sys.stdout = open('./examples/logs/allegro_screwdriver_td3.log', 'w', 1)
sys.stdout.reconfigure(line_buffering=True)

class AllegroScrewdriverRLEnv:
    """Custom environment for the Allegro Hand screwdriver turning task"""
    
    def __init__(self, config):
        # Environment configuration
        self.config = config
        self.device = config['sim_device']
        self.num_fingers = len(config['fingers'])
        self.control_dim = 4 * self.num_fingers  # 4 joints per finger
        self.obj_dof = 3
        
        # Create the underlying environment
        self.env = AllegroScrewdriverTurningEnv(
            1, control_mode='joint_impedance',
            use_cartesian_controller=False,
            viewer=config['visualize'],
            steps_per_action=60,
            friction_coefficient=config['friction_coefficient'] * 2.5,
            device=self.device,
            video_save_path=img_save_dir if config['visualize'] else None,
            joint_stiffness=config['kp'],
            fingers=config['fingers'],
            gradual_control=False,
            gravity=True,
            randomize_obj_start=config.get('randomize_obj_start', False),
            randomize_rob_start=config.get('randomize_rob_start', False),
        )
        
        # Get simulation handles
        self.sim, self.gym, self.viewer = self.env.get_sim()
        
        # Define action bounds
        self.action_low = -0.1
        self.action_high = 0.1
        self.action_shape = (self.control_dim,)
        
        # Define observation shape
        self.obs_dim = self.control_dim + self.obj_dof  # Removed *2 as we're no longer including angular velocities
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = config.get('max_episode_steps', 100)
        
        # Previous state for calculating deltas - no longer needed
        # self.prev_orientation = None
        
        # Set up kinematic chain for visualization and state extraction
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
        ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }
        self.ee_names = ee_names
        
        chain = pk.build_chain_from_urdf(open(asset).read())
        frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]
        frame_indices = torch.tensor(frame_indices)
        self.state2ee_pos = partial(
            state2ee_pos, 
            fingers=config['fingers'], 
            chain=chain, 
            frame_indices=frame_indices,
            world_trans=self.env.world_trans
        )
        self.chain = chain.to(device='cuda:0')
        
        # Define pregrasp positions for different fingers
        self.pregrasp_positions = {
            'index': torch.tensor([0.1, 0.6, 0.6, 0.6]),
            'middle': torch.tensor([-0.1, 0.5, 0.9, 0.9]),
            'ring': torch.tensor([0.0, 0.5, 0.65, 0.65]),
            'thumb': torch.tensor([1.2, 0.3, 0.3, 1.2])
        }
    
    def _get_obs(self):
        state_dict = self.env.get_state()
        state = state_dict['q'].reshape(-1)
        
        # Get joint positions
        joint_pos = state[:self.control_dim]
        
        # Get object orientation
        orientation = state[self.control_dim:self.control_dim + self.obj_dof]
        
        # Combine into observation (without angular velocity)
        obs = torch.cat([joint_pos, orientation], dim=0)
        return obs
    
    def reset(self):
        self.env.reset()
        self.episode_steps = 0
        print(f"Episode reset - starting new episode")
        
        # # Execute pregrasp
        # print("Executing pregrasp...")
        
        # # Construct pregrasp position based on active fingers
        # pregrasp_action = torch.cat([self.pregrasp_positions[finger] for finger in self.config['fingers']])
        # pregrasp_action = pregrasp_action.to(device=self.device).reshape(1, -1)
        
        # # Execute pregrasp action - move fingers to initial grasp position
        # for _ in range(5):  # Execute multiple times to ensure stable position
        #     self.env.step(pregrasp_action)
            
        # print("Pregrasp completed")

        self.initial_yaw = self._get_obs()[-1]
        
        return self._get_obs()
    
    def step(self, action):
        # Convert action to target joint positions
        state_dict = self.env.get_state()
        current_pos = state_dict['q'].reshape(-1)[:self.control_dim]
        target_pos = current_pos + action.cpu()
        
        # Step the environment
        self.env.step(target_pos.reshape(1, -1))
        self.episode_steps += 1
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._compute_reward(obs, action)
        
        # Extract orientation
        orientation = obs[self.control_dim:self.control_dim + self.obj_dof]
        roll, pitch, yaw = orientation
        
        # Check if episode is done due to max steps or dropping the screwdriver
        done = (self.episode_steps >= self.max_episode_steps) or (abs(roll) > 0.2) or (abs(pitch) > 0.2)
        
        termination_reason = ""
        if done:
            if self.episode_steps >= self.max_episode_steps:
                termination_reason = "reached max steps"
            elif abs(roll) > 0.2 or abs(pitch) > 0.2:
                termination_reason = f"dropped screwdriver (roll={roll:.3f}, pitch={pitch:.3f})"
            print(f"Episode terminated: {termination_reason}, final reward: {reward:.3f}, step: {self.episode_steps}")
        
        return obs, reward, done

    def _compute_reward(self, obs, action):
        # Extract orientation
        orientation = obs[self.control_dim:self.control_dim + self.obj_dof]
        roll, pitch, yaw = orientation
        
        # Reward turning (yaw change)
        turn_reward = -(yaw - self.initial_yaw) * 1.0
        self.initial_yaw = yaw  # Update initial yaw for next step
        # Penalize deviation from upright position
        # upright_penalty = (roll.abs() + pitch.abs()) * 1.0
        
        # Calculate total reward
        reward = turn_reward #- upright_penalty

        # Compute constraint violation penalty

        
        return reward

    def close(self):
        if hasattr(self, 'env'):
            if hasattr(self.env, 'close'):
                self.env.close()

# Import necessary ODE integration package
from torchdiffeq import odeint

# Neural network for the actor (deterministic policy)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, force_dim, action_low, action_high, hidden_dim=256, constraint_problem=None):
        super(ActorNetwork, self).__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.force_dim = force_dim
        self.combined_dim_no_force = input_dim + output_dim
        self.combined_dim = input_dim + output_dim + force_dim

        self.action_low = action_low
        self.action_high = action_high
        
        # Vector field network that models the dynamics of the state-action system
        self.vector_field = nn.Sequential(
            nn.Linear(self.combined_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.combined_dim*2)
        )

        # Predicts initial x dot for ODE integration
        self.xdot0 = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.combined_dim)
        )
        
        # Store constraint problem for projection
        self.constraint_problem = constraint_problem

        self.fingers = ['index', 'middle', 'thumb']
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999])
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227])
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999])
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162])
        joint_min = {'index': index_x_min, 'middle': index_x_min, 'ring': index_x_min, 'thumb': thumb_x_min}
        joint_max = {'index': index_x_max, 'middle': index_x_max, 'ring': index_x_max, 'thumb': thumb_x_max}
        self.x_max = torch.cat([joint_max[finger] for finger in self.fingers])
        self.x_min = torch.cat([joint_min[finger] for finger in self.fingers])
        self.robot_joint_x_max = torch.cat([joint_max[finger] for finger in self.fingers]).to('cuda:0')
        self.robot_joint_x_min = torch.cat([joint_min[finger] for finger in self.fingers]).to('cuda:0')

        self.robot_joint_x_min = torch.cat([self.robot_joint_x_min, torch.tensor([-torch.pi, -torch.pi, -torch.pi]).to('cuda:0')])
        self.robot_joint_x_max = torch.cat([self.robot_joint_x_max, torch.tensor([torch.pi, torch.pi, torch.pi]).to('cuda:0')])

        self.constrained_dynamics = partial(self.dynamics, constrained=True)
        self.unconstrained_dynamics = partial(self.dynamics, constrained=False)
        
        # Add timing metrics dictionary
        self.timing_metrics = {
            'total_time': [],
            'clamp_setup': [],
            'preprocessing': [],
            'constraint_computation': [],
            'masks_creation': [],
            'matrix_solving': [],
            'projection_application': [],
            'calls': 0
        }
        
    def forward(self, x, constrained=True):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Instead of using stochastic policy, we use a deterministic one for TD3
        # Generate zero vectors instead of random normal ones
        z = torch.zeros(batch_size, self.output_dim+self.force_dim, device=x.device)
        
        # Concatenate state and zero vector
        x_z = torch.cat([x, z], dim=-1)

        # x_z_dot0 = self.xdot0(x_z)
        x_z_dot0 = torch.zeros_like(x_z)

        integration_input = torch.cat([x_z, x_z_dot0], dim=-1)
        
        # Integrate ODE from t=0 to t=1
        integration_times = torch.tensor([0.0, 1.0], device=x.device)

        # Unsqueeze for CSVTO
        if constrained:
            dynamics = self.constrained_dynamics
            integration_input = integration_input.unsqueeze(1)  # Add time dimension for odeint
        else:
            dynamics = self.unconstrained_dynamics
        trajectory = odeint(dynamics, integration_input, integration_times, method='rk4', options={'step_size': .3})
        if constrained:
            trajectory = trajectory.squeeze(1)
        trajectory = trajectory.permute(1, 0, 2)  # Change to (batch_size, time_steps, dim)
        trajectory[..., self.combined_dim_no_force:self.combined_dim] = torch.clamp(trajectory[..., self.combined_dim_no_force:self.combined_dim].clone(), -2, 2)

        if constrained:
            self.constraint_problem.start = trajectory[0, 0, :15]
            self.constraint_problem._preprocess(trajectory[:, 1:2], contact_constraint_only=True)

            u_hat = self.constraint_problem.solve_for_u_hat(trajectory[:, 1:2])
            
            # Extract the final point of the trajectory
            final_state = trajectory[0, -1]

            final_state[self.input_dim:self.combined_dim_no_force] = u_hat.squeeze(1)
            
            # Return deterministic action and force
            action = final_state[self.input_dim:self.combined_dim_no_force]
            action = torch.clamp(action, self.action_low, self.action_high)
            force = final_state[self.combined_dim_no_force:self.combined_dim]
        else:
            final_state = trajectory[:, -1]
            action_ = final_state[:, self.input_dim:self.combined_dim_no_force]
            action = torch.clamp(action_.clone(), self.action_low, self.action_high)
            force_ = final_state[:, self.combined_dim_no_force:self.combined_dim]
            force = torch.clamp(force_.clone(), -2, 2)
        return action, force


    def dynamics(self, t, x, constrained=True):
        """ODE function representing the vector field"""
        # Apply vector field
        dx = self.vector_field(x)
        
        # Apply constraint projection if constraint_problem is available
        if self.constraint_problem is not None and constrained:
            dx = self.project_vector_field(x, dx)
            
        return dx
    
    def compute_constraint_violation(self, obs, action, force):
        
        obs = obs.reshape(1, 1, -1)
        action = action.reshape(1, 1, -1)
        force = force.reshape(1, 1, -1)
        decision_variable = torch.cat([obs, action,force], dim=-1)
        self.constraint_problem._preprocess(decision_variable, projected_diffusion=True)
        C, _, _ = self.constraint_problem.combined_constraints(
            decision_variable,
            compute_hess=False,
            compute_grads=False,
            projected_diffusion=True,
            include_slack=False,
            compute_inequality=True,
            include_deriv_grad=False
        )
        return C.pow(2).sum()   

    def project_vector_field(self, x, dx):
        """Project vector field to satisfy constraints"""
        start_time = time.time()
        self.timing_metrics['calls'] += 1
        
        # Phase 1: Initial setup and clipping
        clamp_start = time.time()
        batch_size = x.shape[0]
        x_ = x.clone()
        x_[:, :, :self.input_dim] = torch.clamp(x[:, :, :self.input_dim], self.robot_joint_x_min.to('cuda:0'), self.robot_joint_x_max.to('cuda:0'))
        x_[:, :, self.input_dim:self.combined_dim_no_force] = torch.clamp(x[:, :, self.input_dim:self.combined_dim_no_force], self.action_low, self.action_high)
        x_[:, :, self.combined_dim_no_force:] = torch.clamp(x[:, :, self.combined_dim_no_force:], -2, 2)
        x = x_
        clamp_end = time.time()
        self.timing_metrics['clamp_setup'].append(clamp_end - clamp_start)
        
        # Parameters for projection
        Kh = 3
        Ktheta = 1

        # Phase 2: Preprocessing with constraint problem
        preprocess_start = time.time()
        decision_variable = x[:, :, :self.combined_dim]
        # Apply projection
        self.constraint_problem.start = decision_variable[0, 0, :self.input_dim]
        self.constraint_problem._preprocess(decision_variable, dxu=x[:, :, self.combined_dim:], projected_diffusion=True)
        preprocess_end = time.time()
        self.timing_metrics['preprocessing'].append(preprocess_end - preprocess_start)
        
        # Phase 3: Computing constraints and derivatives
        constraint_start = time.time()
        # Get constraints and their derivatives
        C, dC, _ = self.constraint_problem.combined_constraints(
            decision_variable,
            compute_hess=False,
            projected_diffusion=True,
            include_slack=False,
            compute_inequality=True,
            include_deriv_grad=True
        )
        constraint_end = time.time()
        self.timing_metrics['constraint_computation'].append(constraint_end - constraint_start)
        
        # Phase 4: Creating masks for active constraints
        mask_start = time.time()
        # Create masks for active constraints
        dC_mask = ~(dC == 0).all(dim=-1)
        g_dim = self.constraint_problem.dg
        dg_mask = dC_mask[:, :g_dim]
        g = C[:, :g_dim]
        dg = dC[:, :g_dim]
        dh_mask = dC_mask[:, g_dim:]
        h = C[:, g_dim:]
        dh = dC[:, g_dim:]
        
        # Check which inequality constraints are active
        I_p = h >= 0
        
        # Extract active constraints
        dg_masked = dg[dg_mask].reshape(batch_size, -1, dC.shape[-1])
        h_masked = h[dh_mask].reshape(batch_size, -1)
        dh_masked = dh[dh_mask].reshape(batch_size, -1, dC.shape[-1])
        
        h_I_p = h_masked[I_p].reshape(batch_size, -1)
        dh_I_p = dh_masked[I_p].reshape(batch_size, -1, dC.shape[-1])
        
        g_masked = g[dg_mask].reshape(batch_size, -1)
        
        # Combine active constraints
        h_bar = torch.cat((g_masked, h_I_p), dim=1)
        dh_bar = torch.cat((dg_masked, dh_I_p), dim=1)
        
        # Add small damping factor for numerical stability
        eye = torch.eye(dh_bar.shape[1]).repeat(batch_size, 1, 1).to(device=C.device)
        eye_0 = eye.clone() * 1e-6
        mask_end = time.time()
        self.timing_metrics['masks_creation'].append(mask_end - mask_start)
        
        # Phase 5: Matrix operations and solving
        matrix_start = time.time()
        # Compute projection matrix
        dCdCT = dh_bar @ dh_bar.permute(0, 2, 1)
        dCdCT = dCdCT * Ktheta
        
        # Solve the system
        try:
            dCdCT_inv = torch.linalg.solve(dCdCT + eye_0, eye)
            if torch.any(torch.isnan(dCdCT_inv)):
                raise ValueError('nan in inverse')
        except Exception as e:
            print(f"Using CG due to error: {e}")
            A_bmm = lambda x: dCdCT @ x
            dCdCT_inv, _ = cg_batch(A_bmm, eye, verbose=False)
        matrix_end = time.time()
        self.timing_metrics['matrix_solving'].append(matrix_end - matrix_start)
        
        # Phase 6: Final projection and clipping
        projection_start = time.time()
        # Project the vector field
        dh_bar_dJ = torch.bmm(dh_bar, dx.permute(0, 2, 1))
        second_term = Ktheta * dh_bar_dJ - Kh * h_bar.unsqueeze(-1)
        pi = -dCdCT_inv @ second_term
        
        # Ensure inequality constraints are properly handled
        pi[:, g_masked.shape[-1]:] = torch.clamp(pi[:, g_masked.shape[-1]:], min=0)
        
        # Apply projection
        projected_dx = dx + (dh_bar.permute(0, 2, 1) @ pi).squeeze(-1)

        x_ = x.clone()
        x_[:, :, :self.input_dim] = torch.clamp(x[:, :, :self.input_dim], self.robot_joint_x_min.to('cuda:0'), self.robot_joint_x_max.to('cuda:0'))
        x_[:, :, self.input_dim:self.combined_dim_no_force] = torch.clamp(x[:, :, self.input_dim:self.combined_dim_no_force], self.action_low, self.action_high)
        x_[:, :, self.combined_dim_no_force:] = torch.clamp(x[:, :, self.combined_dim_no_force:], -2, 2)

        x = x_
        projection_end = time.time()
        self.timing_metrics['projection_application'].append(projection_end - projection_start)
        
        # Record total time
        end_time = time.time()
        self.timing_metrics['total_time'].append(end_time - start_time)
        
        # Print timing summary every 1 calls
        if self.timing_metrics['calls'] % 1 == 0:
            self._print_timing_summary()
        
        return projected_dx
    
    def _print_timing_summary(self):
        """Print summary of timing metrics for project_vector_field"""
        calls = self.timing_metrics['calls']
        
        # Calculate average times for each phase
        avg_total = np.mean(self.timing_metrics['total_time'][-100:]) * 1000  # Convert to ms
        avg_clamp = np.mean(self.timing_metrics['clamp_setup'][-100:]) * 1000
        avg_preprocess = np.mean(self.timing_metrics['preprocessing'][-100:]) * 1000
        avg_constraint = np.mean(self.timing_metrics['constraint_computation'][-100:]) * 1000
        avg_masks = np.mean(self.timing_metrics['masks_creation'][-100:]) * 1000
        avg_matrix = np.mean(self.timing_metrics['matrix_solving'][-100:]) * 1000
        avg_projection = np.mean(self.timing_metrics['projection_application'][-100:]) * 1000
        
        print(f"\n===== project_vector_field Timing Analysis (call {calls}) =====")
        print(f"Total time:            {avg_total:.2f} ms (100%)")
        print(f"1. Initial setup:      {avg_clamp:.2f} ms ({avg_clamp/avg_total*100:.1f}%)")
        print(f"2. Preprocessing:      {avg_preprocess:.2f} ms ({avg_preprocess/avg_total*100:.1f}%)")
        print(f"3. Constraint comp:    {avg_constraint:.2f} ms ({avg_constraint/avg_total*100:.1f}%)")
        print(f"4. Masks creation:     {avg_masks:.2f} ms ({avg_masks/avg_total*100:.1f}%)")
        print(f"5. Matrix operations:  {avg_matrix:.2f} ms ({avg_matrix/avg_total*100:.1f}%)")
        print(f"6. Final projection:   {avg_projection:.2f} ms ({avg_projection/avg_total*100:.1f}%)")
        
        # Check for outliers
        max_time = max(self.timing_metrics['total_time'][-100:]) * 1000
        if max_time > avg_total * 2:
            print(f"WARN: Max execution time ({max_time:.2f} ms) is {max_time/avg_total:.1f}x the average")
        
        # Clean up old metrics to prevent memory growth
        if len(self.timing_metrics['total_time']) > 1000:
            self.timing_metrics['total_time'] = self.timing_metrics['total_time'][-500:]
            self.timing_metrics['clamp_setup'] = self.timing_metrics['clamp_setup'][-500:]
            self.timing_metrics['preprocessing'] = self.timing_metrics['preprocessing'][-500:]
            self.timing_metrics['constraint_computation'] = self.timing_metrics['constraint_computation'][-500:]
            self.timing_metrics['masks_creation'] = self.timing_metrics['masks_creation'][-500:]
            self.timing_metrics['matrix_solving'] = self.timing_metrics['matrix_solving'][-500:]
            self.timing_metrics['projection_application'] = self.timing_metrics['projection_application'][-500:]

# Neural network for the critic (Q-value function) - TD3 uses twin critics
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        # Concatenate state and action
        sa = torch.cat([state, action], 1)
        
        # Return both Q-values
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state, action):
        # Return only Q1 value (used for actor updates)
        sa = torch.cat([state, action], 1)
        return self.q1(sa)
    
    def q2_forward(self, state, action):
        # Return only Q2 value (used for actor updates)
        sa = torch.cat([state, action], 1)
        return self.q2(sa)

# SumTree implementation for efficient priority sampling
class SumTree:
    def __init__(self, capacity, device='cpu'):
        """
        Initialize SumTree for efficient priority-based sampling
        
        Args:
            capacity: Maximum capacity of the buffer
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        
        # Tree structure to store priorities
        # Size of the tree is 2*capacity - 1
        # Leaf nodes (priorities): capacity
        # Internal nodes: capacity - 1
        self.tree = torch.zeros(2 * capacity - 1, dtype=torch.float32, device=device)
        
        # Data write index
        self.data_pointer = 0
    
    def update(self, idx, priority):
        """Update priority in the tree"""
        # Convert data index to leaf index
        tree_idx = idx + self.capacity - 1
        
        # Change in priority
        change = priority - self.tree[tree_idx]
        
        # Update leaf node
        self.tree[tree_idx] = priority
        
        # Propagate change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def add(self, priority):
        """Add new priority to tree"""
        # Get leaf index for data
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Update tree
        self.update(self.data_pointer, priority)
        
        # Update data pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return tree_idx - (self.capacity - 1)
    
    def get_leaf(self, v):
        """
        Get leaf node and corresponding data index from value v
        v is a value between 0 and the total sum of priorities
        """
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # If we reach leaf nodes, end search
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Otherwise, search left or right child node
            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = leaf_idx - (self.capacity - 1)
        
        return leaf_idx, data_idx
    
    def total_priority(self):
        """Return the sum of all priorities (root node value)"""
        return self.tree[0]

# Updated Replay buffer for prioritized experience replay
class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu', alpha=0.6, beta=0.4, beta_annealing=0.001, epsilon=1e-6):
        self.max_size = max_size
        self.size = 0
        self.device = device
        
        # PER hyperparameters
        self.alpha = alpha  # How much prioritization to use (0=none, 1=full)
        self.beta = beta    # Importance sampling correction (0=no correction, 1=full correction)
        self.beta_annealing = beta_annealing  # How much to increase beta each sampling
        self.epsilon = epsilon  # Small constant to ensure all priorities > 0
        
        # Create SumTree for efficient sampling
        self.sum_tree = SumTree(max_size, device)
        
        # Maximum priority for new transitions
        self.max_priority = 1.0
        
        # Storage for transitions
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
    
    def add(self, state, action, next_state, reward, done):
        # Convert to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.FloatTensor([reward]).to(self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.FloatTensor([done]).to(self.device)
        
        # Get index where to store new transition
        idx = self.sum_tree.add(self.max_priority ** self.alpha)
        
        # Store transition
        self.state[idx] = state
        self.action[idx] = action
        self.next_state[idx] = next_state
        self.reward[idx] = reward
        self.done[idx] = done
        
        # Update size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """Sample a batch based on priorities"""
        batch_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        tree_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        weights = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        # Get segment interval
        segment = self.sum_tree.total_priority() / batch_size
        
        # Increase beta over time to reduce bias
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        # Calculate min priority for importance sampling weights
        min_prob = torch.min(self.sum_tree.tree[self.sum_tree.capacity - 1:self.sum_tree.capacity - 1 + self.size]) / self.sum_tree.total_priority()
        if min_prob <= 0:
            min_prob = self.epsilon
        
        # Sample batch and calculate weights
        for i in range(batch_size):
            # Sample uniformly within each segment
            a, b = segment * i, segment * (i + 1)
            value = torch.rand(1, device=self.device) * (b - a) + a
            
            # Get index from sum tree
            leaf_idx, data_idx = self.sum_tree.get_leaf(value.item())
            
            # Store indices
            tree_indices[i] = leaf_idx
            batch_indices[i] = data_idx
            
            # Calculate importance sampling weight
            priority = self.sum_tree.tree[leaf_idx]
            sampling_prob = priority / self.sum_tree.total_priority()
            weights[i] = (sampling_prob * self.size) ** (-self.beta)
        
        # Normalize weights
        weights = weights / torch.max(weights)
        
        return (
            self.state[batch_indices],
            self.action[batch_indices],
            self.next_state[batch_indices],
            self.reward[batch_indices],
            self.done[batch_indices],
            batch_indices,
            weights
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities in the sum tree"""
        priorities = priorities.detach().cpu().numpy()  # Convert to numpy for easier iteration
        
        for idx, priority in zip(indices, priorities):
            # Add small constant to avoid zero priority
            priority = max(priority, self.epsilon)
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority.item())
            
            # Update sum tree with priority^alpha
            self.sum_tree.update(idx, priority.item() ** self.alpha)

# Replay buffer for off-policy learning
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
    
    def add(self, state, action, next_state, reward, done):
        # Convert to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.FloatTensor([reward]).to(self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.FloatTensor([done]).to(self.device)
        
        # Store transition
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.done[ind]
        )

# Import necessary modules for constraint projection

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        force_dim,
        action_low,
        action_high,
        device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        buffer_size=int(1e6),
        use_constraints=False,
        env=None,
        use_per=True,  # Flag to use PER
        per_alpha=0.6,  # PER alpha parameter (prioritization strength)
        per_beta=0.4,   # PER beta parameter (importance sampling)
        per_beta_annealing=0.0001  # Beta annealing rate
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_low = action_low
        self.action_high = action_high
        self.total_it = 0
        
        self.use_constraints = use_constraints
        self.use_per = use_per  # Whether to use PER
        
        # Create constraint problem if constraints are enabled
        constraint_problem = None
        if use_constraints and env is not None:
            constraint_problem = AllegroScrewdriver(
                start=torch.zeros(state_dim).to('cuda:0'),
                goal=torch.zeros(3).to('cuda:0'),  # Goal will be set dynamically
                T=1,
                chain=env.chain,
                device=torch.device('cuda:0'),
                object_asset_pos=env.env.table_pose,
                object_location=torch.tensor([0, 0, 1.205]).to('cuda:0'),
                object_type='screwdriver',
                world_trans=env.env.world_trans,
                contact_fingers=['index', 'middle', 'thumb'],
                obj_dof=3,
                obj_joint_dim=1,
                optimize_force=True,
                turn=True,
                obj_gravity=True
            )
            self.constraint_problem = constraint_problem
        
        # Actor and Critics
        self.actor = ActorNetwork(state_dim, action_dim, force_dim,
                                 action_low, action_high,
                                 constraint_problem=constraint_problem).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer - use PER or standard buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                state_dim,
                action_dim,
                buffer_size,
                device,
                alpha=per_alpha,
                beta=per_beta,
                beta_annealing=per_beta_annealing
            )
        else:
            self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)
        
        # For action selection timing
        self.prediction_times = []

    def select_action(self, state, evaluate=False):
        """Select an action from the policy given the state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            
            # Start timing action prediction
            start_time = time.time()
            
            action, force = self.actor(state, constrained=False)#True if self.use_constraints else False)
            
            # Add exploration noise when not evaluating
            if not evaluate:
                noise = torch.randn_like(action) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                action = action + noise
                action = torch.clamp(action, self.action_low, self.action_high)
            
            # End timing and record
            end_time = time.time()
            self.prediction_times.append(end_time - start_time)
        
        return action, force
    
    def update(self, batch_size=256):
        self.total_it += 1
        
        # Sample replay buffer
        if self.use_per:
            state, action, next_state, reward, done, indices, weights = self.replay_buffer.sample(batch_size)
            # Convert weights to appropriate shape for loss weighting
            weights = weights.reshape(-1, 1)
        else:
            state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
            weights = torch.ones((batch_size, 1), device=self.device)  # Equal weights when not using PER
        
        # Update critics
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action, _ = self.actor_target(next_state, constrained=False)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(self.action_low, self.action_high)
            
            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute TD errors for prioritization (before applying weights to the loss)
        td_errors = torch.max(
            torch.abs(target_q - current_q1),
            torch.abs(target_q - current_q2)
        ).detach()
        
        # Compute critic loss with importance sampling weights
        critic_loss_q1 = (weights * F.huber_loss(current_q1, target_q, reduction='none')).mean()
        critic_loss_q2 = (weights * F.huber_loss(current_q2, target_q, reduction='none')).mean()
        critic_loss = critic_loss_q1 + critic_loss_q2
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update priorities in PER buffer
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Delayed policy updates
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_action, _ = self.actor(state, constrained=False)
            actor_q1 = self.critic.q1_forward(state, actor_action)
            actor_q2 = self.critic.q2_forward(state, actor_action)
            actor_q = torch.min(actor_q1, actor_q2)
            actor_loss = -(weights * actor_q).mean()  # Apply importance sampling weights
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss, critic_loss.item()
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

def make_env(config):
    """
    Creates an environment
    """
    return AllegroScrewdriverRLEnv(config)

def train_td3(config, total_timesteps=1000000, save_path=None):
    """
    Train a TD3 agent on the AllegroScrewdriver environment
    """
    # Try to import torchdiffeq, install if not available
    try:
        import torchdiffeq
    except ImportError:
        print("torchdiffeq not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchdiffeq"])
        
    # Create environment
    env = make_env(config)
    
    # Create log directory
    if save_path is None:
        save_path = f"{CCAI_PATH}/models/allegro_screwdriver_td3"
    os.makedirs(save_path, exist_ok=True)
    
    # Use GPU for neural networks but keep simulation on CPU
    train_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Simulation running on: {config['sim_device']}")
    print(f"TD3 training running on: {train_device}")
    
    # Initialize TD3 agent
    td3_agent = TD3(
        state_dim=env.obs_dim,
        action_dim=env.control_dim,
        force_dim=9,
        action_low=env.action_low,
        action_high=env.action_high,
        device=train_device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.1,  # relative to the action range
        noise_clip=0.5,
        policy_freq=2,
        buffer_size=config.get('buffer_size', int(1e6)),
        use_constraints=config.get('use_constraints', False),
        env=env if config.get('use_constraints', False) else None,
        use_per=config.get('use_per', True),  # Enable PER by default
        per_alpha=config.get('per_alpha', 0.6),
        per_beta=config.get('per_beta', 0.4),
        per_beta_annealing=config.get('per_beta_annealing', 0.0001)
    )
    
    # Training hyperparameters
    batch_size = config.get('batch_size', 256)
    exploration_noise = 0.1
    start_timesteps =128# config.get('start_timesteps', 25000)  # Timesteps before using policy
    eval_freq = config.get('eval_freq', 5000)
    checkpoint_freq = config.get('checkpoint_freq', 10000)
    max_episode_steps = config.get('max_episode_steps', 100)
    
    # Tracking variables
    evaluations = []
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_rewards = []
    exec_timesteps = 0
    best_avg_reward = -float('inf')
    
    state = env.reset()
    done = False
    
    print(f"Starting training for {total_timesteps} timesteps")
    while exec_timesteps < total_timesteps:
        episode_timesteps += 1
        exec_timesteps += 1
        
        # Select action with noise for exploration
        # if exec_timesteps < start_timesteps:
        #     # Random actions until start_timesteps
        #     action = torch.FloatTensor(
        #         np.random.uniform(
        #             env.action_low, 
        #             env.action_high, 
        #             size=env.control_dim
        #         )
        #     ).to(train_device)
        # else:
            # Use policy with exploration noise
        action, force = td3_agent.select_action(state)
        
        # Perform action and get next state
        next_state, reward, done = env.step(action)
        
        # For TD3, we need to compute constraint violation as part of the reward
        if td3_agent.use_constraints:
            state_tensor = torch.FloatTensor(state).to(train_device)
            action_tensor = action.to(train_device)
            reward += -10 * td3_agent.actor.compute_constraint_violation(
                state_tensor, action_tensor, force
            ).item()
        
        # Store data in replay buffer
        td3_agent.replay_buffer.add(state, action, next_state, reward, done)
        
        state = next_state
        episode_reward += reward
        
        # Train agent after collecting enough samples
        if exec_timesteps >= start_timesteps:
            actor_loss, critic_loss = td3_agent.update(batch_size)
        
        # If episode is done or max steps reached
        if done or episode_timesteps >= max_episode_steps:
            print(f"Episode {episode_num}: {episode_timesteps} steps, reward={episode_reward:.2f}")
            
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
            # Store episode reward
            episode_rewards.append(episode_reward)
        
        # Evaluate agent periodically
        if exec_timesteps % eval_freq == 0:
            print(f"\n--- Evaluation at timestep {exec_timesteps} ---")
            avg_reward = evaluate_policy(td3_agent, config, num_episodes=5, render=False)
            evaluations.append(avg_reward)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                td3_agent.save(f"{save_path}/best_model.pt")
                print(f"New best model saved! Avg reward: {avg_reward:.2f}")
            
            # Save evaluation results
            np.save(f"{save_path}/evaluations.npy", evaluations)
        
        # Save checkpoint
        if exec_timesteps % checkpoint_freq == 0:
            td3_agent.save(f"{save_path}/model_{exec_timesteps}.pt")
            print(f"Checkpoint saved at timestep {exec_timesteps}")
            
            # Calculate and print average action prediction time
            avg_prediction_time = np.mean(td3_agent.prediction_times) * 1000 if td3_agent.prediction_times else 0
            max_prediction_time = np.max(td3_agent.prediction_times) * 1000 if td3_agent.prediction_times else 0
            print(f"Action prediction - avg: {avg_prediction_time:.2f}ms, max: {max_prediction_time:.2f}ms")
            
            # Reset prediction times
            td3_agent.prediction_times = []
    
    # Save final model
    td3_agent.save(f"{save_path}/final_model.pt")
    print(f"Training completed. Final model saved at {save_path}/final_model.pt")
    
    return td3_agent

def evaluate_policy(model, config, num_episodes=10, render=True):
    """
    Evaluate a trained policy
    """
    env = make_env(config)
    
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        episode_start_time = time.time()
        
        while not done:
            action, force = model.select_action(obs, evaluate=True)
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            if render:
                # Depending on your rendering setup, you might need to adjust this
                if hasattr(env.env, 'write_image'):
                    env.env.write_image()
                time.sleep(0.01)  # Add a small delay for visualization
        
        episode_duration = time.time() - episode_start_time
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, Duration: {episode_duration:.2f}s")
    
    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"===== Evaluation Results =====")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min(total_rewards):.2f}, Max reward: {max(total_rewards):.2f}")
    return avg_reward

if __name__ == "__main__":
    # Load configuration
    config_path = f'{CCAI_PATH}/examples/config/allegro_screwdriver_rl.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = yaml.safe_load(pathlib.Path(config_path).read_text())
    
    # Set simulation device to CPU
    config['sim_device'] = 'cpu'
    
    # Train TD3 agent (replaced PPO with TD3)
    model = train_td3(
        config, 
        total_timesteps=config.get('total_timesteps', 1000000),
        save_path=config.get('save_path', f"{CCAI_PATH}/models/allegro_screwdriver_td3")
    )
    
    # # Evaluate the trained policy
    # if config.get('evaluate', True):
    #     evaluate_policy(
    #         model, 
    #         config, 
    #         num_episodes=config.get('eval_episodes', 10),
    #         render=config.get('render_eval', True)
    #     )
