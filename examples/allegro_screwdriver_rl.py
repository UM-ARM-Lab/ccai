from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
from ccai.utils.allegro_utils import state2ee_pos, partial_to_full_state, visualize_trajectory
from ccai.dataset import AllegroScrewDriverDataset
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



# Try to import wandb, set flag if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")

"""
Wandb Logging Overview:
This script logs the following metrics to Weights & Biases:

Training Metrics (logged every 100 steps):
- training/actor_loss: TD3 actor network loss
- training/critic_loss: TD3 critic network loss  
- training/buffer_size: Current replay buffer size
- training/constraint_violation: Constraint violation magnitude (if constraints enabled)

Episode Metrics (logged at episode completion):
- episode/reward: Individual episode reward
- episode/length: Episode length in steps
- episode/number: Episode number
- episode/reward_avg_10: Rolling average reward over last 10 episodes
- episode/reward_avg_100: Rolling average reward over last 100 episodes

Timing Metrics (logged at checkpoints):
- timing/action_prediction_avg_ms: Average action prediction time
- timing/action_prediction_max_ms: Maximum action prediction time
- timing/projection_avg_ms: Average constraint projection time (if constraints enabled)
- timing/constraint_computation_ms: Time spent computing constraints
- timing/matrix_solving_ms: Time spent solving constraint projection matrices

Hyperparameters (logged once at start):
- hyperparams/*: All training hyperparameters and configuration

To enable wandb logging, set 'use_wandb: True' in your config file and configure:
- wandb_project: Your wandb project name
- wandb_run_name: Optional run name (null for auto-generated)
- wandb_tags: List of tags for organizing runs
"""

from ccai.allegro_contact import AllegroManipulationProblem

import copy
from torch_cg import cg_batch

torch.autograd.set_detect_anomaly(True)

sys.path.append('..')

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

# sys.stdout = open('./logs/allegro_screwdriver_td3_force_0_c_1_1000_init_no_per_force_critic_euler.log', 'w', 1)
sys.stdout.reconfigure(line_buffering=True)

class AllegroScrewdriver(AllegroManipulationProblem):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 regrasp_fingers=[],
                 contact_fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                #  friction_coefficient=0.5,
                #  friction_coefficient=1000,
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 optimize_force=False,
                 turn=False,
                 obj_gravity=False,
                 min_force_dict=None,
                 device='cuda:0',
                 proj_path=None,
                 full_dof_goal=False, 
                 project=False,
                 test_recovery_trajectory=False, **kwargs):
        self.obj_mass = 0.1
        self.obj_dof_type = None
        if obj_dof == 3:
            object_link_name = 'screwdriver_body'
        elif obj_dof == 1:
            object_link_name = 'valve'
        elif obj_dof == 6:
            object_link_name = 'card'
        self.obj_link_name = object_link_name
        self.object_type = object_type


        self.contact_points = None
        contact_points_object = None
        if proj_path is not None:
            self.proj_path = proj_path.to(device=device)
        else:
            self.proj_path = None

        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1,
                                                 optimize_force=optimize_force, device=device,
                                                 turn=turn, obj_gravity=obj_gravity,
                                                 min_force_dict=min_force_dict, 
                                                 full_dof_goal=full_dof_goal,
                                                  contact_points_object=contact_points_object,
                                                  contact_points_dict = self.contact_points,
                                                  project=project,
                                                   **kwargs)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 0
        if not self.project:
            upright_cost = 500 * torch.sum(
                (state[:, -self.obj_dof:-1] + goal[-self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, start, goal)

class AllegroScrewdriverRLEnv:
    """Custom environment for the Allegro Hand screwdriver turning task"""
    
    def __init__(self, config):
        # Environment configuration
        self.config = config
        self.device = config['sim_device']
        self.num_fingers = len(config['fingers'])
        self.control_dim = 4 * self.num_fingers  # 4 joints per finger
        self.obj_dof = 3
        
        self.default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                dim=1)
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
            # default_dof_pos=self.default_dof_pos
        )
        
        self.env.frame_fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/')
        # Create the directory if it doesn't exist
        if not self.env.frame_fpath.exists():
            self.env.frame_fpath.mkdir(parents=True, exist_ok=True)
        
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
        self.chain = chain.to(device=config['device'])
        
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
    
    def step(self, action, force):
        # Convert action to target joint positions
        state_dict = self.env.get_state()
        current_pos = state_dict['q'].reshape(-1)[:self.control_dim]
        target_pos = current_pos + action.cpu()
        
        # Step the environment
        self.env.step(target_pos.reshape(1, -1), ignore_img=True)
        self.episode_steps += 1
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._compute_reward(obs, action, force)
        
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

    def _compute_reward(self, obs, action, force):
        # Extract orientation
        orientation = obs[self.control_dim:self.control_dim + self.obj_dof]
        roll, pitch, yaw = orientation
        
        # Reward turning (yaw change)
        turn_reward = -(yaw - self.initial_yaw) * 1.0
        self.initial_yaw = yaw  # Update initial yaw for next step
        # Penalize deviation from upright position
        # upright_penalty = (roll.abs() + pitch.abs()) * 1.0
        
        # Force magnitude reward
        force_reward = 0#-force.norm(dim=-1) * 0
        
        # Calculate total reward
        reward = turn_reward + force_reward #- upright_penalty

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
    def __init__(self, input_dim, output_dim, force_dim, action_low, action_high, hidden_dim=256, constraint_problem=None, device='cuda:0'):
        super(ActorNetwork, self).__init__()
        
        self.device = device
        # Store dimensions
        self.input_dim = input_dim
        # self.output_dim = output_dim
        self.force_dim = force_dim
        # self.combined_dim_no_force = input_dim + output_dim
        self.combined_dim = input_dim + force_dim

        self.action_low = action_low
        self.action_high = action_high
        
        # Vector field network that models the dynamics of the state-action system
        self.vector_field = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.combined_dim)
        )

        # # Predicts initial x dot for ODE integration
        # self.xdot0 = nn.Sequential(
        #     nn.Linear(self.combined_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, self.combined_dim)
        # )
        
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
        self.robot_joint_x_max = torch.cat([joint_max[finger] for finger in self.fingers]).to(device)
        self.robot_joint_x_min = torch.cat([joint_min[finger] for finger in self.fingers]).to(device)

        self.robot_joint_x_min = torch.cat([self.robot_joint_x_min, torch.tensor([-torch.pi, -torch.pi, -torch.pi]).to(device)])
        self.robot_joint_x_max = torch.cat([self.robot_joint_x_max, torch.tensor([torch.pi, torch.pi, torch.pi]).to(device)])

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
            'calls': 0,
            'select_action': []
        }
        
        self.fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/')
        self.iter = 0
        
    def forward(self, x, constrained=True):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Instead of using stochastic policy, we use a deterministic one for TD3
        # Generate zero vectors instead of random normal ones
        z = torch.randn(batch_size, self.force_dim, device=x.device)
        
        # Concatenate state and zero vector
        x_z = torch.cat([x, z], dim=-1)

        # x_z_dot0 = self.xdot0(x_z)
        # x_z_dot0 = torch.zeros_like(x_z)

        # integration_input = torch.cat([x_z, x_z_dot0], dim=-1)
        integration_input = x_z.unsqueeze(0)
        
        # Integrate ODE from t=0 to t=1
        # integration_times = torch.tensor([0.0, 1.0], device=x.device)
        integration_times = torch.linspace(0, 1, 6, device=x.device)

        # Unsqueeze for CSVTO
        if constrained:
            dynamics = self.constrained_dynamics
            # integration_input = integration_input.unsqueeze(1)  # Add time dimension for odeint
        else:
            dynamics = self.unconstrained_dynamics
        trajectory = odeint(dynamics, integration_input, integration_times, method='euler', options={'step_size': .2})
        trajectory = trajectory.squeeze(1)
        # if constrained:
        #     trajectory = trajectory.squeeze(1)
        trajectory = trajectory.permute(1, 0, 2)  # Change to (batch_size, time_steps, dim)
        trajectory[..., self.input_dim:] = torch.clamp(trajectory[..., self.input_dim:].clone(), -2, 2)

        # if constrained:
        self.constraint_problem.start = trajectory[0, 0, :15]
        self.constraint_problem._preprocess(trajectory[:, -1:])

        u_hat = self.constraint_problem.solve_for_u_hat(trajectory[:, -1:])
        
        # Extract the final point of the trajectory
        final_state = trajectory[:, -1]

        # final_state[self.input_dim:self.combined_dim_no_force] = u_hat.squeeze(1)
        
        # Return deterministic action and force
        action = u_hat
        action = torch.clamp(action, self.action_low, self.action_high)
        force = final_state[:,self.input_dim:]
        
        if self.iter % 100 == 0:
            traj_for_viz = trajectory.clone()[0, :, :15]
            
            tmp = torch.zeros((traj_for_viz.shape[0], 1),
                            device=traj_for_viz.device)  # add the joint for the screwdriver cap
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

            constrained = 'constrained' if constrained else 'unconstrained'
            viz_fpath = pathlib.PurePath.joinpath(self.fpath, f"integration_paths/{self.iter}/{constrained}/")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, self.constraint_problem.contact_scenes_for_viz, viz_fpath,
                                self.constraint_problem.fingers, self.constraint_problem.obj_dof + 1)
        # else:
        #     final_state = trajectory[:, -1]
        #     action_ = final_state[:, self.input_dim:self.combined_dim_no_force]
        #     action = torch.clamp(action_.clone(), self.action_low, self.action_high)
        #     force_ = final_state[:, self.combined_dim_no_force:self.combined_dim]
        #     force = torch.clamp(force_.clone(), -2, 2)
        
        self.iter += 1
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
            compute_inequality=False,
            include_deriv_grad=False
        )
        return C.abs().sum()  

    def project_vector_field(self, x, dx):
        """Project vector field to satisfy constraints"""
        start_time = time.time()
        self.timing_metrics['calls'] += 1
        
        # dx *= 0
        
      
        # Phase 1: Initial setup and clipping
        clamp_start = time.time()
        batch_size = x.shape[1]
        # if batch_size > 1:
        #     print(f"Batch size: {batch_size}")
        
        dx_ = dx.clone().permute(1, 0, 2)
        x_ = x.clone().permute(1, 0, 2)
        x_[:, :, :self.input_dim] = torch.clamp(x_[:, :, :self.input_dim], self.robot_joint_x_min.to(self.device), self.robot_joint_x_max.to(self.device))
        x_[:, :, self.input_dim:] = torch.clamp(x_[:, :, self.input_dim:], -2, 2)

        clamp_end = time.time()
        self.timing_metrics['clamp_setup'].append(clamp_end - clamp_start)
        
        # Parameters for projection
        Kh = 3
        Ktheta = 1

        # Phase 2: Preprocessing with constraint problem
        preprocess_start = time.time()
        decision_variable = x_[:, :, :self.combined_dim]
        # Apply projection
        self.constraint_problem.start = decision_variable[0, 0, :self.input_dim]
        self.constraint_problem._preprocess(decision_variable, dxu=x_[:, :, self.combined_dim:], projected_diffusion=True)
        preprocess_end = time.time()
        self.timing_metrics['preprocessing'].append(preprocess_end - preprocess_start)
        
        # Phase 3: Computing constraints and derivatives
        constraint_start = time.time()
        # Get constraints and their derivatives
        decision_variable_for_constraints = torch.cat((decision_variable[:, :, :self.input_dim], torch.zeros_like(decision_variable[:, :, :12]), decision_variable[:, :, self.input_dim:]), dim=-1)
        C, dC, _ = self.constraint_problem.combined_constraints(
            decision_variable_for_constraints,
            compute_hess=False,
            projected_diffusion=True,
            include_slack=False,
            compute_inequality=False,
            include_deriv_grad=False
        )
        dC = torch.cat((dC[:, :, :self.input_dim], dC[:, :, self.input_dim+12:]), dim=-1)
        # dC[:, :, 12:15] = 0
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
        
        
        # Project the vector field with the ODE method
        # dh_bar_dJ = torch.bmm(dh_bar, dx_.permute(0, 2, 1))
        # second_term = Ktheta * dh_bar_dJ - Kh * h_bar.unsqueeze(-1)
        # pi = -dCdCT_inv @ second_term
        
        # # Ensure inequality constraints are properly handled
        # pi[:, g_masked.shape[-1]:] = torch.clamp(pi[:, g_masked.shape[-1]:], min=0)
        
        # # Apply projection
        # projected_dx = dx_ - (dh_bar.permute(0, 2, 1) @ pi).permute(0, 2, 1)

        # Project the vector field with the CSVTO method
        projection = dCdCT_inv @ dC
        eye = torch.eye(dC.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
        projection = eye - dC.permute(0, 2, 1) @ projection
        # compute term for repelling towards constraint


        xi_C = dCdCT_inv @ C.unsqueeze(-1)
        xi_C = (dC.permute(0, 2, 1) @ xi_C).squeeze(-1)

        xi_J = torch.einsum('ijk,ilk->lij', projection, dx_)
        # Apply projection
        
        projected_dx = -(1 * xi_J + 3 * xi_C)

        # x__ = x_.clone()
        # x__[:, :, :self.input_dim] = torch.clamp(x__[:, :, :self.input_dim], self.robot_joint_x_min.to(self.device), self.robot_joint_x_max.to(self.device))
        # x__[:, :, self.input_dim:] = torch.clamp(x__[:, :, self.input_dim:], -2, 2)

        # x_ = x__
        projection_end = time.time()
        self.timing_metrics['projection_application'].append(projection_end - projection_start)
        
        # Record total time
        end_time = time.time()
        self.timing_metrics['total_time'].append(end_time - start_time)
        
        # # Print timing summary every 1 calls
        # if self.timing_metrics['calls'] % 1 == 0:
        #     self._print_timing_summary()
        
        # If forces are pointing away from the object, project them to contact normal's perpendicular plane
        force = x[0, :, self.input_dim:].reshape(batch_size, len(self.constraint_problem.contact_fingers), 3)
        
        force_robot_frame = self.constraint_problem.world_trans.inverse().transform_normals(force)
        
        for finger in self.constraint_problem.contact_fingers:
            contact_normal = self.constraint_problem.data[finger]['contact_normal']
            contact_normal_robot_frame = self.constraint_problem.world_trans.inverse().transform_normals(contact_normal)
            force_this_finger = force_robot_frame[:, self.constraint_problem.contact_fingers.index(finger)]
            force_dot_normal = torch.sum(force_this_finger * contact_normal, dim=-1, keepdim=True).flatten()
            
            start_index = self.constraint_problem.contact_fingers.index(finger) * 3
            end_index = start_index + 3
            # Handle the case where force_dot_normal is a vector (batch)
            # Create a mask for all batch elements where force_dot_normal > 0
            mask = force_dot_normal > 0  # shape: (batch_size,)
            if mask.any():
                # Only compute for those elements where mask is True
                # Expand mask for broadcasting
                mask_expanded = mask.unsqueeze(-1)  # shape: (batch_size, 1)
                # Compute vector rejection only for masked elements
                force_rejection = force_this_finger - force_dot_normal.unsqueeze(-1) * contact_normal_robot_frame
                force_rejection_world_frame = self.constraint_problem.world_trans.transform_normals(force_rejection)
                # Only update projected_dx for masked elements
                # projected_dx[0, mask, ...] = ... for those indices
                # projected_dx shape: (1, batch_size, D)
                # force_rejection_world_frame shape: (batch_size, 3)
                # We want to update projected_dx[0, mask, self.input_dim+start_index:self.input_dim+end_index]
                projected_dx[0, mask, self.input_dim+start_index:self.input_dim+end_index] = -force_rejection_world_frame[mask] / 0.2

        
        return projected_dx#.permute(1, 0, 2)
    
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
                start=torch.zeros(state_dim).to(device),
                goal=torch.zeros(3).to(device),  # Goal will be set dynamically
                T=1,
                chain=env.chain,
                device=torch.device(device),
                object_asset_pos=env.env.table_pose,
                object_location=torch.tensor([0, 0, 1.205]).to(device),
                object_type='screwdriver',
                world_trans=env.env.world_trans,
                contact_fingers=['index', 'middle', 'thumb'],
                obj_dof=3,
                obj_joint_dim=1,
                optimize_force=True,
                turn=True,
                obj_gravity=True,
            )
            self.constraint_problem = constraint_problem
        
        # Actor and Critics
        self.actor = ActorNetwork(state_dim, action_dim, force_dim,
                                 action_low, action_high,
                                 constraint_problem=constraint_problem,
                                 device=device).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = CriticNetwork(state_dim, force_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer - use PER or standard buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                state_dim,
                force_dim,
                buffer_size,
                device,
                alpha=per_alpha,
                beta=per_beta,
                beta_annealing=per_beta_annealing
            )
        else:
            self.replay_buffer = ReplayBuffer(state_dim, force_dim, buffer_size, device)
        
        # For action selection timing
        self.prediction_times = []

    def select_action(self, state, evaluate=False):
        """Select an action from the policy given the state."""
        begin_time = time.perf_counter()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            
            # Start timing action prediction
            start_time = time.time()
            
            action, force = self.actor(state, constrained=True)#True if self.use_constraints else False)
            
            # Add exploration noise when not evaluating
            # if not evaluate:
            #     noise = torch.randn_like(action) * self.policy_noise
            #     noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            #     action = action + noise
            #     action = torch.clamp(action, self.action_low, self.action_high)
            
            # End timing and record
            end_time = time.time()
            self.prediction_times.append(end_time - start_time)
        end_time = time.perf_counter()
        # print(f"Select action time: {end_time - begin_time}")
        self.actor.timing_metrics['select_action'].append(end_time - begin_time)
        return action, force
    
    def update(self, batch_size=256):
        self.total_it += 1
        
        # Sample replay buffer
        if self.use_per:
            state, force, next_state, reward, done, indices, weights = self.replay_buffer.sample(batch_size)
            # Convert weights to appropriate shape for loss weighting
            weights = weights.reshape(-1, 1)
        else:
            state, force, next_state, reward, done = self.replay_buffer.sample(batch_size)
            weights = torch.ones((batch_size, 1), device=self.device)  # Equal weights when not using PER
        
        # Update critics
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            _, next_force = self.actor_target(next_state, constrained=True)
            # noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # next_action = (next_action + noise).clamp(self.action_low, self.action_high)
            
            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_force)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, force)
        
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
            actor_action, actor_force = self.actor(state, constrained=False) # TODO: change to True. Might need to cache intermediate constraint gradients for loss calculation
            actor_q1 = self.critic.q1_forward(state, actor_force)
            actor_q2 = self.critic.q2_forward(state, actor_force)
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
    # Initialize wandb if available and enabled
    use_wandb = config.get('use_wandb', True) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'allegro-screwdriver-td3'),
            name=config.get('wandb_run_name', None),
            config={
                **config,
                'total_timesteps': total_timesteps,
                'algorithm': 'TD3'
            },
            tags=config.get('wandb_tags', ['allegro', 'td3', 'constrained-rl'])
        )
        print("wandb logging initialized")
    else:
        print("wandb logging disabled")
    
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
    train_device = config['device'] if torch.cuda.is_available() else 'cpu'
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
        lr_actor=config.get('lr_actor', 3e-4),
        lr_critic=config.get('lr_critic', 1e-3),
        gamma=config.get('gamma', 0.99),
        tau=config.get('tau', 0.005),
        policy_noise=config.get('policy_noise', 0.1),
        noise_clip=config.get('noise_clip', 0.5),
        policy_freq=config.get('policy_freq', 2),
        buffer_size=config.get('buffer_size', int(1e6)),
        use_constraints=config.get('use_constraints', False),
        env=env if config.get('use_constraints', False) else None,
        use_per=config.get('use_per', True),  # Enable PER by default
        per_alpha=config.get('per_alpha', 0.6),
        per_beta=config.get('per_beta', 0.4),
        per_beta_annealing=config.get('per_beta_annealing', 0.0001)
    )
    
    # td3_agent.actor.load_state_dict(torch.load(f"/home/abhinav/Documents/constrained_rl/examples/models/allegro_screwdriver_td3/best_model_allegro_screwdriver_constrained_rl_f_0_c_1_start_1000_no_per_force_critic.pt")['actor'])
    # td3_agent.critic.load_state_dict(torch.load(f"/home/abhinav/Documents/constrained_rl/examples/models/allegro_screwdriver_td3/best_model_allegro_screwdriver_constrained_rl_f_0_c_1_start_1000_no_per_force_critic.pt")['critic'])
    # td3_agent.actor_target.load_state_dict(torch.load(f"/home/abhinav/Documents/constrained_rl/examples/models/allegro_screwdriver_td3/best_model_allegro_screwdriver_constrained_rl_f_0_c_1_start_1000_no_per_force_critic.pt")['actor_target'])
    # td3_agent.critic_target.load_state_dict(torch.load(f"/home/abhinav/Documents/constrained_rl/examples/models/allegro_screwdriver_td3/best_model_allegro_screwdriver_constrained_rl_f_0_c_1_start_1000_no_per_force_critic.pt")['critic_target'])
    # td3_agent.actor_optimizer.load_state_dict(torch.load(f"/home/abhinav/Documents/constrained_rl/examples/models/allegro_screwdriver_td3/best_model_allegro_screwdriver_constrained_rl_f_0_c_1_start_1000_no_per_force_critic.pt")['actor_optimizer'])
    # td3_agent.critic_optimizer.load_state_dict(torch.load(f"/home/abhinav/Documents/constrained_rl/examples/models/allegro_screwdriver_td3/best_model_allegro_screwdriver_constrained_rl_f_0_c_1_start_1000_no_per_force_critic.pt")['critic_optimizer'])
    
    # Log hyperparameters to wandb
    if use_wandb:
        wandb.log({
            'hyperparams/state_dim': env.obs_dim,
            'hyperparams/action_dim': env.control_dim,
            'hyperparams/lr_actor': config.get('lr_actor', 3e-4),
            'hyperparams/lr_critic': config.get('lr_critic', 1e-3),
            'hyperparams/gamma': config.get('gamma', 0.99),
            'hyperparams/tau': config.get('tau', 0.005),
            'hyperparams/policy_noise': config.get('policy_noise', 0.1),
            'hyperparams/noise_clip': config.get('noise_clip', 0.5),
            'hyperparams/policy_freq': config.get('policy_freq', 2),
            'hyperparams/buffer_size': config.get('buffer_size', int(1e6)),
            'hyperparams/use_constraints': config.get('use_constraints', False),
            'hyperparams/use_per': config.get('use_per', True),
            'hyperparams/friction_coefficient': config.get('friction_coefficient', 0.95),
            'hyperparams/fingers': str(config.get('fingers', ['index', 'middle', 'thumb']))
        })
    
    # Training hyperparameters
    batch_size = config.get('batch_size', 256)
    exploration_noise = 0.1
    start_timesteps = config.get('start_timesteps', 1000)  # Timesteps before using policy
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
    
    # Load dataset for replay buffer warm start
    if config.get('warm_start_buffer', False):
        data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
        train_dataset = AllegroScrewDriverDataset([p for p in data_path.glob('*train_data*')],
                                                12,
                                                15,
                                                cosine_sine=False,
                                                states_only=False,
                                                skip_pregrasp=True,
                                                best_traj_only=True,
                                                recovery=False)
        
        
        states_warm_start, actions_warm_start, next_states_warm_start, rewards_warm_start, dones_warm_start = train_dataset.get_replay_buffer_warm_start(device=train_device)
        
        for i in range(len(states_warm_start)):
            td3_agent.replay_buffer.add(states_warm_start[i], actions_warm_start[i], next_states_warm_start[i], rewards_warm_start[i], dones_warm_start[i])
    
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
        next_state, reward, done = env.step(action, force)
        
        # For TD3, we need to compute constraint violation as part of the reward
        if td3_agent.use_constraints:
            state_tensor = torch.FloatTensor(state).to(train_device)
            action_tensor = action.to(train_device)
            reward += -1 * td3_agent.actor.compute_constraint_violation(
                state_tensor, action_tensor, force
            ).item()
        
        # Store data in replay buffer
        td3_agent.replay_buffer.add(state, force, next_state, reward, done)
        
        state = next_state
        episode_reward += reward
        
        # Train agent after collecting enough samples
        if td3_agent.replay_buffer.size >= start_timesteps:
            actor_loss, critic_loss = td3_agent.update(batch_size)
            
            # Log training metrics to wandb
            if use_wandb:  # Log every 100 steps to avoid spam
                training_metrics = {
                    'training/actor_loss': actor_loss,
                    'training/critic_loss': critic_loss,
                    'training/timestep': exec_timesteps,
                    'training/buffer_size': td3_agent.replay_buffer.size
                }
                if actor_loss == 0:
                    del training_metrics['training/actor_loss']
                # # Add constraint violation if using constraints
                # if td3_agent.use_constraints and constraint_violation > 0:
                #     training_metrics['training/constraint_violation'] = constraint_violation
                
                wandb.log(training_metrics, step=exec_timesteps)
        
        # If episode is done or max steps reached
        if done or episode_timesteps >= max_episode_steps:
            print(f"Episode {episode_num}: {episode_timesteps} steps, reward={episode_reward:.2f}")
            
            # Log episode metrics to wandb
            if use_wandb:
                episode_metrics = {
                    'episode/reward': episode_reward,
                    'episode/length': episode_timesteps,
                    'episode/number': episode_num,
                    'training/timestep': exec_timesteps
                }
                
                # Add rolling averages if we have enough episodes
                if len(episode_rewards) >= 10:
                    episode_metrics['episode/reward_avg_10'] = np.mean(episode_rewards[-10:])
                if len(episode_rewards) >= 100:
                    episode_metrics['episode/reward_avg_100'] = np.mean(episode_rewards[-100:])
                
                wandb.log(episode_metrics, step=exec_timesteps)
            
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
            td3_agent.save(f"{save_path}/best_model_{config['experiment_name']}.pt")
            # print(f"\n--- Evaluation at timestep {exec_timesteps} ---")
            # avg_reward = evaluate_policy(td3_agent, config, num_episodes=5, render=False)
            # evaluations.append(avg_reward)
            
            # if avg_reward > best_avg_reward:
            #     best_avg_reward = avg_reward
            #     td3_agent.save(f"{save_path}/best_model.pt")
            #     print(f"New best model saved! Avg reward: {avg_reward:.2f}")
            
            # # Save evaluation results
            # np.save(f"{save_path}/evaluations.npy", evaluations)
        
        # Save checkpoint
        if exec_timesteps % checkpoint_freq == 0:
            td3_agent.save(f"{save_path}/model_{exec_timesteps}.pt")
            print(f"Checkpoint saved at timestep {exec_timesteps}")
            
            # Calculate and print average action prediction time
            avg_prediction_time = np.mean(td3_agent.prediction_times) * 1000 if td3_agent.prediction_times else 0
            max_prediction_time = np.max(td3_agent.prediction_times) * 1000 if td3_agent.prediction_times else 0
            print(f"Action prediction - avg: {avg_prediction_time:.2f}ms, max: {max_prediction_time:.2f}ms")
            
            # Log timing metrics to wandb
            if use_wandb:
                timing_metrics = {}
                if td3_agent.prediction_times:
                    timing_metrics['timing/action_prediction_avg_ms'] = avg_prediction_time
                    timing_metrics['timing/action_prediction_max_ms'] = max_prediction_time
                
                # Log constraint projection timing if available
                if hasattr(td3_agent.actor, 'timing_metrics') and td3_agent.actor.timing_metrics['total_time']:
                    avg_projection_time = np.mean(td3_agent.actor.timing_metrics['total_time'][-100:]) * 1000
                    timing_metrics['timing/projection_avg_ms'] = avg_projection_time
                    timing_metrics['timing/projection_calls'] = td3_agent.actor.timing_metrics['calls']
                    
                    # Log breakdown of projection timing
                    if td3_agent.actor.timing_metrics['constraint_computation']:
                        timing_metrics['timing/constraint_computation_ms'] = np.mean(td3_agent.actor.timing_metrics['constraint_computation'][-100:]) * 1000
                    if td3_agent.actor.timing_metrics['matrix_solving']:
                        timing_metrics['timing/matrix_solving_ms'] = np.mean(td3_agent.actor.timing_metrics['matrix_solving'][-100:]) * 1000
                
                if timing_metrics:
                    timing_metrics['training/timestep'] = exec_timesteps
                    wandb.log(timing_metrics, step=exec_timesteps)
            
            # Reset prediction times
            td3_agent.prediction_times = []
    
    # Save final model
    td3_agent.save(f"{save_path}/final_model.pt")
    print(f"Training completed. Final model saved at {save_path}/final_model.pt")
    
    # Log final metrics and finish wandb run
    if use_wandb:
        final_metrics = {
            'training/total_episodes': episode_num,
            'training/final_timestep': exec_timesteps
        }
        if episode_rewards:
            final_metrics['training/final_avg_reward_100'] = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            final_metrics['training/total_reward'] = np.sum(episode_rewards)
        
        wandb.log(final_metrics, step=exec_timesteps)
        wandb.finish()
        print("wandb run finished")
    
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
    
    experiment_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}')
    pathlib.Path.mkdir(experiment_dir, parents=True, exist_ok=True)
    log_file = experiment_dir / 'log.log'
    log_file.touch()
    sys.stdout = open(log_file, 'w', buffering=1)
    
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
