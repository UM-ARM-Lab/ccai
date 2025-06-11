"""
Baseline controllers and OOD detection methods for Allegro screwdriver recovery experiments.
Contains MPPI controller and Q-function based OOD detection functionality.
"""

import torch
import numpy as np
from collections import defaultdict
from pytorch_mppi import MPPI

from baselines.allegro_screwdriver import RunningCostSafeRL, TerminalCostDiffusionLikelihood
from baselines.dynamics_model import DynamicsModel
from baselines.mppi_planner import MPPIPlanner


class BaselineRecoveryController:
    """Base class for baseline recovery controllers."""
    
    def __init__(self, env, config, params, trajectory_sampler_orig=None):
        self.env = env
        self.config = config
        self.params = params
        self.trajectory_sampler_orig = trajectory_sampler_orig
        
        # Initialize based on controller type
        if config['recovery_controller'] == 'mppi':
            self._init_mppi()
        
    def _init_mppi(self):
        """Initialize MPPI controller and related components."""
        env_for_mppi = self.env if self.config['mode'] != 'hardware' else self.env  # sim_env in original
        self.dynamics = DynamicsModel(env_for_mppi, num_fingers=len(self.config['fingers']), 
                                    include_velocity=True, obj_joint_dim=1)
        
        safety_critic_path = self.config['model_path']
        
        if self.params['OOD_metric'] == 'q_function':
            self.running_cost = RunningCostSafeRL(safety_critic_path, self.params['q_cutoff'], 
                                                env_for_mppi, self.config['device'], include_velocity=True)
            self.terminal_cost = None
        elif self.params['OOD_metric'] == 'likelihood':
            self.running_cost = lambda x, y: 0
            self.terminal_cost = TerminalCostDiffusionLikelihood(self.trajectory_sampler_orig, env_for_mppi, self.config['device'])
        else:
            raise ValueError('Invalid OOD metric')
            
        # MPPI parameters
        self.u_max = torch.ones(4 * len(self.config['fingers'])) * np.pi / 5 
        self.u_min = -torch.ones(4 * len(self.config['fingers'])) * np.pi / 5
        self.noise_sigma = torch.eye(4 * len(self.config['fingers'])).to(self.config['device']) * .005
        self.nx = env_for_mppi.dof_states.shape[1] * 2 if hasattr(env_for_mppi, 'dof_states') else 32
        
        self.mppi_ctrl = None
        self.mppi_needs_warmup = True

    def create_mppi_controller(self):
        """Create and return MPPI controller."""
        if self.mppi_ctrl is None:
            self.mppi_ctrl = MPPI(
                dynamics=self.dynamics, 
                running_cost=self.running_cost, 
                terminal_state_cost=self.terminal_cost, 
                nx=self.nx, 
                noise_sigma=self.noise_sigma,
                num_samples=500, 
                horizon=self.params['T'], 
                lambda_=self.params['lambda_'], 
                u_min=self.u_min, 
                u_max=self.u_max,
                device=self.params['device']
            )
            self.mppi_needs_warmup = True
        return self.mppi_ctrl

    def create_mppi_planner(self, ctrl, warmup=False):
        """Create MPPI planner wrapper."""
        return MPPIPlanner(ctrl, 12, self.params['T'], warmup=warmup)

    def check_ood_q_function(self, state, action, num_fingers_to_plan):
        """Check if state-action pair is out-of-distribution using Q-function."""
        if not hasattr(self, 'running_cost') or not hasattr(self.running_cost, 'check_id'):
            return True, None
            
        q_func_action = action[0, :4 * num_fingers_to_plan].to(self.params['device'])
        q_func_action = q_func_action.reshape(1, -1)
        id_check, q_output = self.running_cost.check_id(state.unsqueeze(0), q_func_action)
        return id_check, q_output

    def is_mppi_controller(self):
        """Check if using MPPI controller."""
        return self.config['recovery_controller'] == 'mppi'

    def should_use_mppi_drop_logic(self):
        """Check if should use MPPI-specific drop logic."""
        return self.config['recovery_controller'] == 'mppi'


class BaselineOODDetector:
    """Handles out-of-distribution detection for baseline methods."""
    
    def __init__(self, params, trajectory_sampler_orig=None, running_cost=None):
        self.params = params
        self.trajectory_sampler_orig = trajectory_sampler_orig
        self.running_cost = running_cost
    
    def check_ood(self, state, action=None, num_fingers_to_plan=None, recover=False):
        """
        Check if current state (and optionally action) is out-of-distribution.
        
        Returns:
            tuple: (is_in_distribution, likelihood_or_q_value)
        """
        if recover and self.params['recovery_controller'] == 'mppi':
            # MPPI doesn't use OOD detection during recovery
            return True, None
        elif recover and self.params['recovery_controller'] != 'mppi':
            # Other recovery methods assume in-distribution during recovery
            return True, None
        elif not self.params.get('live_recovery', False):
            # No live recovery, assume in-distribution
            return True, None
        else:
            if self.params['OOD_metric'] == 'likelihood':
                return self._check_ood_likelihood(state)
            elif self.params['OOD_metric'] == 'q_function':
                if action is not None and num_fingers_to_plan is not None:
                    return self._check_ood_q_function(state, action, num_fingers_to_plan)
                else:
                    return True, None
            else:
                return True, None

    def _check_ood_likelihood(self, state):
        """Check OOD using likelihood from trajectory sampler."""
        if self.trajectory_sampler_orig is None:
            return True, None
            
        id_check, final_likelihood = self.trajectory_sampler_orig.check_id(
            state, 
            self.params['likelihood_num_samples'], 
            threshold=self.params.get('likelihood_threshold', -15)
        )
        return id_check, final_likelihood

    def _check_ood_q_function(self, state, action, num_fingers_to_plan):
        """Check OOD using Q-function."""
        if self.running_cost is None or not hasattr(self.running_cost, 'check_id'):
            return True, None
            
        q_func_action = action[0, :4 * num_fingers_to_plan].to(self.params['device'])
        q_func_action = q_func_action.reshape(1, -1)
        id_check, q_output = self.running_cost.check_id(state.unsqueeze(0), q_func_action)
        return id_check, q_output

    def check_drop_condition(self, state, recover=False):
        """Check if object has been dropped based on orientation."""
        roll_abs = np.abs(state[-3].item())
        pitch_abs = np.abs(state[-2].item())
        drop_cutoff = .35
        dropped = (roll_abs > drop_cutoff) or (pitch_abs > drop_cutoff)
        
        # Only apply drop logic for MPPI
        if self.params['recovery_controller'] == 'mppi':
            return dropped
        else:
            return False


def get_baseline_contact_sequence(params, recover=False):
    """Get contact sequence for baseline methods."""
    if params.get('live_recovery', False) and recover:
        if params['recovery_controller'] == 'mppi':
            return ['mppi']
    return ['turn']  # Default for non-baseline methods


def should_skip_diff_init(params, recover=False):
    """Check if should skip diffusion initialization for baseline methods."""
    return 'mppi' in params['recovery_controller'] and recover


def get_num_envs_for_baseline(config):
    """Get number of environments needed for baseline methods."""
    return 500 if 'mppi' in config['recovery_controller'] else 1


def create_baseline_planner(baseline_controller, ctrl, recover, mppi_warmup=False):
    """Create appropriate planner for baseline methods."""
    if baseline_controller.is_mppi_controller() and recover:
        return baseline_controller.create_mppi_planner(ctrl, warmup=mppi_warmup)
    return None


def handle_baseline_trajectory_processing(traj, plans, mode, device):
    """Handle trajectory processing specific to baseline methods."""
    if mode == 'mppi':
        # Add padding for MPPI trajectories
        plans = [torch.cat((plan[..., :-9],
                           torch.zeros(*plan.shape[:-1], 9).to(device=device),
                           plan[..., -9:]),
                          dim=-1) for plan in plans]
        traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=device)), dim=-1)
    
    return traj, plans


def get_baseline_final_likelihood(data, params, state, trajectory_sampler_orig):
    """Get final likelihood for baseline methods that don't use Q-function."""
    if (params.get('live_recovery', False) and 
        len(data['final_likelihoods'][-1]) == 0 and 
        params['OOD_metric'] != 'q_function'):
        
        id_check, likelihood = trajectory_sampler_orig.check_id(
            state, 
            params['likelihood_num_samples'], 
            threshold=params.get('likelihood_threshold', -15)
        )
        return likelihood
    return None 