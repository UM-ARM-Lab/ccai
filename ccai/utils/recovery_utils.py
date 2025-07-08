"""
Utility functions for Allegro screwdriver recovery experiments.
Contains functions to reduce code duplication in recovery logic.
"""

import torch
import pickle as pkl
import pathlib
from copy import deepcopy
import pickle
import numpy as np

from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC
from ccai.utils.allegro_utils import visualize_trajectory


def create_experiment_paths(fpath, fname, mode=None, create_goal_subdir=True):
    """Create directory structure for experiment data."""
    mode_fpath = pathlib.Path(fpath) / fname
    mode_fpath.mkdir(parents=True, exist_ok=True)
    
    paths = {'mode_fpath': mode_fpath}
    
    if mode and create_goal_subdir:
        goal_fpath = mode_fpath / mode / 'goal'
        goal_fpath.mkdir(parents=True, exist_ok=True)
        paths['goal_fpath'] = goal_fpath
    
    return paths


def create_visualization_paths(base_path, subdir_name):
    """Create visualization directory structure."""
    viz_fpath = pathlib.Path(base_path) / subdir_name
    img_fpath = viz_fpath / 'img'
    gif_fpath = viz_fpath / 'gif'
    
    viz_fpath.mkdir(parents=True, exist_ok=True)
    img_fpath.mkdir(parents=True, exist_ok=True)
    gif_fpath.mkdir(parents=True, exist_ok=True)
    
    return viz_fpath, img_fpath, gif_fpath


def setup_and_visualize_trajectory(traj_for_viz, contact_scenes, base_path, subdir_name, fingers, obj_dof):
    """Set up visualization paths and visualize trajectory."""
    viz_fpath, _, _ = create_visualization_paths(base_path, subdir_name)
    visualize_trajectory(traj_for_viz, contact_scenes, viz_fpath, fingers, obj_dof + 1)
    return viz_fpath


def save_goal_info(goal_fpath, goal, state):
    """Save goal information to file."""
    goal_info_path = goal_fpath / "goal_info.pkl"
    with open(goal_info_path, "wb") as f:
        pkl.dump((goal, state), f)


def save_recovery_info(viz_fpath, plans, samples, likelihood):
    """Save recovery planning information to file."""
    recovery_info_path = viz_fpath / "recovery_info.pkl"
    with open(recovery_info_path, "wb") as f:
        pkl.dump((plans, samples, likelihood), f)


def save_projection_results(mode_fpath, initial_samples, initial_samples_0, all_losses, all_samples, all_likelihoods):
    """Save projection results to file."""
    projection_path = mode_fpath / 'projection_results.pkl'
    with open(projection_path, 'wb') as f:
        pickle.dump((initial_samples, initial_samples_0, all_losses, all_samples, all_likelihoods), f)


def create_allegro_screwdriver_problem(problem_type, start, goal, params, env, device, 
                                     contact_fingers=None, regrasp_fingers=None, 
                                     min_force_dict=None, proj_path=None, AllegroScrewdriver=None, **kwargs):
    """Factory function to create AllegroScrewdriver problems with common parameters."""
    if AllegroScrewdriver is None:
        raise ValueError("AllegroScrewdriver class must be provided as parameter")
    
    # Common parameters for all problems
    common_params = {
        'start': start,
        'goal': goal,
        'chain': params['chain'],
        'device': device,
        'object_asset_pos': env.table_pose,
        'object_location': params['object_location'],
        'object_type': params['object_type'],
        'world_trans': env.world_trans,
        'obj_dof': 3,
        'obj_joint_dim': 1,
        'optimize_force': params['optimize_force'],
        'default_dof_pos': env.default_dof_pos[:, :16],
        'obj_gravity': params.get('obj_gravity', False),
        'contact_constraint_only': params.get('contact_constraint_only', False),
    }
    
    # Problem-specific configurations
    if problem_type == 'pregrasp':
        specific_params = {
            'T': 2,
            'contact_fingers': [],
            'regrasp_fingers': regrasp_fingers or [],
            'full_dof_goal': False,
            'proj_path': proj_path,
        }
    elif problem_type == 'index_regrasp':
        specific_params = {
            'T': params['T'],
            'regrasp_fingers': ['index'],
            'contact_fingers': ['middle', 'thumb'],
            'min_force_dict': min_force_dict,
            'full_dof_goal': True,
            'proj_path': None,
            'project': True,
        }
    elif problem_type == 'thumb_middle_regrasp':
        specific_params = {
            'T': params['T'],
            'contact_fingers': ['index'],
            'regrasp_fingers': ['middle', 'thumb'],
            'min_force_dict': min_force_dict,
            'full_dof_goal': True,
            'proj_path': None,
            'project': True,
        }
    elif problem_type == 'all_regrasp':
        specific_params = {
            'T': params['T'],
            'contact_fingers': [],
            'regrasp_fingers': ['index', 'middle', 'thumb'],
            'min_force_dict': min_force_dict,
            'full_dof_goal': True,
            'proj_path': None,
            'project': True,
            'object_asset_pos': env.obj_pose,  # Different for all_regrasp
        }
    elif problem_type == 'turn':
        specific_params = {
            'T': kwargs.get('T_override', params.get('T_orig', params['T'])),
            'contact_fingers': ['index', 'middle', 'thumb'],
            'turn': True,
            'obj_gravity': False,
            'min_force_dict': min_force_dict,
            'full_dof_goal': False,
            'proj_path': proj_path,
            'project': False,
        }
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Merge parameters
    final_params = {**common_params, **specific_params}
    final_params.update(kwargs)  # Allow overrides
    
    return AllegroScrewdriver(**final_params)

class ConstraintScheduledSVGDMPC(PositionControlConstrainedSVGDMPC):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.contact_only_warmup_iters = params.get('contact_only_warmup_iters', 0)
        self.contact_only_online_iters = params.get('contact_only_online_iters', 0)

    def step(self, state, skip_optim=False, **kwargs):
        if self.fix_T:
            new_T = None
        else:
            if self.warmed_up:
                new_T = self.problem.T - 1
            else:
                new_T = self.problem.T

        # Contact only
        self.problem.update(state, T=new_T, contact_constraint_only=True, **kwargs)
        if (self.warmed_up and self.contact_only_online_iters > 0) or (not self.warmed_up and self.contact_only_warmup_iters > 0):
            if self.warmed_up:
                self.solver.iters = self.contact_only_online_iters
                resample = True if (self.iter + 1) % self.resample_steps == 0 else False
            else:
                self.solver.iters = self.contact_only_warmup_iters
                if self.online_iters == 0 and self.warmup_iters == 0:
                    self.warmed_up = True
                resample = False

            path = self.solver.solve(self.x, resample, skip_optim=skip_optim)
        
        if (self.warmed_up and self.online_iters > 0) or (not self.warmed_up and self.warmup_iters > 0):
            # Standard
            self.problem.update(state, T=new_T, contact_constraint_only=False, **kwargs)
            if self.warmed_up:
                self.solver.iters = self.online_iters
                resample = True if (self.iter + 1) % self.resample_steps == 0 else False
            else:
                self.solver.iters = self.warmup_iters
                self.warmed_up = True
                resample = False
            path = self.solver.solve(self.x, resample, skip_optim=skip_optim)

        self.x = path[-1]
        self.path = path
        self.iter += 1
        best_trajectory = self.x[0].clone()
        all_trajectories = self.x.clone()
        self.shift()
        # self.x = self.problem.get_initial_xu(self.N)
        return best_trajectory, all_trajectories

def create_planner(problem, params, planner_type='default'):
    """Create a planner for the given problem."""
    if planner_type == 'recovery':
        recovery_params = deepcopy(params)
        return ConstraintScheduledSVGDMPC(problem, recovery_params)
    else:
        return ConstraintScheduledSVGDMPC(problem, params)


def initialize_data_structure(params):
    """Initialize the data structure for tracking experiment results."""
    data = {}
    t_range = params['T']
    if 'T_orig' in params and params['T_orig'] > t_range:
        t_range = params['T_orig']
    
    for t in range(1, 1 + t_range):
        data[t] = {
            'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 
            'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 
            'contact_state': []
        }
    
    data.update({
        'pre_action_likelihoods': [],
        'final_likelihoods': [],
        'csvto_times': [],
        'project_times': [],
        'all_samples_': [],
        'all_likelihoods_': [],
        'contact_plan_times': [],
        'executed_contacts': []
    })
    
    return data


def add_to_dataset(data, traj, plans, inits, init_sim_rollouts, optimizer_paths, 
                  contact_points, contact_distance, contact_state):
    """Add trajectory data to the dataset."""
    for i, plan in enumerate(plans):
        t = plan.shape[1]
        data[t]['plans'].append(plan)
        data[t]['inits'].append(inits.cpu().numpy())
        data[t]['init_sim_rollouts'].append(init_sim_rollouts)
        try:
            data[t]['optimizer_paths'].append([i.cpu().numpy() for i in optimizer_paths])
        except:
            pass
        data[t]['starts'].append(traj[i].reshape(1, -1).repeat(plan.shape[0], 1))
        data[t]['contact_state'].append(contact_state)
        try:
            data[t]['contact_points'].append(contact_points[t])
            data[t]['contact_distance'].append(contact_distance[t])
        except:
            pass


def partial_to_full_trajectory(traj, mode, device):
    """Convert partial trajectory representation to full representation."""
    if mode == 'index':
        traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=device),
                          traj[..., -6:]), dim=-1)
    elif mode == 'thumb_middle':
        traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=device)), dim=-1)
    elif mode in ['pregrasp', 'all']:
        traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=device)), dim=-1)
    elif mode == 'thumb':
        traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 3).to(device=device)), dim=-1)
    elif mode == 'middle':
        traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=device),
                          traj[..., -3:]), dim=-1)
    return traj


def full_to_partial_trajectory(traj, mode):
    """Convert full trajectory representation to partial representation."""
    if mode == 'index':
        traj = torch.cat((traj[..., :-9], traj[..., -6:]), dim=-1)
    elif mode == 'thumb_middle':
        traj = traj[..., :-6]
    elif mode in ['pregrasp', 'all']:
        traj = traj[..., :-9]
    elif mode == 'thumb':
        traj = traj[..., :-3]
    elif mode == 'middle':
        traj = torch.cat((traj[..., :-6], traj[..., -3:]), dim=-1)
    return traj


def prepare_trajectory_for_visualization(state, traj_for_viz, exclude_index=False, dx=None):
    """Prepare trajectory for visualization by adding state and object joint."""
    if exclude_index:
        traj_for_viz = torch.cat((state[4:4 + dx].unsqueeze(0), traj_for_viz), dim=0)
    else:
        traj_for_viz = torch.cat((state[:dx].unsqueeze(0), traj_for_viz), dim=0)
    
    # Add joint for screwdriver cap
    tmp = torch.zeros((traj_for_viz.shape[0], 1), device=traj_for_viz.device)
    traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
    
    return traj_for_viz


def save_experiment_data(fpath, data, env=None):
    """Save experiment data to files."""
    # Save wrench perturbation indices if available
    if env is not None and hasattr(env, 'wrench_perturb_inds'):
        pickle.dump(env.wrench_perturb_inds, open(f"{fpath}/wrench_perturb_inds.p", "wb"))
    
    # Prepare data for saving
    data_save = deepcopy(data)
    for t in range(1, len([k for k in data.keys() if isinstance(k, int)]) + 1):
        try:
            if data_save.get(t, {}).get('plans'):
                data_save[t]['plans'] = torch.stack(data_save[t]['plans']).cpu().numpy()
            if data_save.get(t, {}).get('starts'):
                data_save[t]['starts'] = torch.stack(data_save[t]['starts']).cpu().numpy()
            if data_save.get(t, {}).get('contact_points'):
                data_save[t]['contact_points'] = torch.stack(data_save[t]['contact_points']).cpu().numpy()
            if data_save.get(t, {}).get('contact_distance'):
                data_save[t]['contact_distance'] = torch.stack(data_save[t]['contact_distance']).cpu().numpy()
            if data_save.get(t, {}).get('contact_state'):
                data_save[t]['contact_state'] = torch.stack(data_save[t]['contact_state']).cpu().numpy()
        except:
            pass
    
    # Save trajectory data
    pathlib.Path(fpath).mkdir(parents=True, exist_ok=True)
    pickle.dump(data_save, open(f"{fpath}/traj_data.p", "wb"))
    
    return data_save


def save_trajectory_data(fpath, actual_trajectory_save):
    """Save actual trajectory data."""
    with open(f'{fpath}/trajectory.pkl', 'wb') as f:
        # Filter empty lists from actual_trajectory_save
        filtered_trajectory = [i for i in actual_trajectory_save if not isinstance(i, list) or len(i) > 0]
        pickle.dump([i.cpu().numpy() for i in filtered_trajectory], f)


def get_contact_state_mappings():
    """Get contact state label mappings."""
    contact_label_to_vec = {
        'pregrasp': 0,
        'thumb_middle': 1,
        'index': 2,
        'turn': 3,
        'thumb': 4,
        'middle': 5
    }
    contact_vec_to_label = dict((v, k) for k, v in contact_label_to_vec.items())
    
    contact_state_dict = {
        'all': torch.tensor([0.0, 0.0, 0.0]),
        'index': torch.tensor([0.0, 1.0, 1.0]),
        'thumb_middle': torch.tensor([1.0, 0.0, 0.0]),
        'turn': torch.tensor([1.0, 1.0, 1.0]),
        'thumb': torch.tensor([1.0, 1.0, 0.0]),
        'middle': torch.tensor([1.0, 0.0, 1.0]),
    }
    
    contact_state_dict_flip = dict([(tuple(v.numpy()), k) for k, v in contact_state_dict.items()])
    
    return contact_label_to_vec, contact_vec_to_label, contact_state_dict, contact_state_dict_flip


def create_mode_planner_dict(env, params, device, min_force_dict, goal, AllegroScrewdriver):
    """Create dictionary of mode-specific planners."""
    mode_planner_dict = {}
    
    # Create problems for different modes
    index_problem = create_allegro_screwdriver_problem(
        'index_regrasp', goal, goal, params, env, device, min_force_dict=min_force_dict,
        AllegroScrewdriver=AllegroScrewdriver
    )
    thumb_middle_problem = create_allegro_screwdriver_problem(
        'thumb_middle_regrasp', goal, goal, params, env, device, min_force_dict=min_force_dict,
        AllegroScrewdriver=AllegroScrewdriver
    )
    all_problem = create_allegro_screwdriver_problem(
        'all_regrasp', goal, goal, params, env, device, min_force_dict=min_force_dict,
        AllegroScrewdriver=AllegroScrewdriver
    )
    
    # Create planners
    mode_planner_dict['index'] = create_planner(index_problem, params, 'recovery')
    mode_planner_dict['thumb_middle'] = create_planner(thumb_middle_problem, params, 'recovery')
    mode_planner_dict['all'] = create_planner(all_problem, params, 'recovery')
    
    return mode_planner_dict 