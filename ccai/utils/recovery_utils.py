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
from ccai.controller.tactile_feedback_controller import ControllerConfig, TactileFeedbackQPController
from ccai.controller.grampc_motion_contact_controller import GRAMPCMotionContactTracker

def process_contact_normals_generic(problem):
    """Get contact normals from an AllegroManipulationProblem."""
    contact_normals = []
    for finger in problem.fingers:
        contact_normals.append(problem.data[finger]["contact_n"])
    return torch.stack(contact_normals, dim=1).cpu().numpy()

def compute_reference_trajectory_spline_generic(problem, reference_trajectory, reference_trajectory_length_s, start_time, end_time, dt, all_normals=None):
    """
    Create a spline interpolation function for the reference trajectory and compute contact info.
    
    Args:
        problem: AllegroManipulationProblem instance
        reference_trajectory: Tensor or array of shape (T, state_dim)
        reference_trajectory_length_s: Length of the reference trajectory in seconds
        start_time: Start time
        end_time: End time
        dt: Timestep
        all_normals: Optional precomputed normals of shape (T, num_fingers, 3)
        
    Returns:
        spline_func, avg_normal, segment_tangents
    """
    from scipy.interpolate import interp1d
    
    # Convert to numpy for scipy interpolation
    if isinstance(reference_trajectory, torch.Tensor):
        traj_np = reference_trajectory.cpu().numpy()
    else:
        traj_np = reference_trajectory
    
    T, state_dim = traj_np.shape
    full_time_points = np.linspace(0, reference_trajectory_length_s, T)
    
    # Clamp times
    start_time = max(0, min(start_time,reference_trajectory_length_s))
    end_time = max(start_time, min(end_time, reference_trajectory_length_s))
    duration = end_time - start_time
    
    start_idx = max(0, int(start_time / dt))
    end_idx = min(T - 1, int(end_time / dt ))
    
    if start_idx == end_idx:
        end_idx = min(T - 1, start_idx + 1)
    if start_idx > 0 and start_time < full_time_points[start_idx]:
        start_idx -= 1
    if end_idx < T - 1 and end_time > full_time_points[end_idx]:
        end_idx += 1
        
    segment_traj = traj_np[start_idx:end_idx + 1]
    segment_time_points = full_time_points[start_idx:end_idx + 1]
    
    # Get normals
    if all_normals is None:
        all_normals = process_contact_normals_generic(problem)
    
    segment_normals = all_normals[start_idx:end_idx + 1]
    avg_normal = np.mean(segment_normals, axis=0)
    avg_normal = avg_normal / (np.linalg.norm(avg_normal, axis=1, keepdims=True) + 1e-6)
    
    # Compute tangents
    z_axis = np.array([0, 0, 1])
    segment_tangents = np.cross(avg_normal, z_axis)
    segment_tangents = segment_tangents / (np.linalg.norm(segment_tangents, axis=1, keepdims=True) + 1e-6)
    
    # Compute binormal
    segment_binormal = np.cross(avg_normal, segment_tangents)
    segment_binormal = segment_binormal / (np.linalg.norm(segment_binormal, axis=1, keepdims=True) + 1e-6)
    
    segment_tangents = np.stack((segment_tangents, segment_binormal), axis=2)
    
    if duration < 1e-6:
        def spline_func(t):
            return traj_np[start_idx]
    else:
        # Create spline for the segment
        # Adjusted time for spline to be relative to start_time
        relative_time_points = segment_time_points - start_time
        spline_func = interp1d(relative_time_points, segment_traj, axis=0, kind='linear', fill_value="extrapolate")
        
    return spline_func, avg_normal, segment_tangents

def get_contact_points_for_tactile_controller(problem):
    """Best-effort contact point extraction in contact_fingers order."""
    points = []
    for finger in problem.contact_fingers:
        finger_data = problem.data.get(finger, {})
        point = None
        for key in ("closest_obj_pt_object", "closest_rob_pt_object", "closest_pt_world"):
            if key in finger_data:
                point = finger_data[key]
                break
        if point is None:
            points.append(np.zeros(3))
            continue
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
        point = np.asarray(point).reshape(-1, 3)
        points.append(point[0])
    return np.asarray(points)

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


def _default_dof_reference_for_problem(env, AllegroScrewdriver, device):
    """Return the planner's full default joint reference without assuming Allegro's 16 DOFs."""
    default_dof_pos = torch.as_tensor(env.default_dof_pos, dtype=torch.float32, device=device)
    if default_dof_pos.ndim > 1:
        default_dof_pos = default_dof_pos[0]

    problem_name = getattr(AllegroScrewdriver, "__name__", "")
    if problem_name == "Proto5Screwdriver":
        if default_dof_pos.numel() != 18:
            raise ValueError(
                "Proto5Screwdriver requires an 18-DOF default_dof_pos/full_dof_reference, "
                f"got {default_dof_pos.numel()} values."
            )
        return default_dof_pos.reshape(18)

    if default_dof_pos.numel() < 16:
        raise ValueError(
            "AllegroScrewdriver requires at least 16 default DOF values, "
            f"got {default_dof_pos.numel()}."
        )
    return default_dof_pos[:16].reshape(16)


def _default_active_dof_reference_for_problem(env, AllegroScrewdriver, device):
    full_reference = _default_dof_reference_for_problem(env, AllegroScrewdriver, device)
    hand_spec = getattr(env, "hand_spec", None)
    if hand_spec is not None:
        active_joint_names = tuple(getattr(hand_spec, "active_joint_names", ()))
        all_joint_names = tuple(getattr(hand_spec, "all_joint_names", ()))
        if active_joint_names and all_joint_names:
            active_ids = [all_joint_names.index(name) for name in active_joint_names]
            return full_reference[active_ids].reshape(-1)

    problem_name = getattr(AllegroScrewdriver, "__name__", "")
    if problem_name == "Proto5Screwdriver":
        return torch.cat((full_reference[2:10], full_reference[14:18]), dim=0)
    return torch.cat((full_reference[:8], full_reference[12:16]), dim=0)


def build_pregrasp_reference_target_kwargs(params, env, device, AllegroScrewdriver):
    """Build target contact kwargs from a one-step default-pose reference problem."""
    reference_finger_q = _default_active_dof_reference_for_problem(env, AllegroScrewdriver, device)
    obj_dof = 3 if env.table_pose is not None else 1
    reference_theta = torch.zeros(obj_dof, device=device, dtype=torch.float32)
    reference_state = torch.cat((reference_finger_q, reference_theta), dim=0)
    reference_params = deepcopy(params)
    reference_params["T"] = 1
    reference_problem = create_allegro_screwdriver_problem(
        "pregrasp",
        reference_state,
        reference_state,
        reference_params,
        env,
        device,
        regrasp_fingers=params.get("fingers", ["index", "middle", "thumb"]),
        AllegroScrewdriver=AllegroScrewdriver,
        T=1,
        full_dof_goal=True,
        fingertip_contact_only=True,
        use_default_ee_locs_cost=True,
    )
    return {
        "target_contact_points_object": reference_problem.contact_points_object.detach().clone(),
        "target_contact_points_rob_link": reference_problem.contact_points_rob_link.detach().clone(),
        "use_default_ee_locs_cost": False,
        "target_contact_patch_cost_weight": params.get("pregrasp_target_contact_patch_cost_weight", 100.0),
        "target_contact_link_cost_weight": params.get("pregrasp_target_contact_link_cost_weight", 100.0),
        "target_contact_patch_mode": params.get("pregrasp_target_contact_patch_mode", "cost"),
    }


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
        'object_asset_pos': env.table_pose if env.table_pose is not None else env.obj_pose,
        'object_location': params['object_location'],
        'object_type': params['object_type'],
        'world_trans': env.world_trans,
        'obj_dof': 3 if env.table_pose is not None else 1,
        'obj_joint_dim': 1 if env.table_pose is not None else 9,
        'optimize_force': params['optimize_force'],
        'friction_coefficient': params.get('friction_coefficient', 0.95),
        'yaw_joint_friction': params.get('yaw_joint_friction', 0.0),
        'default_dof_pos': _default_dof_reference_for_problem(env, AllegroScrewdriver, device),
        'obj_gravity': params.get('obj_gravity', False),
        'contact_constraint_only': params.get('contact_constraint_only', False),
        'tactile_controller': kwargs.get('tactile_controller', False),
        'skip_csvto': kwargs.get('skip_csvto', False),
        'yaw_friction_model_type': params.get('yaw_friction_model_type', 'legacy_scalar'),
        'yaw_friction_model_params': params.get('yaw_friction_model_params'),
        'yaw_friction_model_path': params.get('yaw_friction_model_path'),
        'yaw_inertia_model_type': params.get('yaw_inertia_model_type', 'none'),
        'yaw_inertia_model_params': params.get('yaw_inertia_model_params'),
        'yaw_inertia_model_path': params.get('yaw_inertia_model_path'),
        'dt': params.get('dt', 1/12),
        'start_yaw_velocity': params.get('start_yaw_velocity', 0.0),
    }
    for optional_key in (
        'action_dt',
        'hand_joint_stiffness',
        'hand_joint_damping',
        'fingertip_contact_only',
        'object_asset_path',
        'object_mass',
        'contact_patch_link_frame_z_max',
        'filter_self_collision_query_points',
    ):
        if params.get(optional_key) is not None:
            common_params[optional_key] = params[optional_key]
    if getattr(AllegroScrewdriver, "__name__", "") == "Proto5Screwdriver":
        common_params.update({
            'full_dof_reference': common_params['default_dof_pos'],
            'robot_sdf_path_prefix': params.get('robot_sdf_path_prefix'),
            'control_wrist': params.get('proto5_control_wrist', False),
        })
    
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
    elif problem_type == 'middle_regrasp':
        specific_params = {
            'T': params['T'],
            'regrasp_fingers': ['middle'],
            'contact_fingers': ['index', 'thumb'],
            'min_force_dict': min_force_dict,
            'full_dof_goal': True,
            'proj_path': None,
            'project': True,
        }
    elif problem_type == 'thumb_regrasp':
        specific_params = {
            'T': params['T'],
            'regrasp_fingers': ['thumb'],
            'contact_fingers': ['index', 'middle'],
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
    def __init__(self, problem, params, mode):
        super().__init__(problem, params)
        self.contact_only_warmup_iters = params.get('contact_only_warmup_iters', 0)
        self.contact_only_online_iters = params.get('contact_only_online_iters', 0)
        
        self.tactile_controller_bool = params.get('tactile_controller', False)
        if self.tactile_controller_bool:
            self.contact_only_online_iters = 0
            self.contact_only_warmup_iters = 0
            self.online_iters = 0
            default_backend = 'grampc' if mode == 'simulation' else 'qp'
            self.tactile_controller_backend = params.get('tactile_controller_backend', default_backend).lower()
            
            self.controller_config = ControllerConfig(
                K_e=params.get('K_e', 113.058390),
                K_P=params.get('K_P', 3),
                K_D=params.get('K_D', 1),
                force_threshold=params.get('force_threshold', 0.2),
                dt=params.get('dt', 1/12),
                horizon_length=params.get('horizon_length', 2),
                w_f=params.get('w_f', 5.410153),
                w_q=params.get('w_q', 49.967775),
                w_p=params.get('w_p', 2.994464),
                w_u=params.get('w_u', 4.369370),
                w_ori=params.get('w_ori',0.263349),
                dq=4 * len(problem.fingers),
                df=3 * len(problem.contact_fingers),
            )
            if self.tactile_controller_backend == 'grampc':
                if mode != 'simulation':
                    raise NotImplementedError(
                        "tactile_controller_backend='grampc' is currently wired for mode='simulation' only."
                    )
                self.tactile_controller = GRAMPCMotionContactTracker(problem, self.controller_config, params=params)
            elif self.tactile_controller_backend == 'qp':
                self.tactile_controller = TactileFeedbackQPController(problem, self.controller_config)
            else:
                raise ValueError(
                    f"Unknown tactile_controller_backend={self.tactile_controller_backend!r}. "
                    "Expected 'grampc' or 'qp'."
                )
            self.t = 0
            self.dt = params.get('dt', 1/12)
        self.mode = mode
        self.default_skip_csvto = self.problem.skip_csvto

    def step(self, state, skip_optim=False, shift=True, **kwargs):
        if self.fix_T:
            new_T = None
        else:
            if self.warmed_up:
                new_T = self.problem.T - 1
            else:
                new_T = self.problem.T
        if 'q_d_init' in kwargs:
            q_d_init = kwargs['q_d_init']
            del kwargs['q_d_init']
        if 'f_ext_init' in kwargs:
            f_ext_init = kwargs['f_ext_init']
            del kwargs['f_ext_init']
        if self.default_skip_csvto != self.problem.skip_csvto and not self.warmed_up:
            self.problem.skip_csvto = self.default_skip_csvto
        # if not self.tactile_controller_bool:
        #     # Contact only
        #     self.problem.update(state, T=new_T, contact_constraint_only=True, **kwargs)
        # if (self.warmed_up and self.contact_only_online_iters > 0) or (not self.warmed_up and self.contact_only_warmup_iters > 0):
        #     if self.warmed_up:
        #         self.solver.iters = self.contact_only_online_iters
        #         resample = True if (self.iter + 1) % self.resample_steps == 0 else False
        #     else:
        #         self.solver.iters = self.contact_only_warmup_iters
        #         if self.online_iters == 0 and self.warmup_iters == 0:
        #             self.warmed_up = True
        #         resample = False

        #     path = self.solver.solve(self.x, resample, skip_optim=skip_optim)
        
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
        if not self.warmed_up and self.warmup_iters == 0:
            self.warmed_up = True
        if self.online_iters == 0:
            self.problem.skip_csvto = True
        try:
            path[0]
        except:
            path = [self.x]
        self.x = path[-1]
        self.path = path
        self.iter += 1
        best_trajectory = self.x[0].clone()
        
        all_trajectories = self.x.clone()
        if not self.tactile_controller_bool and shift:
            self.shift()
        elif self.tactile_controller_bool:
            # Need to create a version of best_trajectory with q_d instead of u
            self.best_trajectory_for_spline = best_trajectory.clone()
            self.best_trajectory_for_spline[1:, self.problem.dx:self.problem.dx+self.controller_config.dq] = best_trajectory[1:, self.problem.dx:self.problem.dx+self.controller_config.dq] + best_trajectory[:-1, :self.controller_config.dq]
            self.best_trajectory_for_spline[0, self.problem.dx:self.problem.dx+self.controller_config.dq] = best_trajectory[0, self.problem.dx:self.problem.dx+self.controller_config.dq] + self.problem.start[:self.controller_config.dq]

            self.best_trajectory_for_spline = partial_to_full_trajectory(self.best_trajectory_for_spline, self.mode, self.problem.device)
            self.problem._preprocess(self.x, tactile_controller=self.tactile_controller_bool)
            
            reference_trajectory_length_s = max(
                self.dt,
                (self.best_trajectory_for_spline.shape[0] - 1) * self.dt,
            )
            reference_trajectory_spline, avg_normal, segment_tangents = compute_reference_trajectory_spline_generic(
                self.problem,
                self.best_trajectory_for_spline,
                reference_trajectory_length_s,
                self.t,
                self.t + self.controller_config.horizon_length * self.dt,
                self.dt,
            )
            
            if self.tactile_controller_backend == 'grampc':
                contact_points = get_contact_points_for_tactile_controller(self.problem)
                self.tactile_controller.set_reference_trajectory(
                    reference_trajectory_spline,
                    normals=avg_normal,
                    contact_points=contact_points,
                )
                controller_q_d_delta = self.tactile_controller.solve(
                    self.t, state[:self.controller_config.dq], q_d_init, f_ext_init
                )
            else:
                if hasattr(self.tactile_controller, 'set_reference_trajectory'):
                    self.tactile_controller.set_reference_trajectory(reference_trajectory_spline)
                else:
                    self.tactile_controller.mpc_problem_definition.set_reference_trajectory(reference_trajectory_spline)
                controller_q_d_delta = self.tactile_controller.solve(
                    self.t,
                    state[:self.controller_config.dq],
                    q_d_init,
                    f_ext_init,
                    avg_normal,
                    segment_tangents,
                    method='precomputed_jacobians',
                )
            traj_idx = min(int(self.t / self.dt), best_trajectory.shape[0] - 1)
            if isinstance(controller_q_d_delta, np.ndarray):
                controller_q_d_delta = torch.tensor(
                    controller_q_d_delta,
                    device=best_trajectory.device,
                    dtype=best_trajectory.dtype,
                )
            best_trajectory[traj_idx, self.problem.dx:self.problem.dx+self.controller_config.dq] += controller_q_d_delta
            
        # self.x = self.problem.get_initial_xu(self.N)
        if self.tactile_controller_bool:
            traj_idx = min(int(self.t / self.dt), best_trajectory.shape[0] - 1)
            ret =  best_trajectory[traj_idx:], all_trajectories[:, traj_idx:]
            self.t += self.dt
            return ret

        else:
            return best_trajectory, all_trajectories

    
        
def create_planner(problem, mode, params, planner_type='default'):
    """Create a planner for the given problem."""
    if planner_type == 'recovery':
        recovery_params = deepcopy(params)
        recovery_params['N'] = recovery_params['recovery_N']
        recovery_params['skip_csvto'] = recovery_params['recovery_skip_csvto']
        recovery_params['online_iters'] = recovery_params['recovery_online_iters']
        recovery_params['warmup_iters'] = recovery_params['recovery_warmup_iters']
        recovery_params['tactile_controller'] = recovery_params['recovery_tactile_controller']
        return ConstraintScheduledSVGDMPC(problem, recovery_params, mode)
    else:
        return ConstraintScheduledSVGDMPC(problem, params, mode)


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
    mode_planner_dict['index'] = create_planner(index_problem, 'index', params, 'recovery')
    mode_planner_dict['thumb_middle'] = create_planner(thumb_middle_problem, 'thumb_middle', params, 'recovery')
    mode_planner_dict['all'] = create_planner(all_problem, 'all', params, 'recovery')
    
    return mode_planner_dict 
