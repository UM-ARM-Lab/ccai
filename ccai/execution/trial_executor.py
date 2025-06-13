"""
Trial execution functionality for Allegro screwdriver recovery.
Contains the main trial execution logic and trajectory execution.
"""

import torch
import numpy as np
import time
import copy
from copy import deepcopy

from ccai.utils.allegro_utils import convert_yaw_to_sine_cosine, convert_sine_cosine_to_yaw
from ccai.utils.recovery_utils import (
    create_experiment_paths, save_goal_info, save_projection_results,
    partial_to_full_trajectory, full_to_partial_trajectory, 
    setup_and_visualize_trajectory
)
from ccai.trajectory_shortcut import shortcut_trajectory
from ccai.baselines.allegro_recovery_baselines import should_skip_diff_init


class TrajectoryExecutor:
    """Handles trajectory execution for different contact modes."""
    
    def __init__(self, params, env, sim_viz_env=None):
        self.params = params
        self.env = env
        self.sim_viz_env = sim_viz_env
        
    def execute_traj(self, planner, mode, goal=None, fname=None, initial_samples=None, 
                    recover=False, start_timestep=0, max_timesteps=None, ctrl=None, 
                    mppi_warmup=False, fpath=None, baseline_controller=None, 
                    baseline_ood_detector=None, data=None, trajectory_sampler=None,
                    trajectory_sampler_orig=None, turn_problem=None, num_fingers=None,
                    obj_dof=None, episode_num_steps=None, max_episode_num_steps=None,
                    min_force_dict=None, proj_path=None, AllegroScrewdriver=None):
        """Execute a trajectory with the given planner and mode."""
        
        rand_pct = self.params.get('rand_pct', 1/3)
        data['pre_action_likelihoods'].append([])
        data['final_likelihoods'].append([])
        data['csvto_times'].append([])
        orig_torque_perturb = self.env.external_wrench_perturb if self.params['mode'] != 'hardware' else False
        
        # reset planner
        state = self.env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + 4)[0, :15].to(device=self.params['device'])
        planned_trajectories = []
        actual_trajectory = []
        optimizer_paths = []
        contact_points = {}
        contact_distance = {}

        paths = create_experiment_paths(fpath, fname, mode, create_goal_subdir=True)
        mode_fpath = paths['mode_fpath']
        goal_fpath = paths['goal_fpath']
        save_goal_info(goal_fpath, goal, state)

        # Use baseline OOD detector if available
        if baseline_ood_detector is not None:
            id_check, final_likelihood = baseline_ood_detector.check_ood(state, recover=recover)
            dropped = baseline_ood_detector.check_drop_condition(state, recover=recover)
        else:
            # Fallback to original logic for non-baseline methods
            if recover:
                id_check, final_likelihood = True, None
            elif not self.params.get('live_recovery', False):
                id_check, final_likelihood = True, None
            else:
                if self.params['OOD_metric'] == 'likelihood':
                    id_check, final_likelihood = trajectory_sampler_orig.check_id(
                        state, self.params['likelihood_num_samples'], 
                        threshold=self.params.get('likelihood_threshold', -15))
                else:
                    id_check, final_likelihood = True, None
            dropped = False
            
        if final_likelihood is not None:
            data['pre_action_likelihoods'][-1].append(final_likelihood)

        if dropped:
            print('dropped')

        # Check if we need to trigger recovery or handle drop
        should_recover = (baseline_controller is not None and 
                         baseline_controller.is_mppi_controller() and recover and not id_check) or \
                         (baseline_controller is not None and not baseline_controller.is_mppi_controller() and not id_check) or \
                        (baseline_controller is None and not id_check)
        
        if should_recover or dropped:
            # State is OOD or dropped
            if planner is not None:
                planner.problem.data = {}
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])

            if self.params['mode'] != 'hardware':
                self.env.zero_obj_velocity()
            else:
                self.sim_viz_env.zero_obj_velocity()
            
            return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, not id_check

        # generate context from mode
        contact = -torch.ones(self.params['N'], 3).to(device=self.params['device'])
        if mode == 'thumb_middle':
            contact[:, 0] = 1
        elif mode == 'index':
            contact[:, 1] = 1
            contact[:, 2] = 1
        elif mode == 'turn':
            contact[:, :] = 1
        elif mode == 'thumb':
            contact[:, 0] = 1
            contact[:, 1] = 1
        elif mode == 'middle':
            contact[:, 0] = 1
            contact[:, 2] = 1

        recovery_params = copy.deepcopy(self.params)

        skip_diff_init = should_skip_diff_init(self.params, recover)
        planner_returns_action = False
        
        # Create baseline planner if needed
        if baseline_controller is not None and baseline_controller.is_mppi_controller() and recover:
            planner = baseline_controller.create_mppi_planner(ctrl, warmup=mppi_warmup)
            planner_returns_action = True
        else:
            planner = self._create_mode_planner(mode, planner, state, goal, num_fingers, obj_dof, 
                                              recovery_params, min_force_dict, proj_path, 
                                              max_timesteps, AllegroScrewdriver, recover)
            
        # Handle initial sampling and diffusion
        initial_samples, new_T, sim_rollouts = self._handle_initial_sampling(
            mode, trajectory_sampler, trajectory_sampler_orig, recover, skip_diff_init,
            initial_samples, state, contact, num_fingers, obj_dof, mode_fpath, planner)

        # Reset planner with new parameters
        state = self.env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + 4)[0, :15].to(device=self.params['device'])
        state = state[:planner.problem.dx]

        planner.reset(state, T=new_T, goal=goal, initial_x=initial_samples, proj_path=proj_path)
        if initial_samples is None and skip_diff_init:
            initial_samples = torch.zeros(self.params['N'], 12)
        elif initial_samples is None:
            initial_samples = planner.x.detach().clone()
        
        # Execute trajectory steps
        actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, recover = self._execute_trajectory_steps(
            planner, mode, state, goal, initial_samples, start_timestep, max_timesteps,
            num_fingers, obj_dof, episode_num_steps, max_episode_num_steps, fpath, fname,
            baseline_controller, baseline_ood_detector, data, trajectory_sampler,
            trajectory_sampler_orig, turn_problem, recover, planner_returns_action,
            planned_trajectories, actual_trajectory, optimizer_paths, contact_points, contact_distance)

        # Cleanup
        if self.params['controller'] != 'diffusion_policy':
            planner.problem.data = {}
        if self.params.get('external_wrench_perturb') and self.params['mode'] != 'hardware':
            rand_pct = self.params.get('rand_pct', 1/3)
            self.env.set_external_wrench_perturb(orig_torque_perturb, rand_pct)
            
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover

    def _create_mode_planner(self, mode, planner, state, goal, num_fingers, obj_dof, 
                           recovery_params, min_force_dict, proj_path, max_timesteps, 
                           AllegroScrewdriver, recover):
        """Create planner for specific mode."""
        from ccai.utils.recovery_utils import create_allegro_screwdriver_problem, create_planner
        
        if mode == 'index' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'index_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params, 
                self.env, self.params['device'], min_force_dict=min_force_dict,
                AllegroScrewdriver=AllegroScrewdriver)
            planner = create_planner(problem, recovery_params, 'recovery')
            
        elif mode == 'thumb_middle' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'thumb_middle_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params,
                self.env, self.params['device'], min_force_dict=min_force_dict,
                AllegroScrewdriver=AllegroScrewdriver)
            planner = create_planner(problem, recovery_params, 'recovery')
            
        elif mode == 'all' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'all_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params,
                self.env, self.params['device'], min_force_dict=min_force_dict, obj_dof=obj_dof,
                AllegroScrewdriver=AllegroScrewdriver)
            planner = create_planner(problem, recovery_params, 'recovery')
            
        elif mode == 'turn' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'turn', state[:4 * num_fingers + obj_dof], goal, self.params, self.env,
                self.params['device'], min_force_dict=min_force_dict, proj_path=proj_path,
                T_override=self.params['T_orig'] if max_timesteps is None else max_timesteps,
                AllegroScrewdriver=AllegroScrewdriver)
            planner = create_planner(problem, self.params)
        
        if recover:
            planner.problem.goal[-1] = state[-1]
            
        return planner

    def _handle_initial_sampling(self, mode, trajectory_sampler, trajectory_sampler_orig, recover,
                               skip_diff_init, initial_samples, state, contact, num_fingers, 
                               obj_dof, mode_fpath, planner):
        """Handle initial sampling with diffusion model."""
        initial_samples_0 = None
        new_T = self.params['T'] if (mode in ['index', 'thumb_middle', 'all']) else self.params['T_orig']
        
        if (self.params.get('diff_init', True) and not skip_diff_init and 
            (trajectory_sampler is not None or trajectory_sampler_orig is not None) and 
            initial_samples is None):

            sampler = trajectory_sampler if recover else trajectory_sampler_orig
            start = state.clone()

            a = time.perf_counter()
            if self.params['sine_cosine']:
                start_for_diff = convert_yaw_to_sine_cosine(start)
            else:
                start_for_diff = start
                
            ret = sampler.sample(N=self.params['N'], start=start_for_diff.reshape(1, -1),
                               H=sampler.T, constraints=contact)
            
            print('Sampling time', time.perf_counter() - a)
            initial_samples, _, likelihood = ret
                
            if self.params['sine_cosine']:
                initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                if initial_samples_0 is not None:
                    initial_samples_0 = convert_sine_cosine_to_yaw(initial_samples_0)
        
        if initial_samples is not None:
            initial_samples = initial_samples.to(device=self.params['device'])
            mode_fpath.mkdir(parents=True, exist_ok=True)

            if self.params.get('shortcut_trajectory', False) and mode != 'turn':
                s = time.perf_counter()
                initial_samples = shortcut_trajectory(initial_samples, 4 * num_fingers, obj_dof, epsilon=.04)
                print(f'Shortcut time', time.perf_counter() - s)
                new_T = initial_samples.shape[1] - 1

            sim_rollouts = torch.zeros_like(initial_samples)
            torch.cuda.empty_cache()

            if (not skip_diff_init and self.params.get('diff_init', True) and 
                (not recover or self.params.get('task_model_path', None))):
                initial_samples = full_to_partial_trajectory(initial_samples, mode)
                initial_x = initial_samples[:, 1:, :planner.problem.dx]
                initial_u = initial_samples[:, :-1, -planner.problem.du:]
                initial_samples = torch.cat((initial_x, initial_u), dim=-1)
        else:
            sim_rollouts = None
            
        return initial_samples, new_T, sim_rollouts

    def _execute_trajectory_steps(self, planner, mode, state, goal, initial_samples, start_timestep,
                                max_timesteps, num_fingers, obj_dof, episode_num_steps, 
                                max_episode_num_steps, fpath, fname, baseline_controller,
                                baseline_ood_detector, data, trajectory_sampler, trajectory_sampler_orig,
                                turn_problem, recover, planner_returns_action, planned_trajectories,
                                actual_trajectory, optimizer_paths, contact_points, contact_distance):
        """Execute the trajectory steps."""
        resample = self.params.get('diffusion_resample', False)
        plans = None
        
        # Get max steps to execute
        total_steps = planner.problem.T if max_timesteps is None else max_timesteps
        if self.params['recovery_controller'] == 'mppi' and recover:
            total_steps = max_episode_num_steps - episode_num_steps
        
        best_traj = None
        for k in range(start_timestep, total_steps):
            state = self.env.get_state()
            state_16 = state['q'].reshape(-1, 4 * num_fingers + 4).to(device=self.params['device'])[0]
            state = state_16[:15]
            print(state)

            if k > 0:
                # Check OOD and exit conditions
                exit_, recover_ = self._check_exit_conditions(k, state, baseline_ood_detector, trajectory_sampler_orig, 
                                             baseline_controller, recover, data, actual_trajectory, 
                                             planned_trajectories, initial_samples)
                if exit_:
                    return actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, recover_

            current_state = state[:4 * num_fingers + obj_dof].clone()
            state = state[:planner.problem.dx]

            # Do diffusion replanning if needed
            if self.params['controller'] != 'diffusion_policy' and plans is not None and resample:
                self._handle_diffusion_replanning(actual_trajectory, plans, mode, state, trajectory_sampler, planner, k)

            # Planning step
            s = time.perf_counter()
            if self.params['mode'] == 'hardware':
                self.sim_viz_env.set_pose(state_16.cpu())
                self.sim_viz_env.zero_obj_velocity()
                
            best_traj, plans = planner.step(state)
            csvto_time = time.perf_counter() - s
            data['csvto_times'][-1].append(csvto_time)
            print(f'Solve time for step {k+1} (global step {episode_num_steps})', csvto_time)

            planned_trajectories.append(plans)
            optimizer_paths.append(copy.deepcopy(planner.path))
            N, T, _ = plans.shape

            # Store contact information
            self._store_contact_info(planner, contact_distance, contact_points, N, T)

            # Get current state and print orientation
            state = self.env.get_state()
            state = state['q'].reshape(-1, 4 * num_fingers + 4)[0, :15].to(device=self.params['device'])
            ori = state[:15][-3:]
            print('Current ori:', ori)
            
            # Print force information
            self._print_force_info(mode, best_traj)

            # Check Q-function OOD detection
            if self._check_q_function_ood(baseline_ood_detector, state, best_traj, num_fingers, data, 
                                        planner, actual_trajectory, planned_trajectories, initial_samples, recover):
                return actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, True

            # Handle action perturbation and execution
            self._handle_action_execution(best_traj, planner_returns_action, planner, state, 
                                        num_fingers, actual_trajectory, mode, k, turn_problem, 
                                        fpath, fname, state_16, recover)

        # Stack actual trajectory
        if len(actual_trajectory) > 0:
            actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
        
        return actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, False

    def _check_exit_conditions(self, k, state, baseline_ood_detector, trajectory_sampler_orig, 
                             baseline_controller, recover, data, actual_trajectory, 
                             planned_trajectories, initial_samples):
        """Check if we should exit trajectory execution."""
        # OOD detection logic
        if baseline_ood_detector is not None:
            id_check, final_likelihood = baseline_ood_detector.check_ood(state, recover=recover)
            dropped = baseline_ood_detector.check_drop_condition(state, recover=recover)
        else:
            if recover:
                id_check, final_likelihood = True, None
            elif not self.params.get('live_recovery', False):
                id_check, final_likelihood = True, None
            else:
                if self.params['OOD_metric'] == 'likelihood':
                    id_check, final_likelihood = trajectory_sampler_orig.check_id(
                        state, self.params['likelihood_num_samples'], 
                        threshold=self.params.get('likelihood_threshold', -15))
                else:
                    id_check, final_likelihood = True, None
            dropped = False
            
        if final_likelihood is not None:
            data['pre_action_likelihoods'][-1].append(final_likelihood)
            
        if dropped:
            print('dropped')
            
        # Check exit conditions
        if (baseline_controller is not None and 
            baseline_controller.is_mppi_controller() and recover and id_check):
            print('MPPI returned state to ID. Exiting recovery loop')
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
            return True, False
        
        elif ((baseline_controller is None or not baseline_controller.is_mppi_controller()) and 
              not id_check and not dropped):
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
            if self.params['mode'] != 'hardware':
                self.env.zero_obj_velocity()
            else:
                self.sim_viz_env.zero_obj_velocity()
            return True, True
        
        elif dropped:
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
            return True, False
            
        return False, False

    def _store_contact_info(self, planner, contact_distance, contact_points, N, T):
        """Store contact distance and point information."""
        if planner.problem.data is not None and len(planner.problem.data) > 0:
            contact_distance[T] = torch.stack((
                planner.problem.data['index']['sdf'][:, -T-1:].reshape(N, T + 1),
                planner.problem.data['middle']['sdf'][:, -T-1:].reshape(N, T + 1),
                planner.problem.data['thumb']['sdf'][:, -T-1:].reshape(N, T + 1)
            ), dim=1).detach().cpu()

            contact_points[T] = torch.stack((
                planner.problem.data['index']['closest_pt_world'].reshape(N, -1, 3)[:, -T-1:],
                planner.problem.data['middle']['closest_pt_world'].reshape(N, -1, 3)[:, -T-1:],
                planner.problem.data['thumb']['closest_pt_world'].reshape(N, -1, 3)[:, -T-1:]
            ), dim=2).detach().cpu()

    def _print_force_info(self, mode, best_traj):
        """Print force information for different modes."""
        if mode == 'turn':
            index_force = torch.norm(best_traj[..., 27:30], dim=-1)
            middle_force = torch.norm(best_traj[..., 30:33], dim=-1)
            thumb_force = torch.norm(best_traj[..., 33:36], dim=-1)
            print('Middle force:', middle_force)
            print('Thumb force:', thumb_force)
            print('Index force:', index_force)
        elif mode == 'index':
            middle_force = torch.norm(best_traj[..., 27:30], dim=-1)
            thumb_force = torch.norm(best_traj[..., 30:33], dim=-1)
            print('Middle force:', middle_force)
            print('Thumb force:', thumb_force)
        elif mode == 'thumb_middle':
            index_force = torch.norm(best_traj[..., 27:30], dim=-1)
            print('Index force:', index_force)

    def _check_q_function_ood(self, baseline_ood_detector, state, best_traj, num_fingers, data, 
                            planner, actual_trajectory, planned_trajectories, initial_samples, recover):
        """Check Q-function OOD detection."""
        if (baseline_ood_detector is not None and 
            self.params['live_recovery'] and self.params['OOD_metric'] == 'q_function' and not recover):
            
            action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
            id_check, q_output = baseline_ood_detector._check_ood_q_function(state, action, num_fingers)
            
            if q_output is not None:
                data['pre_action_likelihoods'][-1].append(q_output)
                print(f'Q function output: {q_output.item():.2f}')
                
            if not id_check:
                print('OOD detected by Q function:', q_output)
                if planner is not None:
                    planner.problem.data = {}
                if len(actual_trajectory) > 0:
                    actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])

                if self.params['mode'] != 'hardware':
                    self.env.zero_obj_velocity()
                else:
                    self.sim_viz_env.zero_obj_velocity()
                planned_trajectories.pop(-1)
                return True
        return False

    def _handle_action_execution(self, best_traj, planner_returns_action, planner, state, 
                               num_fingers, actual_trajectory, mode, k, turn_problem, 
                               fpath, fname, state_16, recover):
        """Handle action computation and execution."""
        # Handle action perturbation
        if self.params.get('perturb_action', False):
            rand_pct = self.params.get('rand_pct', 1/3)
            if np.random.rand() < rand_pct:
                std = .1 if self.params.get('perturb_this_trial', False) else .0
                if mode == 'turn':
                    best_traj[:, :4 * 1] += std * torch.randn_like(best_traj[:, :4 * 1])

        xu = torch.cat((state.cpu(), best_traj[0].cpu()))
        actual_trajectory.append(xu)

        # Compute action
        if planner_returns_action or self.params['controller'] == 'diffusion_policy':
            action = best_traj
        else:
            action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
            x = best_traj[0, :planner.problem.dx + planner.problem.du]
            x = x.reshape(1, planner.problem.dx + planner.problem.du)
            action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=self.env.device)

        action = action[:, :4 * num_fingers]
        action = action.to(device=self.env.device) + state.unsqueeze(0)[:, :4 * num_fingers].to(device=self.env.device)

        # Visualization
        if ((self.params['visualize_plan'] and not recover) or 
            (self.params['visualize_recovery_plan'] and recover)):
            self._handle_visualization(best_traj, planner, state, turn_problem, fpath, fname, k, num_fingers)

        # Hardware-specific handling
        if self.params.get('external_wrench_perturb') and self.params['mode'] == 'hardware':
            rand_pct = self.params.get('rand_pct', 1/3)
            if np.random.rand() < rand_pct:
                in_or_side = np.random.rand() < .5
                loc_str = '**INTO PALM**' if in_or_side else '**TO SIDE OF OBJECT**'
                input(f'Apply perturbation now {loc_str}. Press <ENTER> to continue')
                
        if self.params['mode'] == 'hardware':
            action = action[0]
            self.sim_viz_env.set_pose(state_16.cpu())
            self.sim_viz_env.write_image()
            state = self.sim_viz_env.get_state()['q'].reshape(-1).to(device=self.params['device'])
            print(state[:15][-3:])
        elif self.params['mode'] == 'hardware_copy':
            from ccai.utils.allegro_utils import partial_to_full_state
            ros_copy_node.apply_action(partial_to_full_state(action[0], self.params['fingers']))

        self.env.step(action.to(device=self.env.device))

    def _handle_visualization(self, best_traj, planner, state, turn_problem, fpath, fname, k, num_fingers):
        """Handle trajectory visualization."""
        traj_for_viz = best_traj[:, :planner.problem.dx]
        traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
        tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)

        setup_and_visualize_trajectory(
            traj_for_viz, 
            turn_problem.contact_scenes_for_viz, 
            fpath, 
            f"{fname}/timestep_{k}",
            turn_problem.fingers, 
            turn_problem.obj_dof
        )

    def _handle_diffusion_replanning(self, actual_trajectory, plans, mode, state, trajectory_sampler, planner, k):
        """Handle diffusion model replanning."""
        executed_trajectory = torch.stack(actual_trajectory, dim=0)
        executed_trajectory = executed_trajectory.reshape(1, -1, planner.problem.dx + planner.problem.du)
        executed_trajectory = executed_trajectory.repeat(self.params['N'], 1, 1)
        executed_trajectory = partial_to_full_trajectory(executed_trajectory, mode, self.params['device'])
        plans = partial_to_full_trajectory(plans, mode, self.params['device'])
        plans = torch.cat((executed_trajectory, plans), dim=1)

        if trajectory_sampler is not None:
            contact = -torch.ones(self.params['N'], 3).to(device=self.params['device'])
            if mode == 'thumb_middle':
                contact[:, 0] = 1
            elif mode == 'index':
                contact[:, 1] = 1
                contact[:, 2] = 1
            elif mode == 'turn':
                contact[:, :] = 1
                
            with torch.no_grad():
                initial_samples, _ = trajectory_sampler.resample(
                    start=state.reshape(1, -1).repeat(self.params['N'], 1),
                    goal=None,
                    constraints=contact,
                    initial_trajectory=plans,
                    past=executed_trajectory,
                    timestep=50)
            initial_samples = full_to_partial_trajectory(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :planner.problem.dx]
            initial_u = initial_samples[:, :-1, -planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)

            planner.x = initial_samples[:, k:]
        else:
            initial_samples = None
            
        return initial_samples 