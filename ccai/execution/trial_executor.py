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
    setup_and_visualize_trajectory, get_screwdriver_plan_camera_path
)
from ccai.utils.screwdriver_yaw_wrap import (
    unwrap_screwdriver_task_state_yaw,
    wrap_screwdriver_task_state_yaw,
)
from ccai.trajectory_shortcut import shortcut_trajectory
from ccai.baselines.allegro_recovery_baselines import should_skip_diff_init


class TrajectoryExecutor:
    """Handles trajectory execution for different contact modes."""
    
    def __init__(self, params, env, sim_viz_env=None):
        self.params = params
        self.env = env
        self.sim_viz_env = sim_viz_env

    @staticmethod
    def _planner_particle_count(planner):
        for owner in (planner, getattr(planner, "solver", None)):
            if owner is None:
                continue
            count = getattr(owner, "N", None)
            if count is not None:
                return int(count)
        x = getattr(planner, "x", None)
        if torch.is_tensor(x) and x.ndim > 0:
            return int(x.shape[0])
        return None

    def _match_initial_sample_batch(self, initial_samples, planner):
        if initial_samples is None:
            return None
        target_count = self._planner_particle_count(planner)
        if target_count is None or target_count <= 0:
            return initial_samples
        sample_count = int(initial_samples.shape[0])
        if sample_count == target_count:
            return initial_samples
        if sample_count == 0:
            raise ValueError("Cannot initialize CSVTO with an empty initial sample batch.")
        if sample_count > target_count:
            return initial_samples[:target_count]
        repeats = (target_count + sample_count - 1) // sample_count
        repeat_shape = [repeats] + [1] * (initial_samples.ndim - 1)
        return initial_samples.repeat(*repeat_shape)[:target_count]
        
    def execute_traj(self, planner, mode, env, goal=None, fname=None, initial_samples=None, 
                    recover=False, start_timestep=0, max_timesteps=None, ctrl=None, 
                    mppi_warmup=False, fpath=None, baseline_controller=None, 
                    baseline_ood_detector=None, data=None, trajectory_sampler=None,
                    trajectory_sampler_orig=None, turn_problem=None, num_fingers=None,
                    obj_dof=None, obj_joint_dim=1, episode_num_steps=None, max_episode_num_steps=None,
                    min_force_dict=None, proj_path=None, AllegroScrewdriver=None, tactile_controller=False, skip_csvto=False,
                    normal_action_policy=None, recovery_action_policy=None, reset_recovery_policy=True):
        """Execute a trajectory with the given planner and mode."""
        
        rand_pct = self.params.get('rand_pct', 1/3)
        data['pre_action_likelihoods'].append([])
        data['final_likelihoods'].append([])
        data['csvto_times'].append([])
        orig_torque_perturb = self.env.external_wrench_perturb if self.params['mode'] != 'hardware' else False
        executing_recovery = bool(recover) and mode != "turn"

        def reset_normal_policy_after_recovery():
            if (
                executing_recovery
                and normal_action_policy is not None
                and hasattr(normal_action_policy, "reset_after_recovery")
            ):
                reset_belief = bool(self.params.get("diffpf_reset_belief_after_recovery", True))
                try:
                    normal_action_policy.reset_after_recovery(self.env, reset_belief=reset_belief)
                except TypeError:
                    normal_action_policy.reset_after_recovery(self.env)
        
        # reset planner
        state = self.env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + obj_dof + obj_joint_dim)[0, :4 * num_fingers + obj_dof].to(device=self.params['device'])
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
                    task_state = wrap_screwdriver_task_state_yaw(self.params, state)
                    id_check, final_likelihood = trajectory_sampler_orig.check_id(
                        task_state, self.params['likelihood_num_samples'],
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
            reset_normal_policy_after_recovery()
            return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, not id_check, episode_num_steps

        # generate context from mode
        contact = -torch.ones(self.params.get('N_contact_plan', 16), 3).to(device=self.params['device'])
        if mode == 'thumb_middle':
            contact[:, 0] = 1
        elif mode == 'index':
            contact[:, 1] = 1
            contact[:, 2] = 1
        elif mode == 'turn':
            contact[:, :] = 1
        elif mode == 'diffpf_recovery':
            contact[:, :] = 1
        elif mode == 'thumb':
            contact[:, 0] = 1
            contact[:, 1] = 1
        elif mode == 'middle':
            contact[:, 0] = 1
            contact[:, 2] = 1

        if mode == "turn" and not recover and normal_action_policy is not None:
            actual_trajectory, planned_trajectories, recover, episode_num_steps = self._execute_normal_policy_steps(
                normal_action_policy,
                mode,
                state,
                start_timestep,
                max_timesteps,
                num_fingers,
                obj_dof,
                episode_num_steps,
                max_episode_num_steps,
                data,
                trajectory_sampler_orig,
                actual_trajectory,
                planned_trajectories,
            )
            reset_normal_policy_after_recovery()
            return actual_trajectory, planned_trajectories, initial_samples, None, optimizer_paths, contact_points, contact_distance, recover, episode_num_steps

        if mode == "diffpf_recovery" and recover and recovery_action_policy is not None:
            actual_trajectory, planned_trajectories, recover, episode_num_steps = self._execute_recovery_policy_steps(
                recovery_action_policy,
                mode,
                start_timestep,
                max_timesteps,
                num_fingers,
                obj_dof,
                episode_num_steps,
                max_episode_num_steps,
                data,
                trajectory_sampler_orig,
                actual_trajectory,
                planned_trajectories,
                reset_policy=bool(reset_recovery_policy),
            )
            if not recover:
                reset_normal_policy_after_recovery()
            return actual_trajectory, planned_trajectories, initial_samples, None, optimizer_paths, contact_points, contact_distance, recover, episode_num_steps
        if mode == "diffpf_recovery" and recover:
            raise ValueError("diffpf_recovery mode requires recovery_action_policy.")

        recovery_params = copy.deepcopy(self.params)

        skip_diff_init = should_skip_diff_init(self.params, recover)
        planner_returns_action = False
        
        # Create baseline planner if needed
        created_planner = False
        if baseline_controller is not None and baseline_controller.is_mppi_controller() and recover:
            planner = baseline_controller.create_mppi_planner(ctrl, warmup=mppi_warmup)
            planner_returns_action = True
            created_planner = True
        elif planner is None:
            planner = self._create_mode_planner(mode, planner, state, goal, num_fingers, obj_dof, 
                                              recovery_params, min_force_dict, proj_path, 
                                              max_timesteps, AllegroScrewdriver, recover, tactile_controller, skip_csvto)
            created_planner = True
            
        # Handle initial sampling and diffusion
        initial_samples, new_T, sim_rollouts = self._handle_initial_sampling(
            mode, trajectory_sampler, trajectory_sampler_orig, recover, skip_diff_init,
            initial_samples, state, contact, num_fingers, obj_dof, mode_fpath, planner)
        initial_samples = self._match_initial_sample_batch(initial_samples, planner)
        sim_rollouts = self._match_initial_sample_batch(sim_rollouts, planner)

        # Reset planner with new parameters
        state = self.env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + planner.problem.obj_dof + planner.problem.obj_joint_dim)[0, :4 * num_fingers + planner.problem.obj_dof].to(device=self.params['device'])
        state = state[:planner.problem.dx]

        if created_planner or initial_samples is not None:
            planner.reset(state, T=new_T, goal=goal, initial_x=initial_samples, proj_path=proj_path)
        else:
            planner.problem.goal[-1] = state[-1]
            planner.warmed_up = False

        if initial_samples is None and skip_diff_init:
            initial_samples = torch.zeros(self.params['N'], 16)
        elif initial_samples is None:
            initial_samples = planner.x.detach().clone()
            
        # Visualization
        if ((self.params['visualize_plan'] and not recover) or 
            (self.params['visualize_recovery_plan'] and recover)):
            self._handle_visualization(initial_samples[0], planner, state, turn_problem, fpath, fname, -1, num_fingers)
        
        # Execute trajectory steps
        actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, recover, episode_num_steps = self._execute_trajectory_steps(
            planner, mode, env, state, goal, initial_samples, start_timestep, max_timesteps,
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
        reset_normal_policy_after_recovery()
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover, episode_num_steps

    @staticmethod
    def _contact_plan_vector(mode, *, device):
        mapping = {
            "all": (0.0, 0.0, 0.0),
            "index": (0.0, 1.0, 1.0),
            "thumb_middle": (1.0, 0.0, 0.0),
            "turn": (1.0, 1.0, 1.0),
            "thumb": (1.0, 1.0, 0.0),
            "middle": (1.0, 0.0, 1.0),
            "pregrasp": (0.0, 0.0, 0.0),
            "diffpf_recovery": (1.0, 1.0, 1.0),
        }
        return torch.tensor(mapping.get(str(mode), (0.0, 0.0, 0.0)), device=device, dtype=torch.float32)

    def _ensure_execution_timeseries(self, data):
        if data is None:
            return
        for key in ("contact_state", "contact_plan", "contact_wrenches", "contact_forces", "contact_points"):
            data.setdefault(key, [])
        data.setdefault("hri_diffpf_records", [])
        data.setdefault("normal_policy_times", [])
        data.setdefault("normal_policy_likelihood_stats", [])
        data.setdefault("recovery_policy_times", [])
        data.setdefault("recovery_policy_likelihood_stats", [])

    @staticmethod
    def _as_cpu_float_tensor(value):
        return torch.as_tensor(value, dtype=torch.float32).detach().cpu()

    @staticmethod
    def _to_optional_float(value):
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            value = value.detach().cpu().reshape(-1)[0].item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float("nan")
            value = value.reshape(-1)[0].item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _latest_pre_action_likelihood(self, data):
        if data is None:
            return float("nan")
        likelihood_series = data.get("pre_action_likelihoods", [])
        if not likelihood_series:
            return float("nan")
        current_stage = likelihood_series[-1]
        if not current_stage:
            return float("nan")
        return self._to_optional_float(current_stage[-1])

    def _read_tactile_state(self):
        if hasattr(self.env, "get_tactile_observation"):
            obs = self.env.get_tactile_observation()
        else:
            obs = {}
        defaults = {
            "contact_state": torch.zeros(3, dtype=torch.float32),
            "contact_wrenches": torch.zeros(3, 6, dtype=torch.float32),
            "contact_forces": torch.zeros(3, 3, dtype=torch.float32),
            "contact_points": torch.zeros(3, 3, dtype=torch.float32),
        }
        tactile = {}
        for key, default in defaults.items():
            value = obs.get(key, default)
            tensor = self._as_cpu_float_tensor(value)
            if tensor.ndim > default.ndim:
                tensor = tensor[0]
            tactile[key] = tensor
        return tactile

    def _record_tactile_state(self, data, tactile=None):
        if data is None:
            return
        self._ensure_execution_timeseries(data)
        if tactile is None:
            tactile = self._read_tactile_state()
        for key in ("contact_state", "contact_wrenches", "contact_forces", "contact_points"):
            data[key].append(tactile[key])

    def _record_contact_plan(self, data, mode):
        if data is None:
            return
        self._ensure_execution_timeseries(data)
        data["contact_plan"].append(self._contact_plan_vector(mode, device="cpu"))

    def _state15_from_env(self, *, num_fingers, obj_dof):
        del num_fingers, obj_dof
        state = self.env.get_state()["q"]
        state_t = torch.as_tensor(state, device=self.params["device"], dtype=torch.float32)
        state_t = state_t.reshape(-1, state_t.shape[-1])[0]
        return state_t[:15].detach().cpu()

    def _full_dof_reference_from_env(self):
        if not hasattr(self.env, "get_full_dof_reference"):
            return None
        try:
            return self._as_cpu_float_tensor(self.env.get_full_dof_reference(env_id=0)).reshape(-1)
        except Exception:
            if bool(self.params.get("hardware_track_wrist_state", False)):
                raise
            return None

    def _wrist_from_full_dof_reference(self, full_dof_reference):
        if full_dof_reference is None:
            return None
        hand_spec = getattr(self.env, "hand_spec", None)
        wrist_joint_names = tuple(getattr(hand_spec, "wrist_joint_names", ()))
        all_joint_names = tuple(getattr(hand_spec, "all_joint_names", ()))
        if not wrist_joint_names:
            return None
        if all_joint_names:
            wrist_ids = [all_joint_names.index(name) for name in wrist_joint_names]
        else:
            wrist_ids = list(range(len(wrist_joint_names)))
        return full_dof_reference[wrist_ids].reshape(-1)

    def _append_hri_diffpf_record(
        self,
        data,
        *,
        pre_state15,
        post_state15,
        delta12,
        contact_plan,
        pre_tactile,
        post_tactile,
        pre_full_dof_reference=None,
        post_full_dof_reference=None,
        mode,
        recover,
        episode_num_steps,
        ood_likelihood=None,
    ):
        if data is None:
            return
        self._ensure_execution_timeseries(data)
        pre_state15 = self._as_cpu_float_tensor(pre_state15).reshape(-1)[:15]
        post_state15 = self._as_cpu_float_tensor(post_state15).reshape(-1)[:15]
        delta12 = self._as_cpu_float_tensor(delta12).reshape(-1)[:12]
        contact_plan = self._as_cpu_float_tensor(contact_plan).reshape(1, 3)
        record = {
            "states": torch.stack((pre_state15, post_state15), dim=0).numpy(),
            "actions": delta12.reshape(1, 12).numpy(),
            "contact_plan": contact_plan.numpy(),
            "contact_state": torch.stack(
                (pre_tactile["contact_state"].reshape(3), post_tactile["contact_state"].reshape(3)),
                dim=0,
            ).numpy(),
            "contact_wrenches": torch.stack(
                (pre_tactile["contact_wrenches"].reshape(3, 6), post_tactile["contact_wrenches"].reshape(3, 6)),
                dim=0,
            ).numpy(),
            "contact_forces": torch.stack(
                (pre_tactile["contact_forces"].reshape(3, 3), post_tactile["contact_forces"].reshape(3, 3)),
                dim=0,
            ).numpy(),
            "contact_points": torch.stack(
                (pre_tactile["contact_points"].reshape(3, 3), post_tactile["contact_points"].reshape(3, 3)),
                dim=0,
            ).numpy(),
            "contact_mode": str(mode),
            "recover": bool(recover),
            "episode_num_steps": int(episode_num_steps) if episode_num_steps is not None else -1,
            "trial_index": int(self.params.get("trial_index", -1)),
            "stage_index": int(self.params.get("current_stage", -1)),
            "screwdriver_friction": float(self.params.get("friction_coefficient", 1.0)),
            "yaw_joint_friction": float(
                self.params.get(
                    "yaw_joint_friction",
                    self.params.get("planner_yaw_joint_friction_override", 0.0),
                )
            ),
            "likelihood": self._to_optional_float(ood_likelihood),
        }
        if pre_full_dof_reference is not None and post_full_dof_reference is not None:
            pre_full = self._as_cpu_float_tensor(pre_full_dof_reference).reshape(-1)
            post_full = self._as_cpu_float_tensor(post_full_dof_reference).reshape(-1)
            if pre_full.numel() == post_full.numel():
                record["full_joint_pos"] = torch.stack((pre_full, post_full), dim=0).numpy()
                hand_spec = getattr(self.env, "hand_spec", None)
                full_joint_names = tuple(getattr(hand_spec, "all_joint_names", ()))
                if full_joint_names:
                    record["full_joint_names"] = full_joint_names
                pre_wrist = self._wrist_from_full_dof_reference(pre_full)
                post_wrist = self._wrist_from_full_dof_reference(post_full)
                if pre_wrist is not None and post_wrist is not None:
                    record["wrist_joint_pos"] = torch.stack((pre_wrist, post_wrist), dim=0).numpy()
                    record["wrist_joint_names"] = tuple(getattr(hand_spec, "wrist_joint_names", ()))
        data["hri_diffpf_records"].append(record)

    def _stack_actual_trajectory(self, actual_trajectory):
        if len(actual_trajectory) == 0:
            return actual_trajectory
        return torch.stack(actual_trajectory, dim=0).to(device=self.params["device"])

    def _policy_result_value(self, result, *names, default=None):
        if isinstance(result, dict):
            for name in names:
                if name in result:
                    return result[name]
            return default
        for name in names:
            if hasattr(result, name):
                return getattr(result, name)
        return default

    def _execute_normal_policy_steps(self, normal_action_policy, mode, state, start_timestep, max_timesteps,
                                     num_fingers, obj_dof, episode_num_steps, max_episode_num_steps, data,
                                     trajectory_sampler_orig, actual_trajectory, planned_trajectories):
        self._ensure_execution_timeseries(data)
        if data is not None and len(data["contact_state"]) == 0:
            self._record_tactile_state(data)

        total_steps = self.params.get("diffpf_execution_horizon", None)
        if total_steps is None:
            total_steps = self.params.get("T_orig", self.params.get("T", 1))
        if max_timesteps is not None:
            total_steps = min(int(total_steps), int(max_timesteps))
        if max_episode_num_steps is not None and episode_num_steps is not None:
            total_steps = min(int(total_steps), max(0, int(max_episode_num_steps) - int(episode_num_steps)))

        for k in range(int(start_timestep), int(total_steps)):
            state_dict = self.env.get_state()
            state_16 = state_dict["q"].reshape(-1, 4 * num_fingers + obj_dof + 1).to(device=self.params["device"])[0]
            state = state_16[:4 * num_fingers + obj_dof]

            if k > int(start_timestep):
                exit_, recover_ = self._check_exit_conditions(
                    k,
                    state,
                    None,
                    trajectory_sampler_orig,
                    None,
                    False,
                    data,
                    actual_trajectory,
                    planned_trajectories,
                    None,
                )
                if exit_:
                    return self._stack_actual_trajectory(actual_trajectory), planned_trajectories, recover_, episode_num_steps

            start_time = time.perf_counter()
            result = normal_action_policy.plan_next(self.env, k)
            elapsed = time.perf_counter() - start_time
            if data is not None:
                data["normal_policy_times"].append(elapsed)
                stats = self._policy_result_value(result, "likelihood_stats", default=None)
                data["normal_policy_likelihood_stats"].append(stats)
            print(f"DiffPF planning time for step {k + 1} (global step {episode_num_steps})", elapsed)

            delta = self._policy_result_value(result, "delta_action", "delta12", "action_delta", default=None)
            target = self._policy_result_value(
                result,
                "absolute_action_target",
                "target_action",
                "target12",
                "active_joint_target",
                default=None,
            )
            if target is None and delta is None:
                raise ValueError("normal_action_policy.plan_next(...) must return a delta action or absolute target.")
            if delta is None:
                target_t = torch.as_tensor(target, device=self.params["device"], dtype=torch.float32).reshape(-1)[:12]
                delta_t = target_t - state[:12]
            else:
                delta_t = torch.as_tensor(delta, device=self.params["device"], dtype=torch.float32).reshape(-1)[:12]
                target_t = (
                    torch.as_tensor(target, device=self.params["device"], dtype=torch.float32).reshape(-1)[:12]
                    if target is not None
                    else state[:12] + delta_t
                )

            selected_plan_rows = self._policy_result_value(result, "selected_plan_rows", "planned_rows", default=None)
            if selected_plan_rows is None:
                planned_row = torch.cat((state[:15], delta_t, torch.zeros(9, device=self.params["device"])))
                selected_plan_rows_t = planned_row.reshape(1, -1)
            else:
                selected_plan_rows_t = torch.as_tensor(
                    selected_plan_rows,
                    device=self.params["device"],
                    dtype=torch.float32,
                )
                if selected_plan_rows_t.ndim == 1:
                    selected_plan_rows_t = selected_plan_rows_t.reshape(1, -1)
            if selected_plan_rows_t.ndim == 2:
                selected_plan_rows_t = selected_plan_rows_t.reshape(1, selected_plan_rows_t.shape[0], -1)
            planned_trajectories.append(selected_plan_rows_t.detach().cpu())

            pre_state15 = state[:15].detach().cpu()
            pre_tactile = self._read_tactile_state()
            pre_full_dof_reference = self._full_dof_reference_from_env()
            actual_trajectory.append(torch.cat((pre_state15, delta_t.detach().cpu())))
            self.env.step(target_t.reshape(1, -1).to(device=self.env.device))
            if hasattr(normal_action_policy, "observe_transition"):
                normal_action_policy.observe_transition(
                    env=self.env,
                    step_idx=k,
                    state=state.detach(),
                    delta_action=delta_t.detach(),
                    target_action=target_t.detach(),
                )
            post_state15 = self._state15_from_env(num_fingers=num_fingers, obj_dof=obj_dof)
            post_tactile = self._read_tactile_state()
            post_full_dof_reference = self._full_dof_reference_from_env()
            contact_plan = self._contact_plan_vector(mode, device="cpu")
            self._record_contact_plan(data, mode)
            self._record_tactile_state(data, post_tactile)
            self._append_hri_diffpf_record(
                data,
                pre_state15=pre_state15,
                post_state15=post_state15,
                delta12=delta_t,
                contact_plan=contact_plan,
                pre_tactile=pre_tactile,
                post_tactile=post_tactile,
                pre_full_dof_reference=pre_full_dof_reference,
                post_full_dof_reference=post_full_dof_reference,
                mode=mode,
                recover=False,
                episode_num_steps=episode_num_steps,
                ood_likelihood=self._latest_pre_action_likelihood(data),
            )
            episode_num_steps += 1

        return self._stack_actual_trajectory(actual_trajectory), planned_trajectories, False, episode_num_steps

    def _check_recovery_policy_returned_id(self, state, trajectory_sampler_orig, data):
        if (
            not self.params.get("live_recovery", False)
            or self.params.get("OOD_metric") != "likelihood"
            or trajectory_sampler_orig is None
        ):
            return False, None, False
        task_state = wrap_screwdriver_task_state_yaw(self.params, state)
        id_check, likelihood = trajectory_sampler_orig.check_id(
            task_state,
            self.params["likelihood_num_samples"],
            threshold=self.params.get("likelihood_threshold", -15),
        )
        if likelihood is not None and data is not None:
            data["pre_action_likelihoods"][-1].append(likelihood)
        roll_abs = np.abs(state[-3].item())
        pitch_abs = np.abs(state[-2].item())
        drop_cutoff = np.float32(0.15).item()
        dropped = (roll_abs > drop_cutoff) or (pitch_abs > drop_cutoff)
        return bool(id_check), likelihood, bool(dropped)

    def _execute_recovery_policy_steps(self, recovery_action_policy, mode, start_timestep, max_timesteps,
                                       num_fingers, obj_dof, episode_num_steps, max_episode_num_steps, data,
                                       trajectory_sampler_orig, actual_trajectory, planned_trajectories,
                                       reset_policy=True):
        self._ensure_execution_timeseries(data)
        if data is not None and len(data["contact_state"]) == 0:
            self._record_tactile_state(data)
        if episode_num_steps is None:
            episode_num_steps = 0
        if reset_policy and hasattr(recovery_action_policy, "reset_after_recovery"):
            try:
                recovery_action_policy.reset_after_recovery(self.env, reset_belief=True)
            except TypeError:
                recovery_action_policy.reset_after_recovery(self.env)

        total_steps = self.params.get("recovery_diffpf_execution_horizon", None)
        if total_steps is None:
            total_steps = self.params.get("diffpf_execution_horizon", None)
        if total_steps is None:
            total_steps = self.params.get("T_orig", self.params.get("T", 1))
        if max_timesteps is not None:
            total_steps = min(int(total_steps), int(max_timesteps))
        if max_episode_num_steps is not None:
            total_steps = min(int(total_steps), max(0, int(max_episode_num_steps) - int(episode_num_steps)))

        for k in range(int(start_timestep), int(total_steps)):
            state_dict = self.env.get_state()
            state_16 = state_dict["q"].reshape(-1, 4 * num_fingers + obj_dof + 1).to(device=self.params["device"])[0]
            state = state_16[:4 * num_fingers + obj_dof]
            policy_step_idx = int(episode_num_steps)

            start_time = time.perf_counter()
            result = recovery_action_policy.plan_next(self.env, policy_step_idx)
            elapsed = time.perf_counter() - start_time
            if data is not None:
                data["recovery_policy_times"].append(elapsed)
                stats = self._policy_result_value(result, "likelihood_stats", default=None)
                data["recovery_policy_likelihood_stats"].append(stats)
            print(f"DiffPF recovery planning time for step {k + 1} (global step {policy_step_idx})", elapsed)

            delta = self._policy_result_value(result, "delta_action", "delta12", "action_delta", default=None)
            target = self._policy_result_value(
                result,
                "absolute_action_target",
                "target_action",
                "target12",
                "active_joint_target",
                default=None,
            )
            if target is None and delta is None:
                raise ValueError("recovery_action_policy.plan_next(...) must return a delta action or absolute target.")
            if delta is None:
                target_t = torch.as_tensor(target, device=self.params["device"], dtype=torch.float32).reshape(-1)[:12]
                delta_t = target_t - state[:12]
            else:
                delta_t = torch.as_tensor(delta, device=self.params["device"], dtype=torch.float32).reshape(-1)[:12]
                target_t = (
                    torch.as_tensor(target, device=self.params["device"], dtype=torch.float32).reshape(-1)[:12]
                    if target is not None
                    else state[:12] + delta_t
                )

            selected_plan_rows = self._policy_result_value(result, "selected_plan_rows", "planned_rows", default=None)
            if selected_plan_rows is None:
                planned_row = torch.cat((state[:15], delta_t, torch.zeros(9, device=self.params["device"])))
                selected_plan_rows_t = planned_row.reshape(1, -1)
            else:
                selected_plan_rows_t = torch.as_tensor(
                    selected_plan_rows,
                    device=self.params["device"],
                    dtype=torch.float32,
                )
                if selected_plan_rows_t.ndim == 1:
                    selected_plan_rows_t = selected_plan_rows_t.reshape(1, -1)
            if selected_plan_rows_t.ndim == 2:
                selected_plan_rows_t = selected_plan_rows_t.reshape(1, selected_plan_rows_t.shape[0], -1)
            planned_trajectories.append(selected_plan_rows_t.detach().cpu())

            pre_state15 = state[:15].detach().cpu()
            pre_tactile = self._read_tactile_state()
            pre_full_dof_reference = self._full_dof_reference_from_env()
            actual_trajectory.append(torch.cat((pre_state15, delta_t.detach().cpu())))
            self.env.step(target_t.reshape(1, -1).to(device=self.env.device))
            if hasattr(recovery_action_policy, "observe_transition"):
                recovery_action_policy.observe_transition(
                    env=self.env,
                    step_idx=policy_step_idx,
                    state=state.detach(),
                    delta_action=delta_t.detach(),
                    target_action=target_t.detach(),
                )
            post_state15 = self._state15_from_env(num_fingers=num_fingers, obj_dof=obj_dof)
            post_tactile = self._read_tactile_state()
            post_full_dof_reference = self._full_dof_reference_from_env()
            id_check, likelihood, dropped = self._check_recovery_policy_returned_id(
                post_state15.to(device=self.params["device"]),
                trajectory_sampler_orig,
                data,
            )
            contact_plan = self._contact_plan_vector(mode, device="cpu")
            self._record_contact_plan(data, mode)
            self._record_tactile_state(data, post_tactile)
            self._append_hri_diffpf_record(
                data,
                pre_state15=pre_state15,
                post_state15=post_state15,
                delta12=delta_t,
                contact_plan=contact_plan,
                pre_tactile=pre_tactile,
                post_tactile=post_tactile,
                pre_full_dof_reference=pre_full_dof_reference,
                post_full_dof_reference=post_full_dof_reference,
                mode=mode,
                recover=True,
                episode_num_steps=episode_num_steps,
                ood_likelihood=likelihood,
            )
            episode_num_steps += 1
            if dropped:
                print("dropped")
                return self._stack_actual_trajectory(actual_trajectory), planned_trajectories, False, episode_num_steps
            if id_check:
                print("DiffPF recovery returned state to ID. Exiting recovery loop")
                return self._stack_actual_trajectory(actual_trajectory), planned_trajectories, False, episode_num_steps

        return self._stack_actual_trajectory(actual_trajectory), planned_trajectories, True, episode_num_steps

    def _create_mode_planner(self, mode, planner, state, goal, num_fingers, obj_dof, 
                           recovery_params, min_force_dict, proj_path, max_timesteps, 
                           AllegroScrewdriver, recover, tactile_controller, skip_csvto):
        """Create planner for specific mode."""
        from ccai.utils.recovery_utils import create_allegro_screwdriver_problem, create_planner
        # recovery_params['warmup_iters'] = 15
        # recovery_params['skip_csvto'] = False

        if mode == 'index' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'index_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params, 
                self.env, self.params['device'], min_force_dict=min_force_dict,
                AllegroScrewdriver=AllegroScrewdriver, tactile_controller=tactile_controller, skip_csvto=recovery_params['recovery_skip_csvto'])
            planner = create_planner(problem, mode, recovery_params, 'recovery')

        elif mode == 'middle' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'middle_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params,
                self.env, self.params['device'], min_force_dict=min_force_dict,
                AllegroScrewdriver=AllegroScrewdriver, tactile_controller=tactile_controller, skip_csvto=recovery_params['skip_csvto'])
            planner = create_planner(problem, mode, recovery_params, 'recovery')
        
        elif mode == 'thumb' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'thumb_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params,
                self.env, self.params['device'], min_force_dict=min_force_dict,
                AllegroScrewdriver=AllegroScrewdriver, tactile_controller=tactile_controller, skip_csvto=recovery_params['skip_csvto'])
            planner = create_planner(problem, mode, recovery_params, 'recovery')

        elif mode == 'thumb_middle' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'thumb_middle_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params,
                self.env, self.params['device'], min_force_dict=min_force_dict,
                AllegroScrewdriver=AllegroScrewdriver, tactile_controller=tactile_controller, skip_csvto=recovery_params['recovery_skip_csvto'])
            planner = create_planner(problem, mode, recovery_params, 'recovery')
            
        elif mode == 'all' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'all_regrasp', state[:4 * num_fingers + obj_dof], goal, self.params,
                self.env, self.params['device'], min_force_dict=min_force_dict, obj_dof=obj_dof,
                AllegroScrewdriver=AllegroScrewdriver, tactile_controller=tactile_controller, skip_csvto=recovery_params['recovery_skip_csvto'])
            planner = create_planner(problem, mode, recovery_params, 'recovery')
            
        elif mode == 'turn' and planner is None:
            problem = create_allegro_screwdriver_problem(
                'turn', state[:4 * num_fingers + obj_dof], goal, self.params, self.env,
                self.params['device'], min_force_dict=min_force_dict, proj_path=proj_path,
                T_override=self.params['T_orig'] if max_timesteps is None else max_timesteps,
                AllegroScrewdriver=AllegroScrewdriver, tactile_controller=tactile_controller, skip_csvto=skip_csvto)
            planner = create_planner(problem, mode, self.params)
        
        if recover:
            planner.problem.goal[-1] = state[-1]
            
        return planner

    def _handle_initial_sampling(self, mode, trajectory_sampler, trajectory_sampler_orig, recover,
                               skip_diff_init, initial_samples, state, contact, num_fingers, 
                               obj_dof, mode_fpath, planner):
        """Handle initial sampling with diffusion model."""
        initial_samples_0 = None
        new_T = self.params['T'] if recover else self.params['T_orig']
        
        if (self.params.get('diff_init', True) and not skip_diff_init and 
            (trajectory_sampler is not None or trajectory_sampler_orig is not None) and 
            initial_samples is None):

            sampler = trajectory_sampler if recover else trajectory_sampler_orig
            start = wrap_screwdriver_task_state_yaw(self.params, state)

            a = time.perf_counter()
            if self.params['sine_cosine']:
                start_for_diff = convert_yaw_to_sine_cosine(start)
            else:
                start_for_diff = start
                
            ret = sampler.sample(N=self.params['N_contact_plan'], start=start_for_diff.reshape(1, -1),
                               H=sampler.T, constraints=contact)
            
            print('Sampling time', time.perf_counter() - a)
            initial_samples, _, likelihood = ret
            
            max_likelihood_idx = likelihood.argsort(descending=True)
            seed_count = self.params.get('recovery_N', self.params['N']) if recover else self.params['N']
            initial_samples = initial_samples[max_likelihood_idx][:int(seed_count)]
                
            if self.params['sine_cosine']:
                initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                if initial_samples_0 is not None:
                    initial_samples_0 = convert_sine_cosine_to_yaw(initial_samples_0)
            initial_samples = unwrap_screwdriver_task_state_yaw(self.params, initial_samples, yaw_idx=14)
            if initial_samples_0 is not None:
                initial_samples_0 = unwrap_screwdriver_task_state_yaw(
                    self.params,
                    initial_samples_0,
                    yaw_idx=14,
                )
        
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

    def _execute_trajectory_steps(self, planner, mode, env, state, goal, initial_samples, start_timestep,
                                max_timesteps, num_fingers, obj_dof, episode_num_steps, 
                                max_episode_num_steps, fpath, fname, baseline_controller,
                                baseline_ood_detector, data, trajectory_sampler, trajectory_sampler_orig,
                                turn_problem, recover, planner_returns_action, planned_trajectories,
                                actual_trajectory, optimizer_paths, contact_points, contact_distance):
        """Execute the trajectory steps."""
        resample = self.params.get('diffusion_resample', False)
        plans = None
        self._ensure_execution_timeseries(data)
        if data is not None and len(data["contact_state"]) == 0:
            self._record_tactile_state(data)
        
        # Get max steps to execute
        total_steps = planner.problem.T if max_timesteps is None else max_timesteps
        if self.params['recovery_controller'] == 'mppi' and recover:
            total_steps = max_episode_num_steps - episode_num_steps
        
        best_traj = None
        for k in range(start_timestep, total_steps):
            state = self.env.get_state()
            state_16 = state['q'].reshape(-1, 4 * num_fingers + obj_dof +planner.problem.obj_joint_dim).to(device=self.params['device'])[0]
            state = state_16[:4 * num_fingers + obj_dof]
            print(state)

            if k > 0:
                # Check OOD and exit conditions
                exit_, recover_ = self._check_exit_conditions(k, state, baseline_ood_detector, trajectory_sampler_orig, 
                                             baseline_controller, recover, data, actual_trajectory, 
                                             planned_trajectories, initial_samples)
                if exit_:
                    if len(actual_trajectory) > 0:
                        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
                    return actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, recover_, episode_num_steps

            state = state[:planner.problem.dx]

            # Do diffusion replanning if needed
            if self.params['controller'] != 'diffusion_policy' and plans is not None and resample:
                self._handle_diffusion_replanning(actual_trajectory, plans, mode, state, trajectory_sampler, planner, k)

            # Planning step
            s = time.perf_counter()
            if self.params['mode'] == 'hardware':
                self.sim_viz_env.set_pose(state_16.cpu())
                self.sim_viz_env.zero_obj_velocity()
                
            kwargs = {}
            if (self.params.get('tactile_controller', False) and not recover) or (self.params.get('recovery_tactile_controller', False) and recover):
                if best_traj is not None:
                    kwargs['q_d_init'] = best_traj[0, planner.problem.dx:planner.problem.dx+4*num_fingers] + planner.problem.start[:4*num_fingers]
                    kwargs['q_d_init'] = kwargs['q_d_init'][:4*num_fingers]
                else:
                    kwargs['q_d_init'] = state[:planner.problem.dx]
                
                # kwargs['f_ext_init'] = env.get_force_sensor_data(planner.problem.contact_fingers)
                kwargs['f_ext_init'] = env.get_force_sensor_data()
            best_traj, plans = planner.step(state, **kwargs)
            
            tactile_backend = str(self.params.get('tactile_controller_backend', 'grampc')).lower()
            using_grampc_tactile = (
                self.params.get('mode') == 'simulation'
                and tactile_backend == 'grampc'
                and (
                    (self.params.get('tactile_controller', False) and not recover)
                    or (self.params.get('recovery_tactile_controller', False) and recover)
                )
            )
            if not using_grampc_tactile and (
                self.params.get('contact_constraint_only', False)
                or self.params.get('solve_for_u_hat', False)
            ):
                u_hat = planner.problem.solve_for_u_hat(best_traj.unsqueeze(0), planner.solver.best_idx).squeeze(0)
                
                num_contact_fingers = len(planner.problem.contact_fingers)
                
                best_traj = torch.cat((best_traj[:, :planner.problem.dx], u_hat, best_traj[:, -num_contact_fingers*3:]), dim=-1)
                
            csvto_time = time.perf_counter() - s
            data['csvto_times'][-1].append(csvto_time)
            print(f'Solve time for step {k+1} (global step {episode_num_steps})', csvto_time)

            planned_trajectories.append(plans)
            optimizer_paths.append(copy.deepcopy(planner.path))
            N, T, _ = plans.shape

            # Store contact information
            # self._store_contact_info(planner, contact_distance, contact_points, N, T)

            # Get current state and print orientation
            state = self.env.get_state()
            state = state['q'].reshape(-1, 4 * num_fingers + 4)[0, :4 * num_fingers + obj_dof].to(device=self.params['device'])
            ori = state[:4 * num_fingers + obj_dof][-obj_dof:]
            print('Current ori:', ori)
            
            # Print force information
            self._print_force_info(mode, best_traj)

            # Check Q-function OOD detection
            if self._check_q_function_ood(baseline_ood_detector, state, best_traj, num_fingers, data, 
                                        planner, actual_trajectory, planned_trajectories, initial_samples, recover):
                if len(actual_trajectory) > 0:
                    actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
                return actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, True, episode_num_steps

            # Handle action perturbation and execution
            self._handle_action_execution(best_traj, planner_returns_action, planner, state, 
                                        num_fingers, actual_trajectory, mode, k, turn_problem, 
                                        fpath, fname, state_16, recover, data, episode_num_steps, obj_dof)
            
            # Increment episode_num_steps after successful step
            episode_num_steps += 1

        # Stack actual trajectory
        if len(actual_trajectory) > 0:
            actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=self.params['device'])
        
        return actual_trajectory, planned_trajectories, optimizer_paths, contact_points, contact_distance, False, episode_num_steps

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
                    task_state = wrap_screwdriver_task_state_yaw(self.params, state)
                    id_check, final_likelihood = trajectory_sampler_orig.check_id(
                        task_state, self.params['likelihood_num_samples'],
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
                planner.problem.data['index']['sdf'][:, -T-1:].reshape(1, T + 1),
                planner.problem.data['middle']['sdf'][:, -T-1:].reshape(1, T + 1),
                planner.problem.data['thumb']['sdf'][:, -T-1:].reshape(1, T + 1)
            ), dim=1).detach().cpu()
    
            if not planner.problem.contact_constraint_only:
                contact_points[T] = torch.stack((
                    planner.problem.data['index']['closest_pt_world'].reshape(1, -1, 3)[:, -T-1:],
                    planner.problem.data['middle']['closest_pt_world'].reshape(1, -1, 3)[:, -T-1:],
                    planner.problem.data['thumb']['closest_pt_world'].reshape(1, -1, 3)[:, -T-1:]
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
                               fpath, fname, state_16, recover, data, episode_num_steps, obj_dof):
        """Handle action computation and execution."""
        # Handle action perturbation
        if self.params.get('perturb_action', False):
            rand_pct = self.params.get('rand_pct', 1/3)
            if np.random.rand() < rand_pct:
                std = .1 if self.params.get('perturb_this_trial', False) else .0
                if mode == 'turn':
                    best_traj[:, planner.problem.dx:planner.problem.dx + planner.problem.du] += std * torch.randn_like(best_traj[:, planner.problem.dx:planner.problem.dx + planner.problem.du])

        xu = torch.cat((state.cpu(), best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du].cpu()))
        actual_trajectory.append(xu)

        # Compute action
        if planner_returns_action or self.params['controller'] == 'diffusion_policy':
            action = best_traj
        else:
            x = best_traj[0, :planner.problem.dx + planner.problem.du]
            x = x.reshape(1, planner.problem.dx + planner.problem.du)
            action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=self.env.device)

        action = action[:, :4 * num_fingers]
        action = action.to(device=self.env.device) + state.unsqueeze(0)[:, :4 * num_fingers].to(device=self.env.device)
        
        if self.params.get('perturb_action', False):
            action += torch.randn_like(action) * 0.03

        pre_state15 = state[:4 * num_fingers + obj_dof].detach().cpu()
        delta12 = (
            action.reshape(1, -1)[0, :12].detach().cpu()
            - pre_state15[:12]
        )
        pre_tactile = self._read_tactile_state()
        pre_full_dof_reference = self._full_dof_reference_from_env()

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
        post_state15 = self._state15_from_env(num_fingers=num_fingers, obj_dof=obj_dof)
        post_tactile = self._read_tactile_state()
        post_full_dof_reference = self._full_dof_reference_from_env()
        contact_plan = self._contact_plan_vector(mode, device="cpu")
        self._record_contact_plan(data, mode)
        self._record_tactile_state(data, post_tactile)
        self._append_hri_diffpf_record(
            data,
            pre_state15=pre_state15,
            post_state15=post_state15,
            delta12=delta12,
            contact_plan=contact_plan,
            pre_tactile=pre_tactile,
            post_tactile=post_tactile,
            pre_full_dof_reference=pre_full_dof_reference,
            post_full_dof_reference=post_full_dof_reference,
            mode=mode,
            recover=recover,
            episode_num_steps=episode_num_steps,
            ood_likelihood=self._latest_pre_action_likelihood(data),
        )

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
            turn_problem.obj_dof,
            full_dof_reference=turn_problem.full_dof_reference,
            joint_index=turn_problem.joint_index,
            controlled_joint_index=turn_problem.controlled_joint_index,
            camera_parameters_path=get_screwdriver_plan_camera_path(turn_problem),
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
