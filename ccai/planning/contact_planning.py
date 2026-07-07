"""
Contact planning functionality for Allegro screwdriver recovery.
Contains functions for planning recovery contact sequences.
"""

import torch
import numpy as np
import time
import pathlib
import pickle as pkl
from dataclasses import dataclass
from pprint import pprint

from ccai.utils.allegro_utils import convert_yaw_to_sine_cosine, convert_sine_cosine_to_yaw, visualize_trajectory
from ccai.utils.recovery_utils import (
    create_visualization_paths,
    get_contact_state_mappings,
    get_screwdriver_plan_camera_path,
    save_goal_info,
    save_recovery_info,
)
from ccai.utils.screwdriver_yaw_wrap import (
    unwrap_screwdriver_task_state_yaw,
    wrap_screwdriver_task_state_yaw,
)


@dataclass
class ChainedRecoveryNode:
    contact_sequence: list
    terminal_states: torch.Tensor
    trajectories: torch.Tensor = None
    csvto_seed_trajectories: torch.Tensor = None
    score: float = float("-inf")
    terminal_likelihoods: torch.Tensor = None
    recovery_likelihoods: torch.Tensor = None


class ContactPlanner:
    """Handles contact planning for recovery scenarios."""
    
    def __init__(self, params, env, trajectory_sampler=None, trajectory_sampler_orig=None, 
                 turn_problem=None, mode_planner_dict=None):
        self.params = params
        self.env = env
        self.trajectory_sampler = trajectory_sampler
        self.trajectory_sampler_orig = trajectory_sampler_orig
        self.turn_problem = turn_problem
        self.mode_planner_dict = mode_planner_dict

    def _recovery_seed_count(self):
        return int(self.params.get('recovery_N', self.params.get('N', 1)))
        
    def plan_recovery_contacts_w_model(self, state, contact_state_dict_flip, classifier):
        """Plan recovery contacts using a trained model."""
        if self.params.get('chained_recovery_contact_search', False):
            return self._plan_chained_joint_recovery_contacts(state, contact_state_dict_flip)

        start_plan_time = time.perf_counter()
        modes = ['thumb_middle', 'index'] 
        task_state = wrap_screwdriver_task_state_yaw(self.params, state)
        
        if self.params['sine_cosine']:
            start_for_diff = convert_yaw_to_sine_cosine(task_state)
        else:
            start_for_diff = task_state
            
        mean = self.trajectory_sampler.x_mean[:16]
        std = self.trajectory_sampler.x_std[:16]
        start_for_diff_normalized = (start_for_diff - mean) / std
        contact_mode_pred = torch.sigmoid(classifier(start_for_diff_normalized.reshape(1, -1)))

        contact_mode_pred = torch.round(contact_mode_pred).repeat(self.params['N'], 1)
        contact_mode_pred_tuple = tuple(contact_mode_pred[0].cpu().numpy())

        best_mode_traj, _, best_mode_likelihoods = self.trajectory_sampler.sample(
            N=contact_mode_pred.shape[0], 
            start=start_for_diff.reshape(1, -1),
            constraints=contact_mode_pred,
            H=self.trajectory_sampler.T
        )
        
        best_mode_traj = convert_sine_cosine_to_yaw(best_mode_traj)
        best_mode_traj = unwrap_screwdriver_task_state_yaw(self.params, best_mode_traj, yaw_idx=14)
        highest_likelihood_traj_idx = best_mode_likelihoods.argmax(0)
        num_fingers = len(self.params['fingers'])
        obj_dof = 3  # Assuming obj_dof is 3 for screwdriver
        goal_obj_config = best_mode_traj[highest_likelihood_traj_idx, -1, :4 * num_fingers + obj_dof].squeeze()
        mode = contact_state_dict_flip[contact_mode_pred_tuple]
        return [mode], goal_obj_config, best_mode_traj, best_mode_likelihoods, time.perf_counter() - start_plan_time

    def plan_recovery_contacts(self, state, stage, fpath, all_stage, index_regrasp_planner):
        """Plan recovery contacts using recovery model."""
        if self.params.get('chained_recovery_contact_search', False):
            return self._plan_chained_joint_recovery_contacts(state)

        start_plan_time = time.perf_counter()
        
        # If we have a recovery model, use it to get contact mode
        if self.params.get('task_model_path', None):
            # Use recovery model to get contact mode
            state = state[:15]
            task_state = wrap_screwdriver_task_state_yaw(self.params, state)
            start = convert_yaw_to_sine_cosine(task_state)
            start = start.unsqueeze(0)
            contact_mode_str_max = 'unknown'
            
            while contact_mode_str_max == 'unknown':
                initial_samples, raw_contact_mode, likelihood = self.trajectory_sampler.sample(
                    N=self.params['N_contact_plan'], start=start, H=self.params['T']+1, 
                    constraints=None, project=False)
                
                scaled_raw_contact_mode = (raw_contact_mode + 1) / 2
                contact_vec = torch.round(scaled_raw_contact_mode)
                
                all_c_mode_str = []
                for i in range(contact_vec.shape[0]):
                    try:
                        from ccai.utils.recovery_utils import get_contact_state_mappings
                        _, _, _, contact_state_dict_flip = get_contact_state_mappings()
                        c_mode_str = contact_state_dict_flip[tuple(contact_vec[i].cpu().numpy())]
                        all_c_mode_str.append(c_mode_str)
                    except:
                        print('Warning: Contact mode not found for:', contact_vec[i].cpu().numpy())
                        c_mode_str = 'unknown'
                        all_c_mode_str.append(c_mode_str)
                        
                all_c_mode_str = np.array(all_c_mode_str)
                likelihood_sort, indices = torch.sort(likelihood, descending=True)
                indices = indices.cpu().numpy()

                contact_mode_str_sort = all_c_mode_str[indices]
                contact_mode_str_max = contact_mode_str_sort[0]

                initial_samples = initial_samples[indices]
                initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                initial_samples = unwrap_screwdriver_task_state_yaw(self.params, initial_samples, yaw_idx=14)
                plan_time = time.perf_counter() - start_plan_time
                print('Likelihoods:', likelihood_sort)
                print('Contact modes', contact_mode_str_sort)

                # Compute the summed likelihood, grouped by mode
                likelihood_grouped = {}
                inds_grouped = {}
                for i, mode in enumerate(contact_mode_str_sort):
                    if mode not in likelihood_grouped:
                        likelihood_grouped[mode] = []
                        inds_grouped[mode] = []
                    likelihood_grouped[mode].append(likelihood_sort[i].item())
                    inds_grouped[mode].append(indices[i])
                    
                likelihood_mean = {}
                likelihood_sum = {}
                for mode in likelihood_grouped:
                    likelihood_mean[mode] = np.mean(np.exp(likelihood_grouped[mode]))
                    likelihood_sum[mode] = np.sum(np.exp(likelihood_grouped[mode]))
                    
                print('Likelihood grouped mean:', likelihood_mean)
                print('Likelihood grouped sum:', likelihood_sum)

                # Pick mode with highest likelihood sum
                if 'unknown' in likelihood_sum:
                    likelihood_sum.pop('unknown', None)
                    if not likelihood_sum:
                        contact_mode_str_max = 'unknown'
                        print('All modes are unknown')
                        continue
                        
                contact_mode_str_max = max(likelihood_sum, key=likelihood_sum.get)
                print('Contact mode with highest likelihood sum:', contact_mode_str_max)

                contact_mode_str_max = contact_mode_str_sort[0]
                print('Contact mode with highest likelihood trajectory:', contact_mode_str_max)

                best_traj_idx = inds_grouped[contact_mode_str_max][0]
                goal_config = initial_samples[best_traj_idx, -1, :15]
            
                # Use the goal from the planner if using the task model to diffuse the goal
                if self.params['task_diffuse_goal']:
                    goal = index_regrasp_planner.problem.goal.clone()
                    goal[-1] = state[-1]
                    initial_samples = []
                    pre_goal_viz = time.perf_counter()
                    plan_time += time.perf_counter() - pre_goal_viz
                    
                    if self.params['visualize_contact_plan']:
                        self._visualize_goal(fpath, all_stage, goal, state)

            # Visualize all planning samples if requested
            if self.params.get('visualize_recovery_planning_samples', False):
                viz_fpath = pathlib.Path(fpath) / f"recovery_stage_{all_stage}" / "planning_samples"
                viz_fpath.mkdir(parents=True, exist_ok=True)
                
                # Visualize each trajectory in initial_samples
                for i, (traj, mode) in enumerate(zip(initial_samples, contact_mode_str_sort[:self.params['N']])):
                    traj_for_viz = traj[:, :self.turn_problem.dx]
                    traj_for_viz = torch.cat((state[:self.turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                    tmp = torch.zeros((traj_for_viz.shape[0], 1), device=traj.device)
                    traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                    
                    sample_fpath = viz_fpath / f"{mode}_sample_{i}"
                    sample_fpath.mkdir(parents=True, exist_ok=True)
                    sample_fpath_img = sample_fpath / "img"
                    sample_fpath_img.mkdir(parents=True, exist_ok=True)
                    sample_fpath_gif = sample_fpath / "gif"
                    sample_fpath_gif.mkdir(parents=True, exist_ok=True)
                    visualize_trajectory(
                        traj_for_viz,
                        self.turn_problem.contact_scenes_for_viz,
                        sample_fpath,
                        self.turn_problem.fingers,
                        self.turn_problem.obj_dof + 1,
                        full_dof_reference=self.turn_problem.full_dof_reference,
                        joint_index=self.turn_problem.joint_index,
                        controlled_joint_index=self.turn_problem.controlled_joint_index,
                        camera_parameters_path=get_screwdriver_plan_camera_path(self.turn_problem),
                    )
            
            seed_count = self._recovery_seed_count()
            if seed_count > 1:
                initial_samples = initial_samples[:seed_count]
            else:
                # Use highest likelihood trajectory of highst sum likelihood mode
                initial_samples = initial_samples[best_traj_idx:best_traj_idx+1]
            
            return [contact_mode_str_max], goal_config, initial_samples, likelihood, plan_time
        # If we don't have a recovery model, use the task model to plan contacts
        else:
            return self.plan_recovery_contacts_offline(state, stage, fpath, all_stage)

    @torch.no_grad()
    def _plan_chained_joint_recovery_contacts(self, state, contact_state_dict_flip=None):
        """Plan a contact-mode sequence by chaining joint recovery diffusion samples."""
        if self.trajectory_sampler is None:
            raise ValueError("chained_recovery_contact_search requires trajectory_sampler.")
        if self.trajectory_sampler_orig is None:
            raise ValueError("chained_recovery_contact_search requires trajectory_sampler_orig for terminal OOD scoring.")

        start_plan_time = time.perf_counter()
        state_dim = self._state_dim()
        threshold = self._chained_recovery_likelihood_threshold()
        max_depth = self.params.get('max_recovery_stages', 1)

        root_state = state[:state_dim].reshape(1, -1)
        frontier = [ChainedRecoveryNode(contact_sequence=[], terminal_states=root_state)]
        best_node = None

        for depth in range(max_depth):
            children = self._expand_chained_recovery_frontier(
                frontier,
                contact_state_dict_flip=contact_state_dict_flip,
            )
            if not children:
                break

            best_child = max(children, key=lambda node: node.score)
            if best_node is None or best_child.score > best_node.score:
                best_node = best_child

            terminating_children = [child for child in children if child.score > threshold]
            if terminating_children:
                selected = max(terminating_children, key=lambda node: node.score)
                print('Chained recovery terminated at depth:', depth + 1)
                print('Chained recovery contact sequence:', selected.contact_sequence)
                print('Chained recovery terminal task likelihood:', selected.score)
                return self._format_chained_recovery_result(
                    selected,
                    time.perf_counter() - start_plan_time,
                )

            print('Chained recovery continuing with frontier size:', len(children))
            print('Chained recovery best contact sequence so far:', best_child.contact_sequence)
            print('Chained recovery best terminal task likelihood:', best_child.score)
            frontier = children

        if best_node is None:
            raise ValueError("Joint recovery model did not produce any valid contact modes for chained search.")

        print('Chained recovery reached max depth; returning best visited sequence:', best_node.contact_sequence)
        print('Chained recovery best terminal task likelihood:', best_node.score)
        return self._format_chained_recovery_result(
            best_node,
            time.perf_counter() - start_plan_time,
        )

    def _expand_chained_recovery_node(self, node, contact_state_dict_flip=None):
        return self._expand_chained_recovery_frontier([node], contact_state_dict_flip=contact_state_dict_flip)

    def _chained_recovery_likelihood_threshold(self):
        return self.params.get(
            'chained_recovery_likelihood_threshold',
            self.params.get('likelihood_threshold', -15),
        )

    def _expand_chained_recovery_frontier(self, frontier, contact_state_dict_flip=None):
        if not frontier:
            return []

        state_dim = self._state_dim()
        samples_per_node = self.params['N_contact_plan']
        starts_for_diff = torch.cat(
            [self._prepare_chained_expansion_starts(node.terminal_states) for node in frontier],
            dim=0,
        )
        sample_count = starts_for_diff.shape[0]

        trajectories, raw_contact_modes, recovery_likelihoods = self.trajectory_sampler.sample(
            N=sample_count,
            start=starts_for_diff,
            H=self.trajectory_sampler.T,
            constraints=None,
            project=False,
        )
        if raw_contact_modes is None:
            raise ValueError(
                "chained_recovery_contact_search requires a joint recovery model that diffuses contact modes."
            )
        if recovery_likelihoods is None:
            raise ValueError(
                "chained_recovery_contact_search requires recovery-model likelihoods for resampling."
            )

        recovery_likelihoods = recovery_likelihoods.reshape(-1)
        if trajectories.shape[0] != sample_count:
            raise ValueError(
                "Joint recovery sampler returned an unexpected number of trajectories for batched chained search: "
                f"expected {sample_count}, got {trajectories.shape[0]}."
            )
        if raw_contact_modes.shape[0] != sample_count:
            raise ValueError(
                "Joint recovery sampler returned an unexpected number of contact modes for batched chained search: "
                f"expected {sample_count}, got {raw_contact_modes.shape[0]}."
            )
        if recovery_likelihoods.shape[0] != sample_count:
            raise ValueError(
                "Joint recovery sampler returned an unexpected number of likelihoods for batched chained search: "
                f"expected {sample_count}, got {recovery_likelihoods.shape[0]}."
            )

        modes = self._decode_contact_modes(raw_contact_modes, contact_state_dict_flip)
        if self.params.get('sine_cosine', False):
            trajectories_for_scoring = convert_sine_cosine_to_yaw(trajectories)
        else:
            trajectories_for_scoring = trajectories
        trajectories_for_scoring = unwrap_screwdriver_task_state_yaw(
            self.params,
            trajectories_for_scoring,
            yaw_idx=14,
        )

        child_specs = []
        terminal_state_chunks = []
        for node_idx, node in enumerate(frontier):
            chunk_start = node_idx * samples_per_node
            chunk_end = chunk_start + samples_per_node
            chunk_modes = modes[chunk_start:chunk_end]
            chunk_likelihoods = recovery_likelihoods[chunk_start:chunk_end]
            chunk_trajectories = trajectories_for_scoring[chunk_start:chunk_end]
            chunk_csvto_seed_trajectories = chunk_trajectories

            valid_indices = [i for i, mode in enumerate(chunk_modes) if mode is not None]
            if len(valid_indices) == 0:
                print('No valid contact modes were diffused for chained recovery expansion')
                continue

            valid_indices = torch.tensor(valid_indices, device=recovery_likelihoods.device, dtype=torch.long)
            valid_likelihoods = chunk_likelihoods[valid_indices]
            valid_trajectories = chunk_trajectories[valid_indices]
            valid_modes = [chunk_modes[i] for i in valid_indices.cpu().tolist()]

            resampled_local_indices = self._resample_indices_from_recovery_likelihoods(
                valid_likelihoods,
                num_samples=samples_per_node,
            )
            resampled_trajectories = valid_trajectories[resampled_local_indices]
            resampled_likelihoods = valid_likelihoods[resampled_local_indices]
            resampled_modes = [valid_modes[i] for i in resampled_local_indices.cpu().tolist()]

            for mode in dict.fromkeys(valid_modes):
                mode_valid_indices = [
                    i for i, candidate_mode in enumerate(valid_modes) if candidate_mode == mode
                ]
                resampled_mode_positions = [
                    i for i, candidate_mode in enumerate(resampled_modes) if candidate_mode == mode
                ]
                if len(resampled_mode_positions) == 0:
                    continue

                mode_indices = torch.tensor(
                    resampled_mode_positions,
                    device=resampled_trajectories.device,
                    dtype=torch.long,
                )
                score_indices = torch.tensor(
                    mode_valid_indices,
                    device=valid_trajectories.device,
                    dtype=torch.long,
                )
                source_position_by_valid_index = {
                    valid_index: i for i, valid_index in enumerate(mode_valid_indices)
                }
                resampled_score_indices = torch.tensor(
                    [
                        source_position_by_valid_index[i]
                        for i in resampled_local_indices[mode_indices].cpu().tolist()
                    ],
                    device=valid_trajectories.device,
                    dtype=torch.long,
                )
                mode_trajectories = resampled_trajectories[mode_indices]
                terminal_states_for_score = valid_trajectories[score_indices, -1, :state_dim]
                child_specs.append(
                    (
                        node,
                        mode,
                        terminal_states_for_score,
                        valid_likelihoods[score_indices],
                        resampled_score_indices,
                        mode_trajectories,
                        resampled_likelihoods[mode_indices],
                        chunk_csvto_seed_trajectories,
                    )
                )
                terminal_state_chunks.append(terminal_states_for_score)

        if not child_specs:
            return []

        all_terminal_states = torch.cat(terminal_state_chunks, dim=0)
        all_terminal_likelihoods = self._terminal_task_likelihoods(all_terminal_states)

        children = []
        likelihood_offset = 0
        for (
            node,
            mode,
            terminal_states_for_score,
            recovery_likelihoods_for_score,
            resampled_score_indices,
            mode_trajectories,
            mode_recovery_likelihoods,
            csvto_seed_trajectories,
        ) in child_specs:
            likelihood_count = terminal_states_for_score.shape[0]
            terminal_likelihoods_for_score = all_terminal_likelihoods[
                likelihood_offset:likelihood_offset + likelihood_count
            ]
            likelihood_offset += likelihood_count
            score = self._weighted_terminal_task_score(
                recovery_likelihoods_for_score,
                terminal_likelihoods_for_score,
            ).item()
            terminal_likelihoods = terminal_likelihoods_for_score[resampled_score_indices]
            terminal_states = mode_trajectories[:, -1, :state_dim]
            children.append(
                ChainedRecoveryNode(
                    contact_sequence=list(node.contact_sequence) + [mode],
                    terminal_states=terminal_states,
                    trajectories=mode_trajectories,
                    csvto_seed_trajectories=csvto_seed_trajectories,
                    score=score,
                    terminal_likelihoods=terminal_likelihoods,
                    recovery_likelihoods=mode_recovery_likelihoods,
                )
            )

        return children

    def _weighted_terminal_task_score(self, recovery_likelihoods, terminal_likelihoods):
        recovery_likelihoods = recovery_likelihoods.to(
            device=terminal_likelihoods.device,
            dtype=terminal_likelihoods.dtype,
        )
        return torch.logsumexp(
            recovery_likelihoods + terminal_likelihoods,
            dim=0,
        ) - torch.logsumexp(recovery_likelihoods, dim=0)

    def _prepare_chained_expansion_starts(self, terminal_states):
        starts = self._prepare_chained_start_batch(terminal_states)
        if starts.shape[0] == self.params['N_contact_plan']:
            return starts
        if starts.shape[0] != 1:
            raise ValueError(
                "Chained recovery start batching expected one start or N_contact_plan starts, "
                f"got {starts.shape[0]} starts."
            )
        return starts.repeat(self.params['N_contact_plan'], 1)

    def _prepare_chained_start_batch(self, terminal_states):
        states = wrap_screwdriver_task_state_yaw(self.params, terminal_states)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        if states.shape[0] > 1:
            sample_count = self.params['N_contact_plan']
            indices = torch.arange(sample_count, device=states.device) % states.shape[0]
            states = states[indices]

        if self.params.get('sine_cosine', False):
            return convert_yaw_to_sine_cosine(states)
        return states

    def _decode_contact_modes(self, raw_contact_modes, contact_state_dict_flip=None):
        if contact_state_dict_flip is None:
            _, _, _, contact_state_dict_flip = get_contact_state_mappings()

        raw_contact_modes = raw_contact_modes.reshape(raw_contact_modes.shape[0], -1, raw_contact_modes.shape[-1])
        if raw_contact_modes.shape[1] > 1:
            raw_contact_modes = raw_contact_modes[:, 0]
        else:
            raw_contact_modes = raw_contact_modes.squeeze(1)

        contact_vec = torch.round((raw_contact_modes + 1) / 2)
        modes = []
        for i in range(contact_vec.shape[0]):
            key = tuple(contact_vec[i].detach().cpu().numpy())
            mode = contact_state_dict_flip.get(key)
            if mode is None:
                print('Warning: Contact mode not found for:', contact_vec[i].detach().cpu().numpy())
            modes.append(mode)
        return modes

    def _resample_indices_from_recovery_likelihoods(self, recovery_likelihoods, num_samples):
        weights = torch.exp(recovery_likelihoods)
        weights = weights / weights.sum()
        if not torch.isfinite(weights).all():
            raise ValueError("Non-finite recovery likelihood weights in chained recovery search.")
        return torch.multinomial(weights, num_samples=num_samples, replacement=True)

    def _terminal_task_likelihoods(self, terminal_states):
        if terminal_states.numel() == 0:
            return torch.empty(0, device=terminal_states.device, dtype=terminal_states.dtype)

        flat_terminal_states = terminal_states.reshape(-1, terminal_states.shape[-1])
        unique_states, inverse_indices = torch.unique(
            flat_terminal_states,
            dim=0,
            return_inverse=True,
        )

        if hasattr(self.trajectory_sampler_orig, 'sample'):
            unique_likelihoods = self._batched_terminal_task_likelihoods(unique_states)
        else:
            unique_likelihoods = self._serial_terminal_task_likelihoods(unique_states)

        return unique_likelihoods[inverse_indices].reshape(terminal_states.shape[:-1])

    def _batched_terminal_task_likelihoods(self, terminal_states):
        terminal_states = wrap_screwdriver_task_state_yaw(self.params, terminal_states)
        num_states = terminal_states.shape[0]
        num_likelihood_samples = self.params['likelihood_num_samples']
        start = convert_yaw_to_sine_cosine(terminal_states)
        batched_start = start.repeat_interleave(num_likelihood_samples, dim=0)
        constraint_count = num_states * num_likelihood_samples
        constraints = torch.ones(
            constraint_count,
            3,
            device=terminal_states.device,
            dtype=terminal_states.dtype,
        )

        with torch.no_grad():
            _, _, likelihood = self.trajectory_sampler_orig.sample(
                N=constraint_count,
                H=self.trajectory_sampler_orig.T,
                start=batched_start,
                constraints=constraints,
            )

        likelihood = likelihood.reshape(num_states, num_likelihood_samples).mean(dim=1)
        if getattr(self.trajectory_sampler_orig, 'rl_adjustment', False):
            residual, _ = self.trajectory_sampler_orig.gp.gp_model.predict(start, fast_mode=True)
            likelihood = likelihood + residual.reshape(-1).to(device=likelihood.device, dtype=likelihood.dtype)

        return likelihood.to(device=terminal_states.device, dtype=terminal_states.dtype)

    def _serial_terminal_task_likelihoods(self, terminal_states):
        likelihoods = []
        for terminal_state in terminal_states:
            task_terminal_state = wrap_screwdriver_task_state_yaw(self.params, terminal_state)
            likelihood = self.trajectory_sampler_orig.check_id(
                task_terminal_state,
                self.params['likelihood_num_samples'],
                threshold=self.params.get('likelihood_threshold', -15),
                likelihood_only=True,
            )
            if isinstance(likelihood, tuple):
                likelihood = likelihood[0]
            likelihoods.append(float(likelihood))
        return torch.tensor(likelihoods, device=terminal_states.device, dtype=terminal_states.dtype)

    def _format_chained_recovery_result(self, node, plan_time):
        if node.terminal_likelihoods is None:
            best_idx = 0
            likelihoods = None
        else:
            best_idx = torch.argmax(node.terminal_likelihoods).item()
            likelihoods = node.terminal_likelihoods
        goal_config = node.terminal_states[best_idx].clone()
        initial_samples = node.csvto_seed_trajectories
        if initial_samples is None:
            initial_samples = node.trajectories
        return node.contact_sequence, goal_config, initial_samples, likelihoods, plan_time

    def _state_dim(self):
        obj_dof = self.turn_problem.obj_dof if self.turn_problem is not None else 3
        return 4 * len(self.params['fingers']) + obj_dof

    def plan_recovery_contacts_offline(self, state, stage, fpath, all_stage):
        """Plan recovery contacts using CSVTO + likelihood estimation"""
        plan_time = 0
        distances = []

        # Hardcoded modes for screwdriver for now
        modes = ['thumb_middle', 'index']
        goal = self.mode_planner_dict['index'].problem.goal.clone()
        goal[-1] = state[-1]
        initial_samples = []
        
        pre_goal_viz = time.perf_counter()
        plan_time += time.perf_counter() - pre_goal_viz
        
        if self.params['visualize_contact_plan']:
            self._visualize_goal(fpath, all_stage, goal, state)
            
        post_goal_viz = time.perf_counter()
        dist_min = 5e-3
        mode_skip = []
        planner = self.mode_planner_dict['index']
        
        
        # Filter contact modes based on which fingers are in contact with the object. Contact fingers for each mode must be less than dist_min away from the object.
        num_fingers = len(self.params['fingers'])
        obj_dof = self.turn_problem.obj_dof
        cur_q = state[:4 * num_fingers]
        cur_theta = state[4 * num_fingers: 4 * num_fingers + obj_dof]
        planner.problem._preprocess_fingers(cur_q[None, None], cur_theta[None, None], compute_closest_obj_point=True)
        
        print(planner.problem.data['index']['sdf'], planner.problem.data['middle']['sdf'], planner.problem.data['thumb']['sdf'])
        
        for mode in modes:
            if mode == 'index':
                if planner.problem.data['thumb']['sdf'].max() > dist_min or planner.problem.data['middle']['sdf'].max() > dist_min:
                    mode_skip.append(mode)
            elif mode == 'thumb_middle':
                if planner.problem.data['index']['sdf'].max() > dist_min:
                    mode_skip.append(mode)

        if 'index' in mode_skip and 'thumb_middle' in mode_skip:
            mode_skip = []
            
        pre_mode_loop = time.perf_counter()
        plan_time += pre_mode_loop - post_goal_viz
        
        for mode in modes:
            if mode in mode_skip:
                distances.append(float('inf'))
                initial_samples.append(None)
                continue
                
            begin_mode_loop = time.perf_counter()
            planner = self.mode_planner_dict[mode]
            planner.reset(state, T=self.params['T'], goal=goal)
            planner.warmup_iters = self.params['recovery_warmup_iters']

            # Run CSVTO to plan trajectory
            xu, plans = planner.step(state, shift=False)
            planner.problem.data = {}
            planner.warmup_iters = 0
            initial_samples.append(plans)
            
            x = xu[:, :planner.problem.num_fingers * 4 + planner.problem.obj_dof]
            end = x[-1]
            
            # Estimate likelihood of the end state
            task_end = wrap_screwdriver_task_state_yaw(self.params, end)
            likelihood, samples = self.trajectory_sampler_orig.check_id(
                task_end, self.params['likelihood_num_samples'], likelihood_only=True,
                return_samples=True, threshold=self.params.get('likelihood_threshold', -15))
            distances.append(-likelihood)
            
            end_mode_loop = time.perf_counter()
            plan_time += end_mode_loop - begin_mode_loop
            
            # Save planning results
            viz_fpath = pathlib.Path(fpath) / f"recovery_stage_{all_stage}" / mode
            viz_fpath.mkdir(parents=True, exist_ok=True)
            save_recovery_info(viz_fpath, plans, samples, likelihood)
            
            if self.params['visualize_contact_plan']:
                self._visualize_contact_plan(viz_fpath, x, state, planner)

        dists_dict = dict(zip(modes, distances))
        pprint(dists_dict)

        # Return the mode that achieves the highest likelihood state
        if all(d == float('inf') for d in distances):
            return ['turn'], None, plan_time
        else:
            return [modes[np.argmin(distances)]], initial_samples[np.argmin(distances)], plan_time

    def _visualize_goal(self, fpath, all_stage, goal, state):
        """Visualize goal for contact planning."""
        viz_fpath = pathlib.Path(fpath) / f"recovery_stage_{all_stage}" / "goal"
        viz_fpath.mkdir(parents=True, exist_ok=True)
        
        img_fpath, gif_fpath = viz_fpath / 'img', viz_fpath / 'gif'
        img_fpath.mkdir(parents=True, exist_ok=True)
        gif_fpath.mkdir(parents=True, exist_ok=True)
        
        tmp = torch.zeros((2, 1), device=goal.device)
        traj_for_viz = torch.cat((state.unsqueeze(0), goal.unsqueeze(0)), dim=0)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
        
        visualize_trajectory(
            traj_for_viz,
            self.turn_problem.contact_scenes_for_viz,
            viz_fpath,
            self.turn_problem.fingers,
            self.turn_problem.obj_dof + 1,
            full_dof_reference=self.turn_problem.full_dof_reference,
            joint_index=self.turn_problem.joint_index,
            controlled_joint_index=self.turn_problem.controlled_joint_index,
            camera_parameters_path=get_screwdriver_plan_camera_path(self.turn_problem),
        )
        
        save_goal_info(viz_fpath, goal, state)

    def _visualize_contact_plan(self, viz_fpath, x, state, planner):
        """Visualize contact plan trajectory."""
        traj_for_viz = x[:, :planner.problem.dx]

        traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            
        tmp = torch.zeros((traj_for_viz.shape[0], 1), device=x.device)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)

        img_fpath, gif_fpath = viz_fpath / 'img', viz_fpath / 'gif'
        img_fpath.mkdir(parents=True, exist_ok=True)
        gif_fpath.mkdir(parents=True, exist_ok=True)
        
        visualize_trajectory(
            traj_for_viz,
            self.turn_problem.contact_scenes_for_viz,
            viz_fpath,
            self.turn_problem.fingers,
            self.turn_problem.obj_dof + 1,
            full_dof_reference=self.turn_problem.full_dof_reference,
            joint_index=self.turn_problem.joint_index,
            controlled_joint_index=self.turn_problem.controlled_joint_index,
            camera_parameters_path=get_screwdriver_plan_camera_path(self.turn_problem),
        ) 
