"""
Contact planning functionality for Allegro screwdriver recovery.
Contains functions for planning recovery contact sequences.
"""

import torch
import numpy as np
import time
import pathlib
import pickle as pkl
from pprint import pprint

from ccai.utils.allegro_utils import convert_yaw_to_sine_cosine, convert_sine_cosine_to_yaw, visualize_trajectory
from ccai.utils.recovery_utils import create_visualization_paths, save_goal_info, save_recovery_info


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
        
    def plan_recovery_contacts_w_model(self, state, contact_state_dict_flip, classifier):
        """Plan recovery contacts using a trained model."""
        start_plan_time = time.perf_counter()
        modes = ['thumb_middle', 'index'] 
        
        if self.params['sine_cosine']:
            start_for_diff = convert_yaw_to_sine_cosine(state)
        else:
            start_for_diff = state
            
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
        highest_likelihood_traj_idx = best_mode_likelihoods.argmax(0)
        num_fingers = len(self.params['fingers'])
        obj_dof = 3  # Assuming obj_dof is 3 for screwdriver
        goal_obj_config = best_mode_traj[highest_likelihood_traj_idx, -1, :4 * num_fingers + obj_dof].squeeze()
        mode = contact_state_dict_flip[contact_mode_pred_tuple]
        return [mode], goal_obj_config, best_mode_traj, best_mode_likelihoods, time.perf_counter() - start_plan_time

    def plan_recovery_contacts(self, state, stage, fpath, all_stage, index_regrasp_planner):
        """Plan recovery contacts using recovery model."""
        start_plan_time = time.perf_counter()
        
        # If we have a recovery model, use it to get contact mode
        if self.params.get('model_path_orig', None):
            # Use recovery model to get contact mode
            state = state[:15]
            start = convert_yaw_to_sine_cosine(state)
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

                # contact_mode_str_max = contact_mode_str_sort[0]
                # print('Contact mode with highest likelihood trajectory:', contact_mode_str_max)

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

            initial_samples = initial_samples[:self.params['N']]
            return [contact_mode_str_max], goal_config, initial_samples, likelihood, plan_time
        # If we don't have a recovery model, use the task model to plan contacts
        else:
            return self.plan_recovery_contacts_offline(state, stage, fpath, all_stage)

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
        obj_dof = 3
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
            planner.warmup_iters = self.params['warmup_iters']

            # Run CSVTO to plan trajectory
            xu, plans = planner.step(state)
            planner.problem.data = {}
            planner.warmup_iters = 0
            initial_samples.append(plans)
            
            x = xu[:, :planner.problem.num_fingers * 4 + planner.problem.obj_dof]
            end = x[-1]
            
            # Estimate likelihood of the end state
            likelihood, samples = self.trajectory_sampler_orig.check_id(
                end, self.params['likelihood_num_samples'], likelihood_only=True, 
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
        
        visualize_trajectory(traj_for_viz, self.turn_problem.contact_scenes_for_viz, viz_fpath,
                            self.turn_problem.fingers, self.turn_problem.obj_dof + 1)
        
        save_goal_info(viz_fpath, goal, state)

    def _visualize_contact_plan(self, viz_fpath, x, state, planner):
        """Visualize contact plan trajectory."""
        traj_for_viz = x[:, :planner.problem.dx]
        
        if self.params['exclude_index']:
            traj_for_viz = torch.cat((state[4:4 + planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
        else:
            traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            
        tmp = torch.zeros((traj_for_viz.shape[0], 1), device=x.device)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)

        img_fpath, gif_fpath = viz_fpath / 'img', viz_fpath / 'gif'
        img_fpath.mkdir(parents=True, exist_ok=True)
        gif_fpath.mkdir(parents=True, exist_ok=True)
        
        visualize_trajectory(traj_for_viz, self.turn_problem.contact_scenes_for_viz, viz_fpath,
                            self.turn_problem.fingers, self.turn_problem.obj_dof + 1) 