import pathlib
import numpy as np
import pickle as pkl
import os

from itertools import product
from _value_function.screwdriver_problem import (
    init_env, pregrasp, regrasp, do_turn,
    convert_partial_to_full_config, convert_full_to_partial_config, emailer
)
from _value_function.train_value_function import (
    Net, query_ensemble, load_ensemble
)
from _value_function.data_collect.process_final_poses_pregrasp import calculate_turn_cost
from _value_function.test.test_method import get_initialization, get_initializations

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import torch

def load_or_create_checkpoint(starting_values):
    """
    Loads the checkpoint if it exists, otherwise creates a new default checkpoint.
    """
    if checkpoint_path.exists():
        print(f"Loading existing checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pkl.load(f)
    else:
        print(f"No checkpoint found. Creating a new one at {checkpoint_path}")
        # Initial bounding ranges
        vf_bounds = starting_values['vf_bounds']
        other_bounds = starting_values['other_bounds']
        variance_ratio_bounds = starting_values['variance_ratio_bounds']
        grid_size = starting_values['grid_size']
        
        # Create initial search grids
        vf_weights_init = np.logspace(
            np.log10(vf_bounds[0]),
            np.log10(vf_bounds[1]),
            grid_size
        ).tolist()
        other_weights_init = np.logspace(
            np.log10(other_bounds[0]),
            np.log10(other_bounds[1]),
            grid_size
        ).tolist()
        variance_ratios_init = np.linspace(
            variance_ratio_bounds[0],
            variance_ratio_bounds[1],
            grid_size
        ).tolist()
        
        checkpoint = {
            'iteration': 0,
            'max_iterations': 100,
            
            # Current best hyperparameters and cost
            'best_vf_weight': None,
            'best_other_weight': None,
            'best_variance_ratio': None,
            'lowest_total_cost': float('inf'),
            
            # Initial bounding ranges
            'vf_bounds': vf_bounds,
            'other_bounds': other_bounds,
            'variance_ratio_bounds': variance_ratio_bounds,
            
            # Current search grids
            'vf_weights': vf_weights_init,
            'other_weights': other_weights_init,
            'variance_ratios': variance_ratios_init,

            # The log-space or linear ranges used for refinement
            'vf_range': np.log10(vf_bounds[1]) - np.log10(vf_bounds[0]),
            'other_range': np.log10(other_bounds[1]) - np.log10(other_bounds[0]),
            'variance_ratio_range': variance_ratio_bounds[1] - variance_ratio_bounds[0],
            
            # Store final results after each iterations
            'results': [],  # list of (vf_weight, other_weight, variance_ratio, cost) for the best of each iteration
            
            # For partial iteration checkpoints
            # We'll record tested combinations as a dict: 
            # { iteration_number: set_of_(vf_weight, other_weight, variance_ratio) }
            'tested_combinations': {}
        }
    return checkpoint

def save_checkpoint(checkpoint):
    """Saves the checkpoint to file."""
    with open(checkpoint_path, 'wb') as f:
        pkl.dump(checkpoint, f)

def test(checkpoint, n_samples, which_weights): 

    # Load the data or environment objects once upfront
    pregrasps = pkl.load(open(f'{fpath}/test/initializations/weight_sweep_pregrasps.pkl', 'rb'))
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")

    # We extract frequently used fields from checkpoint for convenience
    iteration = checkpoint['iteration']
    max_iterations = checkpoint['max_iterations']
    
    # Main loop for adaptive grid refinement
    while iteration < max_iterations:
        # For convenience, read from checkpoint on each iteration
        vf_weights = checkpoint['vf_weights']
        other_weights = checkpoint['other_weights']
        variance_ratios = checkpoint['variance_ratios']
        best_vf_weight = checkpoint['best_vf_weight']
        best_other_weight = checkpoint['best_other_weight']
        best_variance_ratio = checkpoint['best_variance_ratio']
        lowest_total_cost = checkpoint['lowest_total_cost']
        
        print("==========================================")
        print(f"Iteration {iteration+1} / {max_iterations}")
        print(f"Current search for vf_weights: {vf_weights}")
        print(f"Current search for other_weights: {other_weights}")
        print(f"Current search for variance_ratios: {variance_ratios}")
        print(f"Current best_vf_weight: {best_vf_weight}")
        print(f"Current best_other_weight: {best_other_weight}")
        print(f"Current best_variance_ratio: {best_variance_ratio}")
        print(f"Current lowest_total_cost: {lowest_total_cost}")
        print("==========================================")

        # Make sure we have a set to track which combos we've already tested
        if iteration not in checkpoint['tested_combinations']:
            checkpoint['tested_combinations'][iteration] = set()

        # We'll do a fresh pass at finding the best combination in this iteration
        iteration_best_cost = float('inf')
        iteration_best_combo = (None, None, None)

        hyperparameters = [vf_weights, other_weights, variance_ratios]

        # Search over the entire grid but skip combos we've tested already
        for vf_weight, other_weight, variance_ratio in product(*hyperparameters):
            # combo_tuple = (vf_weight, other_weight, variance_ratio)
            combo_tuple = (vf_weight, 10.0, variance_ratio)
           

            if combo_tuple in checkpoint['tested_combinations'][iteration]:
                # Already finished testing this combo in a previous run, skip it
                continue

            # Evaluate total cost for this combination
            total_cost = 0.0

            for i in range(n_samples):
                # Running your environment logic
                img_save_dir = None
                env.frame_fpath = img_save_dir
                env.frame_id = 0

                pregrasp_pose = pregrasps[i]
                env.reset(dof_pos=pregrasp_pose)
                
                if which_weights == "regrasp":
                    regrasp_pose, regrasp_traj = regrasp(
                        env, config, chain, state2ee_pos_partial, perception_noise=0,
                        image_path=img_save_dir, initialization=pregrasp_pose, mode='vf', iters=regrasp_iters,
                        vf_weight=vf_weight, other_weight=other_weight, variance_ratio=variance_ratio
                    )

                    _, turn_pose, succ, turn_traj = do_turn(
                        regrasp_pose, config, env,
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                        perception_noise=0, image_path=img_save_dir, iters=turn_iters,
                        mode='no_vf'
                    )

                elif which_weights == "turn":
                    regrasp_pose, regrasp_traj = regrasp(
                        env, config, chain, state2ee_pos_partial, perception_noise=0,
                        image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
                    )

                    # SET TO NO VF FOR NOW
                    _, turn_pose, succ, turn_traj = do_turn(
                        regrasp_pose, config, env,
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                        perception_noise=0, image_path=img_save_dir, iters=turn_iters,
                        mode='vf', vf_weight=vf_weight, other_weight=other_weight, variance_ratio=variance_ratio
                    )
                else:
                    raise ValueError("which_weights must be either 'regrasp' or 'turn'")

                turn_cost, _ = calculate_turn_cost(regrasp_pose.numpy(), turn_pose)
                total_cost += turn_cost
                print(f"Sample {i} -> turn cost: {turn_cost}")
            
            # Mark this combo as tested
            checkpoint['tested_combinations'][iteration].add(combo_tuple)
            checkpoint['results'].append((vf_weight, other_weight, variance_ratio, total_cost))
            save_checkpoint(checkpoint)  # save so we don't re-test if we crash now
                
            print(f"Total combinations tested so far: {sum(len(v) for v in checkpoint['tested_combinations'].values())}")
            print(f"[Iteration {iteration+1}] vf_weight: {vf_weight}, "
                  f"other_weight: {other_weight}, "
                  f"variance_ratio: {variance_ratio}, total_cost: {total_cost}")

            # Check if this is better than the best we have so far in *this iteration*
            if total_cost < iteration_best_cost:
                iteration_best_cost = total_cost
                iteration_best_combo = (vf_weight, other_weight, variance_ratio)

            # Also check if this is the best overall across all iterations
            if total_cost < lowest_total_cost:
                checkpoint['lowest_total_cost'] = total_cost
                checkpoint['best_vf_weight'] = vf_weight
                checkpoint['best_other_weight'] = other_weight
                checkpoint['best_variance_ratio'] = variance_ratio
                lowest_total_cost = total_cost
                best_vf_weight = vf_weight
                best_other_weight = other_weight
                best_variance_ratio = variance_ratio

            # Immediately save checkpoint after each combo in case of crash
            save_checkpoint(checkpoint)

        # -- Finished testing all combos for the current iteration. --
        # iteration_best_combo holds the best for *this iteration*, iteration_best_cost is its cost.
        vf_best_iter, other_best_iter, var_best_iter = iteration_best_combo
        print(f"\nBest in iteration {iteration+1}: ")
        print(f"  vf={vf_best_iter}, other={other_best_iter}, var_ratio={var_best_iter}, cost={iteration_best_cost}\n")

        # Store best iteration-level result in results list
        checkpoint['results'].append((
            vf_best_iter,
            other_best_iter,
            var_best_iter,
            iteration_best_cost
        ))

        # Refine the grid around the best *overall* combination so far
        #   (you can also refine around iteration_best_combo if you prefer).
        #   Here, we refine around checkpoint['best_vf_weight'] etc.
        checkpoint['vf_range'] = checkpoint['vf_range'] / 2
        checkpoint['other_range'] = checkpoint['other_range'] / 2
        checkpoint['variance_ratio_range'] = checkpoint['variance_ratio_range'] / 2

        half_vf = checkpoint['vf_range'] / 2
        half_other = checkpoint['other_range'] / 2
        half_var = checkpoint['variance_ratio_range'] / 2

        # Construct new grids. For logs, we do log10 around the best_vf_weight/other_weight
        checkpoint['vf_weights'] = np.logspace(
            np.log10(best_vf_weight) - half_vf, 
            np.log10(best_vf_weight) + half_vf,
            len(vf_weights)
        ).tolist()

        checkpoint['other_weights'] = np.logspace(
            np.log10(best_other_weight) - half_other,
            np.log10(best_other_weight) + half_other,
            len(other_weights)
        ).tolist()

        # Linear refinement for variance ratio
        checkpoint['variance_ratios'] = np.linspace(
            best_variance_ratio - half_var,
            best_variance_ratio + half_var,
            len(variance_ratios)
        ).tolist()

        # Bump the iteration count
        iteration += 1
        checkpoint['iteration'] = iteration

        # Clear tested combinations for the new iteration
        checkpoint['tested_combinations'][iteration] = set()
        
        # Save the refined checkpoint
        save_checkpoint(checkpoint)
        emailer().send()
        
    # -- End while loop --

    print("Finished all iterations of adaptive grid search.\n")
    print("Final checkpoint results in 'results' list:")
    for row in checkpoint['results']:
        print(row)
    print(f"\nBest overall: ")
    print(f"  best_vf_weight={checkpoint['best_vf_weight']}, "
          f"best_other_weight={checkpoint['best_other_weight']}, "
          f"best_variance_ratio={checkpoint['best_variance_ratio']}")
    print(f"  lowest_total_cost={checkpoint['lowest_total_cost']}")

if __name__ == "__main__":

    max_screwdriver_tilt = 0.015
    screwdriver_noise_mag = 0.015
    finger_noise_mag = 0.25

    regrasp_iters = 100
    turn_iters = 100
    visualize = False   

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=visualize)
    sim_device = config['sim_device']
    
    n_samples = 3
    which_weights = "regrasp"
    name = "rgo"

    checkpoint_path = fpath /'test'/'weight_sweep'/f'checkpoint_{which_weights}_{name}.pkl'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    pregrasp_path = fpath /'test'/'initializations'/'weight_sweep_pregrasps.pkl'

    if pregrasp_path.exists() == False or len(pkl.load(open(pregrasp_path, 'rb'))) != n_samples:
        print("Generating new pregrasp initializations...")
        get_initializations(env, config, chain, sim_device, n_samples,
                            max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, save=True,
                            do_pregrasp=True, name='weight_sweep_pregrasps')

    starting_values = {
        'vf_bounds': [5, 50],
        'other_bounds': [10, 10],
        'variance_ratio_bounds': [.5, 2.0],
        'grid_size': 3
    }

    initial_checkpoint = load_or_create_checkpoint(starting_values)
    
    test(initial_checkpoint, n_samples, which_weights)