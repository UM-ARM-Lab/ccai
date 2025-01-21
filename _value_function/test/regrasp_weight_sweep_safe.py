import pathlib
import numpy as np
import pickle as pkl
import os

from itertools import product
from _value_function.screwdriver_problem import (
    init_env, pregrasp, regrasp, do_turn,
    convert_partial_to_full_config, convert_full_to_partial_config
)
from _value_function.train_value_function import (
    Net, query_ensemble, load_ensemble
)
from _value_function.data_collect.process_final_poses_pregrasp import calculate_turn_cost
from _value_function.test.test_method import get_initialization, get_initializations

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import torch

checkpoint_path = fpath /'test'/'weight_sweep'/'checkpoint.pkl'
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


def load_or_create_checkpoint():
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
        vf_bounds = [1, 1000]
        other_bounds = [0.001, 100]
        variance_ratio_bounds = [1, 100]
        grid_size = 3
        
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


def test():

    # Load the data or environment objects once upfront
    initializations = pkl.load(open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'rb'))
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")
    
    # Load or create the checkpoint
    checkpoint = load_or_create_checkpoint()

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
        
        vf_range = checkpoint['vf_range']
        other_range = checkpoint['other_range']
        variance_ratio_range = checkpoint['variance_ratio_range']
        
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

        # n_samples, etc. can be adjusted or passed as parameters
        n_samples = 3
        pregrasp_iters = 80
        regrasp_iters = 100
        turn_iters = 100

        # We'll do a fresh pass at finding the best combination in this iteration
        iteration_best_cost = float('inf')
        iteration_best_combo = (None, None, None)

        hyperparameters = [vf_weights, other_weights, variance_ratios]

        # Search over the entire grid but skip combos we've tested already
        for vf_weight, other_weight, variance_ratio in product(*hyperparameters):
            combo_tuple = (vf_weight, other_weight, variance_ratio)

            if combo_tuple in checkpoint['tested_combinations'][iteration]:
                # Already tested this combo in a previous run, skip it
                continue

            # Mark this combo as tested
            checkpoint['tested_combinations'][iteration].add(combo_tuple)
            save_checkpoint(checkpoint)  # save so we don't re-test if we crash now

            # Evaluate total cost for this combination
            total_cost = 0.0

            for i in range(n_samples):
                # Running your environment logic
                img_save_dir = None
                env.frame_fpath = img_save_dir
                env.frame_id = 0

                pregrasp_pose, planned_pose = pregrasp(
                    env, config, chain, deterministic=True, perception_noise=0,
                    image_path=img_save_dir, initialization=initializations[i], mode='no_vf',
                    iters=pregrasp_iters
                )

                regrasp_pose, regrasp_traj = regrasp(
                    env, config, chain, state2ee_pos_partial, perception_noise=0,
                    image_path=img_save_dir, initialization=pregrasp_pose, mode='vf', iters=regrasp_iters,
                    vf_weight=vf_weight, other_weight=other_weight, variance_ratio=variance_ratio
                )

                # SET TO NO VF FOR NOW
                _, turn_pose, succ, turn_traj = do_turn(
                    regrasp_pose, config, env,
                    sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                    perception_noise=0, image_path=img_save_dir, iters=turn_iters,
                    mode='no_vf', vf_weight=vf_weight, other_weight=other_weight, variance_ratio=variance_ratio
                )

                turn_cost, _ = calculate_turn_cost(regrasp_pose.numpy(), turn_pose)
                total_cost += turn_cost
                print(f"Sample {i} -> turn cost: {turn_cost}")
                
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
    n_samples = 5
    max_screwdriver_tilt = 0.015
    screwdriver_noise_mag = 0.015
    finger_noise_mag = 0.25

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)
    sim_device = config['sim_device']
    
    # get_initializations(env, sim_device, n_samples,
                        # max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, save=True)
    test()