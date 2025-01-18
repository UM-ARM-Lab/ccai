from _value_function.screwdriver_problem import init_env, pregrasp, regrasp, do_turn, \
    convert_partial_to_full_config, convert_full_to_partial_config
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
from _value_function.data_collect.process_final_poses_pregrasp import calculate_turn_cost
import pathlib
import numpy as np
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import time
from itertools import product

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
from _value_function.test.test_method import get_initialization, get_initializations


def test(prediction_only=False):
    initializations = pkl.load(open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'rb'))
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")

    # Initialize best values
    best_vf_weight, best_other_weight, best_variance_ratio = None, None, None

    # Starting values
    lowest_total_cost = float('inf') 
    results = []
    pregrasp_iters = 1#80
    regrasp_iters = 1#100
    turn_iters= 1#100


    # Define grid bounds
    vf_bounds = [1, 100000] 
    other_weight_bounds = [0.001, 100] 
    variance_ratio_bounds = [1,100]

    # Initialize grids
    grid_size = 3
    vf_weights = np.logspace(np.log10(vf_bounds[0]), np.log10(vf_bounds[1]), grid_size).tolist()
    other_weights = np.logspace(np.log10(other_weight_bounds[0]), np.log10(other_weight_bounds[1]), grid_size).tolist()
    variance_ratios = np.linspace(variance_ratio_bounds[0], variance_ratio_bounds[1], grid_size).tolist()

    # initialize grid range so that it can be shrunk later
    vf_range = np.log10(vf_bounds[1]) - np.log10(vf_bounds[0])
    other_range = np.log10(other_weight_bounds[1]) - np.log10(other_weight_bounds[0])
    variance_ratio_range = variance_ratio_bounds[1] - variance_ratio_bounds[0]
    
    hyperparameters = [
        vf_weights,
        other_weights,
        variance_ratios
    ]

    # Maximum iterations for adaptive grid refinement
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        print(f"Iteration {iteration+1}:")
        print(f"vf weight range: {vf_weights}")
        print(f"other weight range: {other_weights}")
        print(f"variance ratio range: {variance_ratios}")
        print(f"current best vf weight: {best_vf_weight}")
        print(f"current best other weight: {best_other_weight}")
        print(f"current best variance ratio: {best_variance_ratio}")
        lowest_total_cost = float('inf')

        # Perform grid search with current vf_weights and other_weights
        # img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{0}')
        # pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)
        img_save_dir = None
        env.frame_fpath = img_save_dir
        env.frame_id = 0

        for vf_weight, other_weight, variance_ratio in product(*hyperparameters):
            total_cost = 0

            for i in range(n_samples):
                
                pregrasp_pose, planned_pose = pregrasp(env, config, chain, deterministic=True, perception_noise=0, 
                        image_path = img_save_dir, initialization = initializations[i], mode='no_vf', iters = pregrasp_iters)

                regrasp_pose, regrasp_traj = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=0, 
                        image_path = img_save_dir, initialization = pregrasp_pose, mode='vf', iters = regrasp_iters,
                        vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio)
                
                _, turn_pose, succ, turn_traj = do_turn(regrasp_pose, config, env, 
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                        perception_noise=0, image_path = img_save_dir, iters = turn_iters,
                        mode='vf', vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio)
                

                turn_cost, _ = calculate_turn_cost(regrasp_pose, turn_pose)
                total_cost += turn_cost
                print("turn cost: ", turn_cost)

            print(f"vf_weight: {vf_weight}, other_weight: {other_weight}, variance_ratio: {variance_ratio}, total_cost: {total_cost}")

            # Update best weights if this combination of weights gives the lowest cost
            if total_cost < lowest_total_cost:
                lowest_total_cost = total_cost
                best_vf_weight = vf_weight
                best_other_weight = other_weight
                best_variance_ratio = variance_ratio

        # Store the best results so far
        results.append((best_vf_weight, best_other_weight, best_variance_ratio, lowest_total_cost))

        # Refine the grid around the best weights
        vf_range = vf_range / 2
        other_range = other_range / 2
        variance_ratio_range = variance_ratio_range / 2

        vf_weight_range = np.logspace(np.log10(best_vf_weight) - vf_range / 2, np.log10(best_vf_weight) + vf_range / 2, grid_size)
        other_weight_range = np.logspace(np.log10(best_other_weight) - other_range / 2, np.log10(best_other_weight) + other_range / 2, grid_size)
        variance_ratio_range = np.linspace(best_variance_ratio - variance_ratio_range / 2, best_variance_ratio + variance_ratio_range / 2, grid_size)

        vf_weights = vf_weight_range.tolist()
        other_weights = other_weight_range.tolist()
        variance_ratios = variance_ratio_range.tolist()

        iteration += 1

    pkl.dump(results, open(f'{fpath}/vf_weight_sweep/results.pkl', 'wb'))
    print("results:")
    print(results)
    print(f'best_vf_weight: {best_vf_weight}, best_other_weight: {best_other_weight}, lowest_total_cost: {lowest_total_cost}')


if __name__ == "__main__":
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)
    sim_device = config['sim_device']
    n_samples = 3
    max_screwdriver_tilt=0.015
    screwdriver_noise_mag=0.015
    finger_noise_mag=0.25
    get_initializations(sim_device, env, n_samples, max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, save = True)
    test()