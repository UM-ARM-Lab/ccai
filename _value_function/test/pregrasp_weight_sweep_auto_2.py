from _value_function.screwdriver_problem import init_env, pregrasp, do_turn, convert_partial_to_full_config, convert_full_to_partial_config
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
from _value_function.data_collect.process_final_poses_pregrasp import calculate_turn_cost
import pathlib
import numpy as np
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import time
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def get_initialization(sim_device, env):
    m = 0.3
    index_noise_mag = torch.tensor([m]*4)
    index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
    middle_thumb_noise_mag = torch.tensor([m]*4)
    middle_thumb_noise = middle_thumb_noise_mag * (2 * torch.rand(4) - 1)
    screwdriver_noise = torch.tensor([
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(0, 2 * np.pi),
        0.0  
    ])
    initialization = torch.cat((torch.tensor([[0.1000,  0.6000,  0.6000,  0.6000]]).float().to(device=sim_device) + index_noise,
                                torch.tensor([[-0.1000,  0.5000,  0.9000,  0.9000]]).float().to(device=sim_device) + middle_thumb_noise,
                                torch.tensor([[0.0000,  0.5000,  0.6500,  0.6500]]).float().to(device=sim_device),
                                torch.tensor([[1.2000,  0.3000,  0.3000,  1.2000]]).float().to(device=sim_device) + middle_thumb_noise,
                                torch.tensor([[0.0000,  0.0000,  0.0000,  0.0000]]).float().to(device=sim_device) + screwdriver_noise),
                                dim=1).to(sim_device)
    
    env.reset(dof_pos= initialization)
    solved_initialization = env.get_state()['q'].reshape(1,16)[:,0:-1].to(device=sim_device)
    return convert_partial_to_full_config(solved_initialization)

def get_initializations():
    initializations = []
    while len(initializations) < n_samples:
        initialization = get_initialization(sim_device, env)
        sd = initialization[0, -4:-1]
        if abs(sd[0]) < 0.05 and abs(sd[1]) < 0.05:
            initializations.append(initialization)
    pkl.dump(initializations, open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'wb'))

def test(prediction_only = False):

    initializations = pkl.load(open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'rb'))
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")
    
    # Starting values
    vf_weight = 1.0
    other_weight = 1.0
    lowest_total_cost = 1e10
    results = []
    pregrasp_iters = 200

    # Initial coarse grid (large steps)
    vf_weights = [3.16, 10.0]
    vf_weights.append(vf_weights[1]*vf_weights[1]/vf_weights[0])
    other_weights = [0.003, 0.01]
    other_weights.append(other_weights[1]*other_weights[1]/other_weights[0])

    variance_ratios = [2.75,5.0,7.25]

    best_vf_weight = None
    best_other_weight = None
    best_variance_ratio = None

    grid_size = len(vf_weights)
    vf_range = np.log10(vf_weights[-1]) - np.log10(vf_weights[0])
    other_range = np.log10(other_weights[-1]) - np.log10(other_weights[0])
    variance_ratio_range = variance_ratios[-1] - variance_ratios[0]
    
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
        lowest_total_cost = 1000
        
        # Perform grid search with current vf_weights and other_weights
        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{0}')
        pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        env.frame_fpath = img_save_dir
        env.frame_id = 0
        
        for vf_weight in vf_weights:
            for other_weight in other_weights:
                for variance_ratio in variance_ratios:
                    total_cost = 0
                    
                    for i in range(n_samples):
                        pose_vf,plan = pregrasp(env, config, chain, deterministic=True, 
                                initialization = initializations[i], mode='vf', 
                                vf_weight = vf_weight, other_weight = other_weight, variance_ratio = variance_ratio,
                                vis_plan=False, iters = pregrasp_iters)
                        
                        if prediction_only:
                            vf_pose = convert_full_to_partial_config(pose_vf.reshape(1,20))
                            vf_pose_norm = (vf_pose - poses_mean) / poses_std
                            vf_pose_norm = vf_pose_norm.float()
                            vf = query_ensemble(vf_pose_norm, models)
                            prediction_vf_norm = vf.mean(dim=0)
                            prediction_vf = prediction_vf_norm * cost_std + cost_mean
                            total_cost += prediction_vf.item()
                        else:

                            vf_pose = convert_full_to_partial_config(pose_vf.reshape(1,20))
                            vf_pose_norm = (vf_pose - poses_mean) / poses_std
                            vf_pose_norm = vf_pose_norm.float()
                            vf = query_ensemble(vf_pose_norm, models)
                            prediction_vf_norm = vf.mean(dim=0)
                            prediction_vf = prediction_vf_norm * cost_std + cost_mean
                            prediction = prediction_vf.item()


                            _, turn_pose_vf, turn_succ, turn_trajectory_vf = do_turn(pose_vf, config, env, 
                                    sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                                    perception_noise=0, image_path = img_save_dir)
                            real_cost, _ = calculate_turn_cost(pose_vf.numpy(), turn_pose_vf)
                            total_cost += real_cost
                            print("predicted cost: ", prediction)
                            print("real cost: ", real_cost)
                
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

        vf_weight_range = np.logspace(np.log10(best_vf_weight) - vf_range/2, np.log10(best_vf_weight) + vf_range/2, grid_size)
        other_weight_range = np.logspace(np.log10(best_other_weight) - other_range/2, np.log10(best_other_weight) + other_range/2, grid_size)
        variance_ratio_range = np.linspace(best_variance_ratio - variance_ratio_range/2, best_variance_ratio + variance_ratio_range/2, grid_size)

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
    get_initializations()
    test()
