from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_pregrasp import calculate_turn_cost
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA
import time

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

experiment_name = 'compare_to_baselines_0'

filename = f'test/{experiment_name}.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    tuples = pkl.load(file)
    initializations, \
    pregrasp_poses_no_vf, turn_poses_no_vf, turn_trajectories_no_vf, \
    pregrasp_poses_vf, turn_poses_vf, turn_trajectories_vf, \
    pregrasp_poses_baseline0, turn_poses_baseline0, turn_trajectories_baseline0, \
    pregrasp_poses_baseline1, turn_poses_baseline1, turn_trajectories_baseline1 = zip(*tuples)


if __name__ == "__main__":

    ############################################################
    # Get real costs for each method
    n_repeat = 3
    novf_costs = []
    vf_costs = []
    baseline0_costs = []
    baseline1_costs = []

    std_novf = []
    std_vf = []
    std_baseline0 = []
    std_baseline1 = []

    assert(len(initializations) % n_repeat == 0)
    for i in range(int(len(initializations)/n_repeat)):
        costs_novf = []
        costs_vf = []
        costs_baseline0 = []
        costs_baseline1 = []

        for j in range(n_repeat):
            current_index = i*n_repeat+j
            assert (initializations[current_index] == initializations[i*n_repeat]).all()
            novf, _ = calculate_turn_cost(pregrasp_poses_no_vf[current_index].numpy(), turn_poses_no_vf[current_index])
            costs_novf.append(novf)
            vf, _ = calculate_turn_cost(pregrasp_poses_vf[current_index].numpy(), turn_poses_vf[current_index])
            costs_vf.append(vf)
            baseline0, _ = calculate_turn_cost(pregrasp_poses_baseline0[i+j].numpy(), turn_poses_baseline0[current_index])
            costs_baseline0.append(baseline0)
            baseline1, _ = calculate_turn_cost(pregrasp_poses_baseline1[i+j].numpy(), turn_poses_baseline1[current_index])
            costs_baseline1.append(baseline1)
        
        novf_costs.append(np.mean(costs_novf))
        std_novf.append(np.std(costs_novf))
        vf_costs.append(np.mean(costs_vf))
        std_vf.append(np.std(costs_vf))
        baseline0_costs.append(np.mean(costs_baseline0))
        std_baseline0.append(np.std(costs_baseline0))
        baseline1_costs.append(np.mean(costs_baseline1))
        std_baseline1.append(np.std(costs_baseline1))
        
    ############################################################

    # Get predicted costs for each method
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name='ensemble')

    predicted_costs_vf = []
    predicted_stds_vf = []
   
    for i in range(len(pregrasp_poses_vf)//n_repeat):
        pred_mean = []
        pred_std = []
        for j in range(n_repeat):
            vf = pregrasp_poses_vf[i*n_repeat+j]
            vf = convert_full_to_partial_config(vf.reshape(1,20))
            vf_norm = (vf - poses_mean) / poses_std
            vf_norm = vf_norm.float()
            ensemble_vf = query_ensemble(vf_norm, models)
            prediction_vf = ensemble_vf.mean(dim=0)
            std_vf_pred = ensemble_vf.std(dim=0)
            prediction_vf = prediction_vf * cost_std + cost_mean
            std_vf_pred = std_vf_pred * cost_std
            pred_mean.append(prediction_vf.detach().numpy())
            pred_std.append(std_vf_pred.detach().numpy())
        predicted_costs_vf.append(np.mean(np.array(pred_mean)))
        predicted_stds_vf.append(np.mean(np.array(pred_std)))
       
    ############################################################

    plt.figure(figsize=(20, 5))

    # Define offsets for spacing 
    predicted_offset = - 0.03
    method_offset = 0.03

    # Limit the values for plotting
    n_plot = 20

    x_novf = np.arange(len(novf_costs))
    plt.errorbar(x_novf, novf_costs, yerr=std_novf, fmt='.', color='black', label='Cost, No Information', linestyle='None', capsize=3)

    x_baseline0 = np.arange(len(baseline0_costs)) + 2 * method_offset
    plt.errorbar(x_baseline0, baseline0_costs, yerr=std_baseline0, fmt='.', color='green', label='Cost, Direct to Nearest Neighbor', linestyle='None', capsize=3)

    x_baseline1 = np.arange(len(baseline1_costs)) + 3 * method_offset
    plt.errorbar(x_baseline1, baseline1_costs, yerr=std_baseline1, fmt='.', color='blue', label='Cost, Guide to Nearest Neighbor', linestyle='None', capsize=3)

    x_vf = np.arange(len(vf_costs)) + 1 * method_offset
    plt.errorbar(x_vf, vf_costs, yerr=std_vf, fmt='.', color='red', label='Cost, Value Function (our method)', linestyle='None', capsize=3)

    # x_predicted_vf = np.arange(len(predicted_costs_vf)) + predicted_offset
    # plt.errorbar(x_predicted_vf, predicted_costs_vf, yerr=predicted_stds_vf, fmt='.', color='orange', label='Predicted Cost, Our Method', linestyle='None', capsize=3)
    
    ts = 16
    plt.xlabel('Sample Index', fontsize=ts)
    plt.ylabel('Cost Value', fontsize=ts)
    plt.title('Predicted and Real Turning Costs for all 4 Methods', fontsize=ts)
    plt.xticks(fontsize=ts-2)
    plt.yticks(fontsize=ts-2)
    # plt.tight_layout()
    plt.legend(fontsize=ts) #plt.legend(loc='upper right', fontsize=ts)

    plt.show()

    # # plt.savefig(f'{fpath}/plots/plot.png', dpi=300, bbox_inches='tight')
    # # plt.close()
    # exit()

    vis = True
    methods_to_vis = ['vf']#['vf', 'baseline0', 'baseline1', 'no_vf']
    small_delay = 0.5
    big_delay = 2.0
    initialization_delay = 1.0
    if vis:
        params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
        for i in range(len(initializations)//n_repeat):
            if 'vf' in methods_to_vis:
                print("vf")
                env.reset(initializations[i*n_repeat].reshape(1,20))
                time.sleep(initialization_delay)
                for j in range(len(turn_trajectories_vf[0])):
                    env.reset(torch.from_numpy(turn_trajectories_vf[i*n_repeat][j]).reshape(1,20).float())
                    time.sleep(small_delay)
                time.sleep(big_delay)
            
            if 'baseline0' in methods_to_vis:
                print("baseline0")
                env.reset(initializations[i*n_repeat].reshape(1,20))
                time.sleep(initialization_delay)
                for j in range(len(turn_trajectories_baseline0[0])):
                    env.reset(torch.from_numpy(turn_trajectories_baseline0[i*n_repeat][j]).reshape(1,20).float())
                    time.sleep(small_delay)
                time.sleep(big_delay)

            if 'baseline1' in methods_to_vis:
                print("baseline1")
                env.reset(initializations[i*n_repeat].reshape(1,20))
                time.sleep(initialization_delay)
                for j in range(len(turn_trajectories_baseline1[0])):
                    env.reset(torch.from_numpy(turn_trajectories_baseline1[i*n_repeat][j]).reshape(1,20).float())
                    time.sleep(small_delay)
                time.sleep(big_delay)

            if 'no_vf' in methods_to_vis:
                print("no_vf")
                env.reset(initializations[i*n_repeat].reshape(1,20))
                time.sleep(initialization_delay)
                for j in range(len(turn_trajectories_no_vf[0])):
                    env.reset(torch.from_numpy(turn_trajectories_no_vf[i*n_repeat][j]).reshape(1,20).float())
                    time.sleep(small_delay)
                time.sleep(big_delay)