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

experiment_name = 'test_method_visualize_vf'
calc_novf = False

filename = f'test/{experiment_name}.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    tuples = pkl.load(file)

    if calc_novf:
        pregrasp_poses, \
        regrasp_poses_vf, regrasp_trajs_vf, turn_poses_vf, turn_trajs_vf,\
        regrasp_poses_novf, regrasp_trajs_novf, turn_poses_novf, turn_trajs_novf = zip(*tuples)
    else:
        pregrasp_poses, \
        regrasp_poses_vf, regrasp_trajs_vf, turn_poses_vf, turn_trajs_vf = zip(*tuples)

if __name__ == "__main__":

    ############################################################
    # Get real costs for each method
    n_repeat = 1
    novf_costs = []
    vf_costs = []
    std_vf = []
    std_novf = []

    assert(len(pregrasp_poses) % n_repeat == 0)
    for i in range(int(len(pregrasp_poses)/n_repeat)):
        costs_vf = []
        costs_novf = []
    
        for j in range(n_repeat):
            current_index = i*n_repeat+j
            assert (pregrasp_poses[current_index] == pregrasp_poses[i*n_repeat]).all()
            vf, _ = calculate_turn_cost(regrasp_poses_vf[current_index].numpy(), turn_poses_vf[current_index])
            costs_vf.append(vf)
            if calc_novf:
                novf, _ = calculate_turn_cost(regrasp_poses_novf[current_index].numpy(), turn_poses_novf[current_index])
                costs_novf.append(novf)
            
        vf_costs.append(np.mean(costs_vf))
        std_vf.append(np.std(costs_vf))

        if calc_novf:
            novf_costs.append(np.mean(costs_novf))
            std_novf.append(np.std(costs_novf))
        
    ############################################################

    # Get predicted costs for each method
    # models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name='ensemble')

    # predicted_costs_vf = []
    # predicted_stds_vf = []

    # for i in range(len(regrasp_poses_vf)//n_repeat):
    #     pred_mean = []
    #     pred_std = []

    #     planned_pred_mean = []
    #     planned_pred_std = []

    #     for j in range(n_repeat):
    #         vf = regrasp_poses_vf[i*n_repeat+j]
    #         vf = convert_full_to_partial_config(vf.reshape(1,20))
    #         vf_norm = (vf - poses_mean) / poses_std
    #         vf_norm = vf_norm.float()
    #         ensemble_vf = query_ensemble(vf_norm, models)
    #         prediction_vf = ensemble_vf.mean(dim=0)
    #         std_vf_pred = ensemble_vf.std(dim=0)
    #         prediction_vf = prediction_vf * cost_std + cost_mean
    #         std_vf_pred = std_vf_pred * cost_std
    #         pred_mean.append(prediction_vf.detach().numpy())
    #         pred_std.append(std_vf_pred.detach().numpy())

    #     predicted_costs_vf.append(np.mean(np.array(pred_mean)))
    #     predicted_stds_vf.append(np.mean(np.array(pred_std)))
       
    ############################################################

    plt.figure(figsize=(20, 5))

    # Define offsets for spacing 
    predicted_offset = 0
    method_offset = 0.03

    # x_planned_vf = np.arange(len(planned_predicted_costs_vf)) - method_offset
    # plt.errorbar(x_planned_vf, planned_predicted_costs_vf, yerr=planned_predicted_stds_vf, fmt='.', color='blue', label='Predicted Cost of Plan (Our method)', linestyle='None', capsize=3)

    # x_predicted_vf = np.arange(len(predicted_costs_vf)) + predicted_offset
    # plt.errorbar(x_predicted_vf, predicted_costs_vf, yerr=predicted_stds_vf, fmt='.', color='orange', label='Predicted Cost, Our Method', linestyle='None', capsize=3)
    
    x_vf = np.arange(len(vf_costs)) + 1 * method_offset
    plt.errorbar(x_vf, vf_costs, yerr=std_vf, fmt='.', color='red', label='Cost, Our Method', linestyle='None', capsize=3)

    if calc_novf:
        x_no_vf = np.arange(len(novf_costs)) + 2 * method_offset
        plt.errorbar(x_no_vf, novf_costs, yerr=std_novf, fmt='.', color='black', label='Cost, No Information', linestyle='None', capsize=3)

    ts = 16
    plt.xlabel('Sample Index', fontsize=ts)
    plt.ylabel('Cost Value', fontsize=ts)
    plt.title('Predicted and Real Turning Costs for Our Method', fontsize=ts)
    plt.xticks(fontsize=ts-2)
    plt.yticks(fontsize=ts-2)
    plt.legend(loc='upper right', fontsize=ts)

    # print average cost decrease between vf_costs and novf_costs
    if calc_novf:
        print(f'Average cost decrease between vf_costs and novf_costs: {np.mean(np.array(novf_costs) - np.array(vf_costs))}')
    plt.tight_layout()
    plt.show()


    # vis = True
    # small_delay = 0.1
    # big_delay = 1.0
    # initialization_delay = 1.0
    # if vis:
    #     params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    #     for i in range(len(pregrasp_poses)//n_repeat):

    #         env.reset(initializations[i*n_repeat].reshape(1,20))
    #         time.sleep(initialization_delay)

   
        