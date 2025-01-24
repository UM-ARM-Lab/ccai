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

experiment_name = 'test_method_0'
calc_novf = True

filename = f'test/{experiment_name}.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    # 'results' is a dict: results["no_vf"][(pregrasp_idx, repeat_idx)] -> [poses...]
    # and results["vf"][(pregrasp_idx, repeat_idx)] -> [poses...], etc.
    results = pkl.load(file)

if __name__ == "__main__":

    n_repeat = 2
    n_trials = 2

    # If your saved file has keys like "no_vf", "vf",
    # define them here. If you have multiple VF keys
    # (e.g. "vf_1_samples", "vf_2_samples", etc.),
    # include them in this list:
    method_names = ['vf', "no_vf"]
    if calc_novf and "no_vf" in results:
        method_names.append("no_vf")
    # If you just have a single "vf" method:
    if "vf" in results:
        method_names.append("vf")
    # Otherwise, if you had many like "vf_1_samples", "vf_2_samples", ...
    # you could gather them: 
    #   method_names.extend(k for k in results.keys() if k.startswith("vf_"))

    data = {}
    for method_name in method_names:
        data[method_name] = {
            'costs': [],
            'stds': []
        }
        for pregrasp_index in range(n_trials):
            all_costs = []
            for repeat_index in range(n_repeat):
                # Retrieve the stored poses for this method and (pregrasp_index, repeat_index)
                pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj = \
                    results[method_name][(pregrasp_index, repeat_index)]

                # Example function to compute "cost"
                cost, _ = calculate_turn_cost(regrasp_pose, turn_pose)
                all_costs.append(cost)

            # Average cost across the repeats for this pregrasp index
            data[method_name]['costs'].append(np.mean(all_costs))
            data[method_name]['stds'].append(np.std(all_costs))

    # ---- Plotting ----
    plt.figure(figsize=(10, 5))
    method_offset = 0.03  # spacing offset for each method

    for method_name in method_names:
        # x positions for the bars/points
        x = np.arange(len(data[method_name]['costs'])) + (method_names.index(method_name) * method_offset)
        plt.errorbar(
            x, 
            data[method_name]['costs'], 
            yerr=data[method_name]['stds'], 
            fmt='.', 
            label=f'Cost, {method_name.upper()}', 
            linestyle='None', 
            capsize=3
        )
    
    ts = 16
    plt.xlabel('Pregrasp Index', fontsize=ts)
    plt.ylabel('Cost Value', fontsize=ts)
    plt.title('Turning Costs by Method', fontsize=ts)
    plt.xticks(fontsize=ts-2)
    plt.yticks(fontsize=ts-2)
    plt.legend(loc='upper right', fontsize=ts)

    # If you have both "no_vf" and "vf", compute the average cost difference
    if calc_novf and "no_vf" in data and "vf" in data:
        mean_diff = np.mean(data["no_vf"]["costs"]) - np.mean(data["vf"]["costs"])
        print(f'Average cost difference (no_vf - vf): {mean_diff}')

    plt.tight_layout()
    plt.show()