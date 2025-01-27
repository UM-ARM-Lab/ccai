import pathlib
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_pregrasp import calculate_turn_cost
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
import torch

# Paths
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = CCAI_PATH / 'data'

experiment_name = 'test_method_sanity4'
calc_novf = True

for i in range(100):

    # Load the pickle file
    filename = fpath / 'regrasp_to_turn_datasets' / f'regrasp_to_turn_dataset_abhi_{i}.pkl'
    # Sometimes using str(filename) can help if you encounter file-like errors
    with open(str(filename), 'rb') as file:
        pose_tuples = pkl.load(file)
        
    pregrasp_poses, regrasp_poses, regrasp_trajs, turn_poses, turn_trajs = zip(*pose_tuples)

    # Load ensemble models and normalization stats
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")

    all_pred_costs = []
    all_pred_stds = []
    all_costs = []

    # IMPORTANT FIX: use zip to iterate over pairs
    for rg, tp in zip(regrasp_poses, turn_poses):
        # 1. Compute the actual (real) cost
        cost, _ = calculate_turn_cost(rg.numpy(), tp)
        all_costs.append(cost)

        # 2. Prepare the input for the value function
        partial_cfg = convert_full_to_partial_config(rg)
        input_norm = (partial_cfg - poses_mean) / poses_std
        input_norm = input_norm.flatten().float()
        
        # Often we add a zero at the end to indicate "no push" or "no turn" in some configurations
        # (depends on your problem statement).
        input_norm = torch.cat([input_norm, torch.tensor([0.0], dtype=torch.float32)], dim=0)

        # 3. Query the ensemble models
        vf_output_norm = query_ensemble(input_norm, models)
        vf_output = vf_output_norm * cost_std + cost_mean  # de-normalize

        pred_mean = torch.mean(vf_output)
        pred_std = torch.std(vf_output)

        all_pred_costs.append(pred_mean.item())
        all_pred_stds.append(pred_std.item())

    # ---- Plot Results ----
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(all_costs))

    # Plot predicted costs with error bars
    plt.errorbar(
        indices, 
        all_pred_costs, 
        yerr=all_pred_stds, 
        fmt='o',
        capsize=4, 
        label='Predicted costs'
    )

    # Plot real costs
    plt.plot(indices, all_costs, 'x', label='Real costs')

    plt.xlabel('Sample Index')
    plt.ylabel('Cost')
    plt.title('Real Costs vs Predicted Costs (with Std. Dev)')
    plt.legend()
    plt.show()