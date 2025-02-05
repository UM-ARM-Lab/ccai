from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import torch

experiment_name = 'test_method_fixed4'
calc_novf = True

filename = f'test/{experiment_name}.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    results = pkl.load(file)

models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name = "ensemble")

if __name__ == "__main__":

    # Get unique method names from the results dictionary
    method_names = list(results.keys())  # Start with all keys in the results dict

    # Automatically infer n_trials and n_repeat from the first method's data
    first_method = method_names[0]
    all_pairs = results[first_method].keys()  # e.g., {(0,0), (0,1), ...}
    n_trials = max(k[0] for k in all_pairs) + 1
    n_repeat = max(k[1] for k in all_pairs) + 1

    print(f"Inferred n_trials = {n_trials}, n_repeat = {n_repeat}")
    print(f"Method names found: {method_names}")

    # Dictionary to hold data for plotting
    data = {}
    for method_name in method_names:
        data[method_name] = {
            'costs': [],
            'stds': [],
            'pred_costs': [],
            'pred_stds': []
        }
        for pregrasp_index in range(n_trials):
            all_costs = []
            all_pred_costs = []
            all_pred_stds = []

            for repeat_index in range(n_repeat):
                # Retrieve the stored poses for this method and (pregrasp_index, repeat_index)
                pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj = \
                    results[method_name][(pregrasp_index, repeat_index)]

                # Compute "cost"
                cost = calculate_turn_cost(regrasp_pose.numpy(), turn_pose)
                all_costs.append(cost)

                # Query the value-function ensemble
                input_norm = ((convert_full_to_partial_config(regrasp_pose) - poses_mean) / poses_std).flatten().float()
                input_norm = torch.cat([input_norm, torch.tensor([12.0], dtype=torch.float32)], dim=0)

                vf_output_norm = query_ensemble(input_norm, models)
                vf_output = vf_output_norm * cost_std + cost_mean

                pred_mean = torch.mean(vf_output)
                pred_std = torch.std(vf_output)

                all_pred_costs.append(pred_mean.item())
                all_pred_stds.append(pred_std.item())

            # Average cost and standard deviation across repeats
            data[method_name]['costs'].append(np.mean(all_costs))
            data[method_name]['stds'].append(np.std(all_costs))

            data[method_name]['pred_costs'].append(np.mean(all_pred_costs))
            data[method_name]['pred_stds'].append(np.mean(all_pred_stds))

    # ---- Plotting ----
    plt.figure(figsize=(10, 5))

    # These offsets help shift the actual cost vs the predicted cost horizontally
    # so that they don't overlap exactly on top of each other.
    method_offset = 0.15  # spacing offset for each method group
    pred_offset = 0.05    # spacing offset for predicted bars within the same method group

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    for method_i, method_name in enumerate(method_names):
        # x positions for the actual cost
        x_actual = np.arange(len(data[method_name]['costs'])) + method_i * method_offset
        # x positions for predicted cost (shift from x_actual by pred_offset)
        x_pred = x_actual + pred_offset

        # Plot actual costs with error bars
        plt.errorbar(
            x_actual, 
            data[method_name]['costs'], 
            yerr=data[method_name]['stds'], 
            fmt='o',  # marker shape
            label=f'Cost ({method_name.upper()})', 
            linestyle='None', 
            capsize=3,
            color=colors[method_i]
        )

        # Plot predicted costs with error bars
        plt.errorbar(
            x_pred,
            data[method_name]['pred_costs'],
            yerr=data[method_name]['pred_stds'],
            fmt='x',   # different marker shape
            label=f'Predicted ({method_name.upper()})',
            linestyle='None',
            capsize=3,
            color=colors[method_i]
        )

    ts = 16
    plt.xlabel('Pregrasp Index', fontsize=ts)
    plt.ylabel('Cost Value', fontsize=ts)
    plt.title('Turning Costs by Method', fontsize=ts)
    plt.xticks(fontsize=ts-2)
    plt.yticks(fontsize=ts-2)
    plt.legend(loc='upper right', fontsize=ts-2)

    # If you have both "no_vf" and "vf", compute the average cost difference
    if calc_novf and "no_vf" in data and "vf" in data:
        mean_diff = np.mean(data["no_vf"]["costs"]) - np.mean(data["vf"]["costs"])
        print(f'Average cost difference (no_vf - vf): {mean_diff}')

    plt.tight_layout()
    plt.show()