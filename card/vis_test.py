from card.card_problem import init_env
from card.process_card_data import calculate_cost
from card.train_vf_card_index import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
import pytorch_kinematics.transforms as tf
from scipy.spatial.transform import Rotation as R
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
save_dir = pathlib.Path(f'{CCAI_PATH}/data/card/figures')
import torch

def calculate_d2g(initial_pose, final_pose):
    return calculate_cost(initial_pose, final_pose)

if __name__ == "__main__":

    annotate = True

    experiment_names = ['sanity3'] 
    budget = "Low Budget"

    results = {}

    for name in experiment_names:
        file_path = fpath / 'card'/ 'test' / 'test_method' / f'result_{name}.pkl'
        with open(file_path, 'rb') as file:
            single_results = pkl.load(file)

        for method, method_results in single_results.items():

            if method == "no_vf":
                method = "T.O."
            elif method == "vf":
                method = "AVO (ours)"
    
            else:
                print(f"{method} is not a valid method")
                exit()
            if method in results:
                print(f"Overriding {method}.")

            results[method] = method_results


    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

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
            'd2gs': [],
            'stds': [],
        }
        for pregrasp_index in range(n_trials):
            all_d2gs = []

            for repeat_index in range(n_repeat):

                initial_state, final_state, *extra = \
                    results[method_name][(pregrasp_index, repeat_index)]

                d2g = calculate_d2g(initial_state, final_state)     
                # if method_name == "T.O.":
                #     print(d2g)
                all_d2gs.append(d2g)

            # Average d2g and standard deviation across repeats
            data[method_name]['d2gs'].append(np.mean(all_d2gs))
            data[method_name]['stds'].append(np.std(all_d2gs))
    
    if "no_vf" in data and "vf" in data:
            mean_no_vf = np.mean(data["no_vf"]["d2gs"])
            mean_vf = np.mean(data["vf"]["d2gs"])

            mean_diff = mean_no_vf - mean_vf
            percent_decrease = (mean_diff / mean_no_vf) * 100

            print(f'Average d2g difference (no_vf - vf): {mean_diff:.2f}')
            print(f'Percent decrease in d2g: {percent_decrease:.2f}%')

    for method in data:
        print(f'{method} mean d2g: {np.mean(data[method]["d2gs"]):.2f} +/- {np.std(data[method]["d2gs"]):.2f}')

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    ts = 22
    if True:
        plt.figure(figsize=(10, 5))
        method_offset = 0.15  # spacing offset for each method group

        for method_i, method_name in enumerate(method_names):
            # x positions for the actual d2g
            x = np.arange(len(data[method_name]['d2gs'])) + method_i * method_offset

            # Plot d2gs with error bars
            plt.errorbar(
                x, 
                data[method_name]['d2gs'], 
                yerr=data[method_name]['stds'], 
                fmt='o',  # marker shape
                label=f'Rollout d2g (method name: {method_name})', 
                linestyle='None', 
                capsize=3,
                color=colors[method_i]
            )
        plt.xlabel('Initial State Index', fontsize=ts)
        plt.ylabel('d2g Value', fontsize=ts)
        plt.title('Turning d2gs by Method', fontsize=ts)
        plt.xticks(fontsize=ts-2)
        plt.yticks(fontsize=ts-2)
        plt.legend(loc='upper right', fontsize=ts-2)

        plt.tight_layout()
        plt.show()
    if True:
        # ----  Boxplot ----
        # Create a new figure for the boxplot
        plt.figure(figsize=(10, 5))

        # Prepare data for boxplot: a list of d2g lists (one per method)
        # boxplot_data = [data[method]['d2gs'] for method in method_names]

        boxplot_data = []
        for method_name, metrics in data.items():
            d2gs = [float(d2g) for d2g in metrics['d2gs']]
            boxplot_data.append(d2gs)

        # Create the boxplot; patch_artist=True allows us to color the boxes.
        bp = plt.boxplot(
            boxplot_data, 
            labels=[method for method in method_names], 
            patch_artist=True, 
            showmeans=False, 
            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='black', markersize=8),
            boxprops=dict(color='black', linewidth=2),
            medianprops=dict(color='black', linewidth=2)
        )

        # Color the boxes using the same colors as before
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Annotate each box with its median value
        if annotate:
            for i, method in enumerate(method_names):
                median_val = np.median(data[method]['d2gs'])
                plt.text(
                    i + 1, 
                    median_val, #- 0.02, 
                    f"{median_val:.2f}", 
                    horizontalalignment='center', 
                    verticalalignment='bottom',
                    fontsize=12,
                    color='black'
                )
        # plt.ylim(0, 5.1)
        plt.xlabel('Method', fontsize=ts)
        plt.ylabel('Angle Difference', fontsize=ts)
        plt.title(f'Simulation, {budget}: Angle Differences From Goal', fontsize=ts)
        plt.xticks(fontsize=ts-2)
        plt.xticks(rotation=20, ha='right')
        plt.yticks(fontsize=ts-2)
        plt.tight_layout()
        plt.savefig(save_dir / f"{budget} sim.png")
        plt.show()