from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
from _value_function.train_value_function_regrasp import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import torch

# make sure tests have the same number of trials and repeats
experiment_names = ['test_method_test_official_high_iter_all'] #['test_method_test_official_all']#
results = {}

for name in experiment_names:
    file_path = fpath / 'test' / f'{name}.pkl'
    with open(file_path, 'rb') as file:
        single_results = pkl.load(file)
    
    for method, method_results in single_results.items():
        if method not in results:
            results[method] = method_results
        else:
            print(f"Warning: method {method} already exists in combined_results. Exiting.")
            exit()

if __name__ == "__main__":
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
            'costs': [],
            'stds': [],
            'tilt_angles': [],
            'yaw_deltas': [],
        }
        for pregrasp_index in range(n_trials):
            all_costs = []

            for repeat_index in range(n_repeat):

                pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj, *initial_samples = \
                    results[method_name][(pregrasp_index, repeat_index)]

                cost = calculate_turn_cost(regrasp_pose.numpy(), turn_pose)
                            
                all_costs.append(cost)

                pose_i = regrasp_pose.numpy().flatten()[-4:-1]*180/np.pi
                pose_f = turn_pose.flatten()[-4:-1]*180/np.pi
                if pose_i[-1] < -180:
                    pose_i[-1] += 720
                if pose_f[-1] < -180:
                    pose_f[-1] += 720

                tilt_angle = max(abs(pose_f[:2]))

                yaw_delta = (pose_i[-1] - pose_f[-1]) 
                if yaw_delta > 100 or yaw_delta < -100:
                    print("bad")
                    exit()
                    # for j in range(13):
                    #     env.reset(torch.from_numpy(turn_traj[j]).reshape(1, 20).float())
                    #     # print(max(abs(turn_traj[j][-4:-2]))*180/np.pi)
                    #     time.sleep(0.1)
            
                data[method_name]['tilt_angles'].append(tilt_angle)
                data[method_name]['yaw_deltas'].append(yaw_delta)

            # Average cost and standard deviation across repeats
            data[method_name]['costs'].append(np.mean(all_costs))
            data[method_name]['stds'].append(np.std(all_costs))
    
    if "no_vf" in data and "vf" in data:
            mean_no_vf = np.mean(data["no_vf"]["costs"])
            mean_vf = np.mean(data["vf"]["costs"])

            mean_diff = mean_no_vf - mean_vf
            percent_decrease = (mean_diff / mean_no_vf) * 100

            print(f'Average cost difference (no_vf - vf): {mean_diff}')
            print(f'Percent decrease in cost: {percent_decrease:.2f}%')

    for method in data:
        print(f'{method} mean cost: {np.mean(data[method]["costs"])}')

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    ts = 16
    if False:
        plt.figure(figsize=(10, 5))
        method_offset = 0.15  # spacing offset for each method group

        for method_i, method_name in enumerate(method_names):
            # x positions for the actual cost
            x = np.arange(len(data[method_name]['costs'])) + method_i * method_offset

            # Plot costs with error bars
            plt.errorbar(
                x, 
                data[method_name]['costs'], 
                yerr=data[method_name]['stds'], 
                fmt='o',  # marker shape
                label=f'Rollout Cost (method name: {method_name.upper()})', 
                linestyle='None', 
                capsize=3,
                color=colors[method_i]
            )
        plt.xlabel('Pregrasp Index', fontsize=ts)
        plt.ylabel('Cost Value', fontsize=ts)
        plt.title('Turning Costs by Method', fontsize=ts)
        plt.xticks(fontsize=ts-2)
        plt.yticks(fontsize=ts-2)
        plt.legend(loc='upper right', fontsize=ts-2)

        plt.tight_layout()
        plt.show()
    if True:
        # ----  Boxplot ----
        # Create a new figure for the boxplot
        plt.figure(figsize=(10, 5))

        # Prepare data for boxplot: a list of cost lists (one per method)
        boxplot_data = [data[method]['costs'] for method in method_names]

        # Create the boxplot; patch_artist=True allows us to color the boxes.
        bp = plt.boxplot(
            boxplot_data, 
            labels=[method.upper() for method in method_names], 
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
        for i, method in enumerate(method_names):
            median_val = np.median(data[method]['costs'])
            plt.text(
                i + 1 - 0.15, 
                median_val - 0.02, 
                f"{median_val:.2f}", 
                horizontalalignment='center', 
                verticalalignment='bottom',
                fontsize=12,
                color='black'
            )
        plt.ylim(0, 5.1)
        plt.xlabel('Method', fontsize=ts)
        plt.ylabel('Cost Value', fontsize=ts)
        plt.title('Boxplot of Turning Costs by Method', fontsize=ts)
        plt.xticks(fontsize=ts-2)
        plt.yticks(fontsize=ts-2)
        plt.tight_layout()
        plt.show()
    if True:
        # ----  Boxplot ----
        # Create a new figure for the boxplot
        plt.figure(figsize=(10, 5))

        # Prepare data for boxplot: a list of cost lists (one per method)
        boxplot_data = []

        for method_name, metrics in data.items():
            tilt_threshold = 10
            tilt_angles = metrics['tilt_angles']
            yaw_deltas = metrics['yaw_deltas']
            
            total_trials = len(tilt_angles)
            # Count dropped trials
            dropped_trials = sum(1 for angle in tilt_angles if angle > tilt_threshold)
            drop_rate = dropped_trials / total_trials if total_trials > 0 else 0
            
            # Filter yaw_deltas for trials that are not dropped 
            valid_yaw_deltas = [yaw for angle, yaw in zip(tilt_angles, yaw_deltas) if angle <= tilt_threshold]
            if valid_yaw_deltas:
                avg_yaw_delta = sum(valid_yaw_deltas) / len(valid_yaw_deltas)
            else:
                avg_yaw_delta = None  # or use float('nan') if preferred
            
            print(f"Method: {method_name}")
            print(f"  Drop Rate: {drop_rate*100:.2f}%")
            print(f"  Average Yaw Delta (non-dropped): {avg_yaw_delta:.2f}")

            boxplot_data.append(valid_yaw_deltas)
    
        # Create the boxplot; patch_artist=True allows us to color the boxes.
        bp = plt.boxplot(
            boxplot_data, 
            labels=[method.upper() for method in method_names], 
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

        # plt.ylim(0, 5.1)
        plt.xlabel('Method', fontsize=ts)
        plt.ylabel('Degrees Turned', fontsize=ts)
        plt.title('Boxplot of Task Completion (Degrees the screwdriver turned) for Method', fontsize=ts)
        plt.xticks(fontsize=ts-2)
        plt.yticks(fontsize=ts-2)
        plt.tight_layout()
        plt.show()
    