from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
from _value_function.train_value_function_regrasp import Net, query_ensemble, load_ensemble
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

data = {}
methods = ['vf', 'novf', 'vf_low']
# params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

for method in methods:
    data[method] = {
            'costs': [],
            'tilt_angles': [],
            'yaw_deltas': [],
        }
    for i in range(10):
        file = f'{fpath.resolve()}/test/hardware_results/{method}_trial_{i}.pkl'
        if not pathlib.Path(file).exists():
            continue
        with open(file, 'rb') as file:
            pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj, *initial_samples = pkl.load(file)
            # results[method_name][(pregrasp_index, repeat_index)]\
            cost = calculate_turn_cost(regrasp_pose.numpy(), turn_pose)

            pose_i = regrasp_pose.numpy().flatten()[-4:-1]*180/np.pi
            pose_f = turn_pose.flatten()[-4:-1]*180/np.pi
            if pose_i[-1] < -180:
                pose_i[-1] += 720
            if pose_f[-1] < -180:
                pose_f[-1] += 720
            tilt_angle = max(abs(pose_f[:2]))
            yaw_delta = (pose_i[-1] - pose_f[-1]) 
            if yaw_delta > 100 or yaw_delta < -100:
                # print("bad")
                # exit()
                pass
            
            data[method]['costs'].append(cost)
            data[method]['tilt_angles'].append(tilt_angle)
            if tilt_angle < 10:
                data[method]['yaw_deltas'].append(yaw_delta)

        # for j in range(13):
        #     env.reset(torch.from_numpy(turn_traj[j]).reshape(1, 20).float())
        #     time.sleep(0.1)

    print(f"{method} costs: {data[method]['costs']}")
    # print(f"{method} tilt angles: {data[method]['tilt_angles']}")
    print(f"{method} yaw deltas: {data[method]['yaw_deltas']}")

    print(f"{method} average cost: {np.mean(data[method]['costs'])}")
    print(f"{method} average yaw delta: {np.mean(data[method]['yaw_deltas'])}")
    




colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
ts = 16
# ----  Boxplot ----
# Create a new figure for the boxplot
plt.figure(figsize=(10, 5))

# Prepare data for boxplot: a list of cost lists (one per method)
boxplot_data = [data[method]['costs'] for method in methods]

# Create the boxplot; patch_artist=True allows us to color the boxes.
bp = plt.boxplot(
    boxplot_data, 
    labels=[method for method in methods], 
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
for i, method in enumerate(methods):
    median_val = np.median(data[method]['costs'])
    plt.text(
        i + 1, 
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
plt.title('Turning Costs (Hardware)', fontsize=ts)
plt.xticks(fontsize=ts-2)
plt.yticks(fontsize=ts-2)
plt.tight_layout()
plt.show()

# ----  Boxplot ----
# Create a new figure for the boxplot
plt.figure(figsize=(10, 5))

# Prepare data for boxplot: a list of cost lists (one per method)
boxplot_data = []

for method_name, metrics in data.items():
    tilt_threshold = 10
    tilt_angles = [float(angle) for angle in metrics['tilt_angles']]
    yaw_deltas = [float(yaw) for yaw in metrics['yaw_deltas']]
    
    total_trials = len(tilt_angles)
    # Count dropped trials
    dropped_trials = sum(1 for angle in tilt_angles if angle > tilt_threshold)
    drop_rate = dropped_trials / total_trials if total_trials > 0 else 0
    
    # Filter yaw_deltas for trials that are not dropped 
    valid_yaw_deltas = [yaw for angle, yaw in zip(tilt_angles, yaw_deltas) if angle <= tilt_threshold]
    avg_yaw_delta = sum(valid_yaw_deltas) / len(valid_yaw_deltas)

    import statistics
    std_tilt = statistics.stdev(tilt_angles)
    std_yaw = statistics.stdev(valid_yaw_deltas)
    
    # Print the results
    print(f"Method: {method_name}")
    print(f"  Drop Rate: {drop_rate*100:.2f}%")
    print(f"  Average Yaw Delta (non-dropped): {avg_yaw_delta:.2f} +- {std_yaw:.2f}")
    # print(f"  Std Dev Tilt Angles: {std_tilt:.2f}")

    boxplot_data.append(valid_yaw_deltas)

# Create the boxplot; patch_artist=True allows us to color the boxes.
bp = plt.boxplot(
    boxplot_data, 
    labels=[method for method in methods], 
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
for i, method in enumerate(methods):
    median_val = np.median(boxplot_data[i])
    plt.text(
        i + 1, 
        median_val - 0.02, 
        f"{median_val:.2f}", 
        horizontalalignment='center', 
        verticalalignment='bottom',
        fontsize=12,
        color='black'
    )

# plt.ylim(0, 5.1)
plt.xlabel('Method', fontsize=ts)
plt.ylabel('Degrees Turned', fontsize=ts)
plt.title('Degrees the screwdriver turned (Hardware)', fontsize=ts)
plt.xticks(fontsize=ts-2)
plt.yticks(fontsize=ts-2)
plt.tight_layout()
plt.show()