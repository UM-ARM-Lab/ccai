from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
from _value_function.train_value_function_regrasp import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
import pytorch_kinematics.transforms as tf
from scipy.spatial.transform import Rotation as R

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
save_dir = pathlib.Path(f'{CCAI_PATH}/data/figures')
import torch

def calculate_d2g(initial_pose, final_pose):
    turn_angle = np.pi/2

    screwdriver_pose = initial_pose.flatten()[-4:-1]

    if screwdriver_pose[2] < -np.pi:
        screwdriver_pose[2] += 4 * np.pi

    screwdriver_goal = np.array([0, 0, -turn_angle]) + screwdriver_pose

    screwdriver_goal_mat = R.from_euler('xyz', screwdriver_goal).as_matrix()
    screwdriver_state = final_pose.flatten()[-4:-1]
    screwdriver_mat = R.from_euler('xyz', screwdriver_state).as_matrix()
    # make both matrices 3D (batch_size, 3, 3)
    screwdriver_mat = torch.tensor(screwdriver_mat).unsqueeze(0)
    screwdriver_goal_mat = torch.tensor(screwdriver_goal_mat).unsqueeze(0).repeat(screwdriver_mat.shape[0], 1, 1)
    distance2goal = tf.so3_relative_angle(screwdriver_mat, screwdriver_goal_mat, cos_angle=False).detach().cpu()
    d2g = float(distance2goal.numpy().flatten().reshape(1))
    return d2g

if __name__ == "__main__":

    annotate = True

    experiment_names = ['test_method_test_official_high_iter_all', 
                        "test_method_test_official_high_iter_singlevf",
                        "test_method_test_official_high_iter_diffusion10k_with_contact",
                        "test_method_test_official_high_iter_diffusion10k_no_contact_and_combined",
                        ] 
    budget = "High Budget"

    # experiment_names = ['test_method_test_official_all',
    #                      'test_method_test_official_low_iter_diffusion10k_no_contact',]

    experiment_names = ['test_method_test_official_low_iter_novfrepeat',]
    budget = "Low Budget"

    results = {}

    for name in experiment_names:
        file_path = fpath / 'test' / f'{name}.pkl'
        with open(file_path, 'rb') as file:
            single_results = pkl.load(file)
        
        for method, method_results in single_results.items():

            if method == "no_vf":
                method = "Vanilla"
            elif method == "vf":
                method = "AVO (ours)"
            elif method == "diffusion":
                method = "Diffusion"
            elif method == "diffusion_no_contact_cost":
                method = "Diffusion"
            elif method == "diffusion_w_contact_cost":
                method = "Diffusion+"
            elif method == 'singlevf':
                method = "Single VF"
            elif method == "combined":
                method = "Diffusion+VF"
            else:
                print("Invalid method")
                exit()

            if method in results:
                print(f"Overriding {method}.")
                
            if budget == "Low Budget":
                if method != "Vanilla" and method != "AVO (ours)":
                    continue

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
            'tilt_angles': [],
            'yaw_deltas': [],
        }
        for pregrasp_index in range(n_trials):
            all_d2gs = []

            for repeat_index in range(n_repeat):

                pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj, *initial_samples = \
                    results[method_name][(pregrasp_index, repeat_index)]

                d2g = calculate_d2g(regrasp_pose.numpy(), turn_pose)
                            
                all_d2gs.append(d2g)

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
    if False:
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
        plt.xlabel('Pregrasp Index', fontsize=ts)
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
            tilt_threshold = 10000
            tilt_angles = [float(angle) for angle in metrics['tilt_angles']]
            d2gs = [float(d2g) for d2g in metrics['d2gs']]
            total_trials = len(tilt_angles)
            
            # Filter yaw_deltas for trials that are not dropped 
            valid_d2gs = [d2g for angle, d2g in zip(tilt_angles, d2gs) if angle <= tilt_threshold]
            avg_d2g = sum(valid_d2gs) / len(valid_d2gs)

            # import statistics
            # std_tilt = statistics.stdev(tilt_angles)
            # std_yaw = statistics.stdev(valid_d2gs)

            boxplot_data.append(valid_d2gs)

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
    if True:
        # ----  Boxplot ----
        # Create a new figure for the boxplot
        plt.figure(figsize=(10, 5))

        # Prepare data for boxplot: a list of d2g lists (one per method)
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
        plt.title(f'Simulation, {budget}: Degrees the screwdriver turned', fontsize=ts)
        plt.xticks(fontsize=ts-2)
        plt.yticks(fontsize=ts-2)
        plt.tight_layout()
        plt.savefig(save_dir / f"{budget} deg turned sim.png")
        plt.show()