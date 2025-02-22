from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
from _value_function.screwdriver_problem import init_env, convert_full_to_partial_config, convert_partial_to_full_config
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
import time

diff_path = '/home/newuser/Desktop/Honda/ccai/data/test/see_constraints_results_diffusion.pkl'
novf_path = '/home/newuser/Desktop/Honda/ccai/data/test/see_constraints_results_no_vf.pkl'


config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

novf_costs = []
with open(novf_path, 'rb') as file:
    novf_results = pkl.load(file)
    for result in novf_results:
        novf_costs.append(calculate_turn_cost(result[1].numpy(), result[3]))

diff_costs = []
with open(diff_path, 'rb') as file:
    diff_results = pkl.load(file)
    for result in diff_results:
        diff_costs.append(calculate_turn_cost(result[1].numpy(), result[3]))
        initial_samples = result[5][0,:,:15]
        initial_samples = convert_partial_to_full_config(initial_samples)
        for i in range(initial_samples.shape[0]):
            env.reset(dof_pos=initial_samples[i].reshape(1,20).float())
            time.sleep(0.1)

#scatter the costs
plt.scatter(range(len(novf_costs)), novf_costs, label='Vanilla')
plt.scatter(range(len(diff_costs)), diff_costs, label='Diffusion')
plt.legend()
plt.show()


data_dir = '/home/newuser/Desktop/Honda/ccai/data/test/print'
cost_file_novf = os.path.join(data_dir, 'cost_stats_novf_save.npy')
constraint_file_novf = os.path.join(data_dir, 'constraint_stats_novf_save.npy')
cost_file_diff = os.path.join(data_dir, 'cost_stats_diff_save.npy')
constraint_file_diff = os.path.join(data_dir, 'constraint_stats_diff_save.npy')

# Load the data from each .npy file
cost_stats_novf = np.load(cost_file_novf)
constraint_stats_novf = np.load(constraint_file_novf)
cost_stats_diff = np.load(cost_file_diff)
constraint_stats_diff = np.load(constraint_file_diff)

# Create iteration indices for each method
iterations_novf = np.arange(len(cost_stats_novf))
iterations_diff = np.arange(len(cost_stats_diff))

# Set up the figure with two subplots: one for cost and one for constraint violation
plt.figure(figsize=(12, 5))

# Scatter plot for Average Cost
plt.subplot(1, 2, 1)

plt.scatter(iterations_novf[::24], cost_stats_novf[::24],
            color='blue', marker='o', s=100, label='Vanilla')
plt.scatter(iterations_diff[::24], cost_stats_diff[::24],
            color='green', marker='x', s=100, label='Diffusion')

plt.xlabel('Trial')
plt.ylabel('Average Cost')
plt.title('Average Cost of Initializations')
plt.legend()
plt.grid(True)

# Scatter plot for Average Constraint Violation
plt.subplot(1, 2, 2)


plt.scatter(iterations_novf[::24], constraint_stats_novf[::24],
            color='blue', marker='o', s=100, label='Vanilla')
plt.scatter(iterations_diff[::24], constraint_stats_diff[::24],
            color='green', marker='x', s=100, label='Diffusion')

plt.xlabel('Trial')
plt.ylabel('Average Constraint Violation')
plt.title('Average Constraint Violations of Initializations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



exit()
# cost_file_novf2 = os.path.join(data_dir, 'cost_stats_novf_save.npy')
# constraint_file_novf2 = os.path.join(data_dir, 'constraint_stats_novf_save.npy')
# cost_file_diff2 = os.path.join(data_dir, 'cost_stats_diff_save.npy')
# constraint_file_diff2 = os.path.join(data_dir, 'constraint_stats_diff_save.npy')

# np.save(cost_file_novf2, cost_stats_novf)
# np.save(constraint_file_novf2, constraint_stats_novf)
# np.save(cost_file_diff2, cost_stats_diff)
# np.save(constraint_file_diff2, constraint_stats_diff)
# exit()
