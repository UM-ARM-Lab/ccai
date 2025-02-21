import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = '/home/newuser/Desktop/Honda/ccai/data/test/print'
cost_file_novf = os.path.join(data_dir, 'cost_stats_novf.npy')
constraint_file_novf = os.path.join(data_dir, 'constraint_stats_novf.npy')
cost_file_diff = os.path.join(data_dir, 'cost_stats_diff.npy')
constraint_file_diff = os.path.join(data_dir, 'constraint_stats_diff.npy')

# Load the data from each .npy file
cost_stats_novf = np.load(cost_file_novf)
constraint_stats_novf = np.load(constraint_file_novf)
cost_stats_diff = np.load(cost_file_diff)
constraint_stats_diff = np.load(constraint_file_diff)

# Reshape the arrays (assuming they have 12 rows; adjust as needed)
cost_stats_novf = cost_stats_novf.reshape(12, -1)[0, :]
constraint_stats_novf = constraint_stats_novf.reshape(12, -1)[0, :]

cost_stats_diff = cost_stats_diff[:120].reshape(12, -1)[0, :]
constraint_stats_diff = constraint_stats_diff[:120].reshape(12, -1)[0, :]

# Create iteration indices for each method
iterations_novf = np.arange(len(cost_stats_novf))
iterations_diff = np.arange(len(cost_stats_diff))

# Set up the figure with two subplots: one for cost and one for constraint violation
plt.figure(figsize=(12, 5))

# Scatter plot for Average Cost
plt.subplot(1, 2, 1)
plt.scatter(iterations_novf, cost_stats_novf, color='blue', marker='o', label='Vanilla')
plt.scatter(iterations_diff, cost_stats_diff, color='green', marker='x', label='Diffusion')
plt.xlabel('Trial')
plt.ylabel('Average Cost')
plt.title('Average Cost of Initializations')
plt.legend()
plt.grid(True)

# Scatter plot for Average Constraint Violation
plt.subplot(1, 2, 2)
plt.scatter(iterations_novf, constraint_stats_novf, color='blue', marker='o', label='Vanilla')
plt.scatter(iterations_diff, constraint_stats_diff, color='green', marker='x', label='Diffusion')
plt.xlabel('Trial')
plt.ylabel('Average Constraint Violation')
plt.title('Constraint Violation of Initializations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()