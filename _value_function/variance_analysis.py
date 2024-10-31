import numpy as np
import pickle as pkl
import pathlib
import sys
from tqdm import tqdm
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from screwdriver_problem import init_env, do_turn, emailer
from process_final_poses import calculate_cost
import torch
import matplotlib.pyplot as plt
import time


fpath = pathlib.Path(f'{CCAI_PATH}/data')

# filename = 'final_pose_comparisons_ablation_ensemble_SGD_10k_iters.pkl'
# with open(f'{fpath.resolve()}/eval/{filename}', 'rb') as file:
#     tuples = pkl.load(file)
#     initial_poses, optimized_poses, initial_final_poses, optimized_final_poses = zip(*tuples)
#     initial_final_poses = np.array(initial_final_poses).reshape(-1, 20)
#     optimized_final_poses = np.array(optimized_final_poses).reshape(-1, 20)

filename = '/initial_poses/initial_poses_10k.pkl'
with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
    total_poses  = pkl.load(file)
    total_poses = np.array([tensor.numpy() for tensor in total_poses]).reshape(-1, 20)


savepath = f'{fpath.resolve()}/eval/variance_analysis.pkl'

def get_data():
    start, end = 0, 10
    initial_poses = total_poses[start:end+1]
    
    index_cost_tuples = []

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)

    for i in tqdm(range(len(initial_poses))):

        costs = []

        for _ in range(5):
            _, final_pose, succ = do_turn(torch.tensor(initial_poses[i]).reshape(1,20), 
                                    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)
            cost = calculate_cost(initial_poses[i].reshape(20,), final_pose.reshape(20,))[0]
            costs.append(cost)
        index_cost_tuples.append((i+start, costs))
    
    with open(savepath, 'wb') as f:
        pkl.dump(index_cost_tuples, f)

    print(f'saved to {savepath}')


if __name__ == "__main__":
    t0 = time.time()
    get_data()
    with open(savepath, 'rb') as file:
        index_cost_tuples = pkl.load(file)
    print(index_cost_tuples)
    
    indices = [item[0] for item in index_cost_tuples]
    cost_lists = [item[1] for item in index_cost_tuples]

    # Plot each cost value as a separate point
    for index, costs in zip(indices, cost_lists):
        plt.scatter([index] * len(costs), costs)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Cost Value')
    plt.title('Costs at Each Index')
    plt.show()

    print(f'Time taken for 50 rollouts!!!!: {time.time() - t0:.2f} seconds')