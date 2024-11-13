import numpy as np
import pickle as pkl
import pathlib
import sys
from tqdm import tqdm
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from screwdriver_problem import init_env, do_turn, emailer
import torch

# experiment_name = '_single_SGD_1k_iters'
# experiment_name = '_single_SGD_10k_iters'
# experiment_name = '_single_Adam_1k_iters'

# experiment_name = '_ensemble_SGD_10k_iters'
# experiment_name = '_ensemble_SGD_1k_iters'
# experiment_name = '_ensemble_Adam_1k_iters'

# experiment_name = '_ensemble_SGD_100k_iters'
# experiment_name = '_ensemble_Adam_100k_iters'
# experiment_name = '_ensemble_Adam_500_iters_optimal'
experiment_name = 'test_100'


fpath = pathlib.Path(f'{CCAI_PATH}/data')

#_____________________________________________________________________________________
# code for evaluating grad descent poses
#_____________________________________________________________________________________

# with open(f'{fpath.resolve()}/eval/initial_and_optimized_poses{experiment_name}.pkl', 'rb') as file:
#     tuples = pkl.load(file)
#     initial_poses, optimized_poses = zip(*tuples)

#     initial_poses = np.array(initial_poses).reshape(-1,20)
#     initial_poses = torch.from_numpy(initial_poses).float()

#     optimized_poses = np.array(optimized_poses).reshape(-1,20)
#     optimized_poses = torch.from_numpy(optimized_poses).float()

#_____________________________________________________________________________________
# code for evaluating actual pregrasp poses
#_____________________________________________________________________________________

with open(f'{fpath.resolve()}/test/{experiment_name}.pkl', 'rb') as file:
    tuples = pkl.load(file)
    _, optimized_poses, initial_poses = zip(*tuples)
    initial_poses = torch.stack(initial_poses).reshape(-1,20).float()
    optimized_poses = torch.stack(optimized_poses).reshape(-1,20).float()

#_____________________________________________________________________________________

if __name__ == "__main__":

    pose_tuples = []

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)


    for i in tqdm(range(len(initial_poses))):
        _, initial_final_pose, succ, initial_full_trajectory = do_turn(initial_poses[i].reshape(1, 20), 
                                                config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)

        _, optimized_final_pose, succ, optimized_full_trajectory = do_turn(optimized_poses[i].reshape(1, 20), 
                                                config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)

        pose_tuples.append((initial_poses[i], optimized_poses[i], 
                            initial_final_pose, optimized_final_pose, 
                            initial_full_trajectory, optimized_full_trajectory))


    savepath = f'{fpath.resolve()}/eval/final_pose_comparisons{experiment_name}.pkl'
    with open(savepath, 'wb') as f:
        pkl.dump(pose_tuples, f)

    print(f'Saved data to {savepath}')
    emailer().send()