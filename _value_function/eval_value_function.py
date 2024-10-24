import numpy as np
import pickle as pkl
import pathlib
import sys
from tqdm import tqdm
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from get_initial_poses import emailer
from screwdriver_problem import init_env, do_turn
import torch


fpath = pathlib.Path(f'{CCAI_PATH}/data')
with open(f'{fpath.resolve()}/eval/initial_and_optimized_poses.pkl', 'rb') as file:
    tuples = pkl.load(file)
    initial_poses, optimized_poses = zip(*tuples)

    initial_poses = np.array(initial_poses).reshape(-1,20)
    initial_poses = torch.from_numpy(initial_poses).float()

    optimized_poses = np.array(optimized_poses).reshape(-1,20)
    optimized_poses = torch.from_numpy(optimized_poses).float()


if __name__ == "__main__":

    pose_tuples = []

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)

    for i in tqdm(range(len(initial_poses))):

        _, initial_final_pose, succ = do_turn(initial_poses[i].reshape(1,20), 
                                              config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)
 
        _, optimized_final_pose, succ = do_turn(optimized_poses[i].reshape(1,20), 
                                                config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial)


        pose_tuples.append((initial_poses[i], optimized_poses[i], initial_final_pose, optimized_final_pose))

    
    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    start_idx = config['start_idx']
    savepath = f'{fpath.resolve()}/eval/final_pose_comparisons_mse_50samples.pkl'
    with open(savepath, 'wb') as f:
        pkl.dump(pose_tuples, f)

    print(f'saved to {savepath}')
    #emailer().send()