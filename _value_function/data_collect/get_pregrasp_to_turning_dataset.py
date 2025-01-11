import numpy as np
import pickle as pkl
import pathlib
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, emailer, delete_imgs
fpath = pathlib.Path(f'{CCAI_PATH}/data')

loop_idx = 0
prog_id = 'aq'
trials_per_save = 100
perception_noise = 0.0
delete_imgs()

while True:
    pose_tuples = []
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)

    for i in tqdm(range(trials_per_save)):

        img_save_dir = None
        # img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{i}')
        # pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        # env.frame_fpath = img_save_dir
        # env.frame_id = 0
    
        initial_pose, planned_pose = pregrasp(env, config, chain, deterministic=False, perception_noise=perception_noise, 
                        image_path = img_save_dir, initialization = None, mode='no_vf', iters = 80)
        
        print("done pregrasp")
        
        _, final_pose, succ, full_trajectory = do_turn(initial_pose, config, env, 
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                        perception_noise=perception_noise, image_path = img_save_dir)
        print("done turn")
        
        pose_tuples.append((initial_pose, final_pose, full_trajectory))

    if perception_noise == 0:
        savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/pregrasp_to_turn_dataset_{prog_id}_{loop_idx}.pkl'
    else:
        savepath = f'{fpath.resolve()}/pregrasp_to_turn_datasets/noisy_pregrasp_to_turn_dataset_{prog_id}_{loop_idx}.pkl'

    pkl.dump(pose_tuples, open(savepath, 'wb'))

    loop_idx += 1
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

emailer().send()