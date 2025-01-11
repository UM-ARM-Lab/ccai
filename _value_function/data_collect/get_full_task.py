import numpy as np
import pickle as pkl
import pathlib
import shutil
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer

fpath = pathlib.Path(f'{CCAI_PATH}/data')

loop_idx = 0
prog_id = 'a'
trials_per_save = 20
perception_noise = 0.0

while True:
    pregrasp_to_turn_tuples = []
    turn_to_regrasp_tuples = []
    regrasp_to_turn_tuples = []
    full_tuples = []

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    for i in tqdm(range(trials_per_save)):
        #pregrasp from random initialization

        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{i}')
        pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        env.frame_fpath = img_save_dir
        env.frame_id = 0
    
        pregrasp_pose = pregrasp(env, config, chain, deterministic=False, perception_noise=perception_noise, 
                        image_path = img_save_dir, initialization = None, useVFgrads=False, iters = 80)
        #turn0
        _, turn0_final_pose, turn0_succ, turn0_full_trajectory = do_turn(pregrasp_pose, config, env, 
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                        perception_noise=perception_noise, image_path = img_save_dir)
        #regrasp
        regrasp_pose = regrasp(env, config, gym, viewer, chain, state2ee_pos_partial, perception_noise=perception_noise, initialization = None, 
                        image_path = img_save_dir, useVFgrads=False)
        
        #turn1
        _, turn1_final_pose, turn1_succ, turn1_full_trajectory = do_turn(regrasp_pose, config, env, 
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                        perception_noise=perception_noise, image_path = img_save_dir)

        full_tuples.append((pregrasp_pose, turn0_final_pose, turn0_full_trajectory, regrasp_pose, 
                            turn1_final_pose, turn1_full_trajectory))
        


    full_savepath = f'{fpath.resolve()}/value_datasets/full_task_dataset_{prog_id}_{loop_idx}.pkl'
    pkl.dump(full_tuples, open(full_savepath, 'wb'))

    loop_idx += 1
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    # exit()

# emailer().send()