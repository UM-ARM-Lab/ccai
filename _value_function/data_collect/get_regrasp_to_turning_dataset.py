import numpy as np
import pickle as pkl
import pathlib
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, delete_imgs
from _value_function.test.test_method import get_initialization
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def validate_pregrasp_pose(pregrasp_pose):
    screwdriver = pregrasp_pose[0,-4:-2]
    # print(screwdriver)
    # print(np.linalg.norm(screwdriver))
    if np.linalg.norm(screwdriver) > 0.3 :
        return False
    else:
        return True

loop_idx = 0
prog_id = 'g'
trials_per_save = 5
perception_noise = 0.0
pregrasp_iters = 80
regrasp_iters = 100
turn_iters = 200
delete_imgs()

visualize = False
config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=visualize)
sim_device = config['sim_device']


while True:

    pose_tuples = []
    
    trials_done = 0

    while trials_done < trials_per_save:

        if visualize:
            img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/regrasp_trial_{trials_done+1}')
            pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        else:
            img_save_dir = None

        env.frame_fpath = img_save_dir
        env.frame_id = 0

        print(f"Starting Trial {trials_done+1}")

        initialization = get_initialization(env, sim_device, max_screwdriver_tilt=0.015, screwdriver_noise_mag=0.015, finger_noise_mag=0.25)
        
        pregrasp_pose, planned_pose = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise, 
                        image_path = img_save_dir, initialization = initialization, mode='no_vf', iters = pregrasp_iters)

        flag = validate_pregrasp_pose(pregrasp_pose)
        if flag:
            print("done pregrasp")
        else:
            print("pregrasp failed")
            continue
        
        regrasp_pose, regrasp_traj = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=perception_noise, 
                                image_path = img_save_dir, initialization = pregrasp_pose, mode='no_vf', iters = regrasp_iters)
        
        print("done regrasp")
        
        _, turn_pose, succ, turn_traj = do_turn(regrasp_pose, config, env, 
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                        iters = turn_iters, perception_noise=perception_noise, image_path = img_save_dir)
        
        print("done turn")
        
        pose_tuples.append((pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj))
        trials_done += 1

    if perception_noise == 0:
        savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/regrasp_to_turn_dataset_{prog_id}_{loop_idx}.pkl'
    else:
        savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/noisy_regrasp_to_turn_dataset_{prog_id}_{loop_idx}.pkl'

    pkl.dump(pose_tuples, open(savepath, 'wb'))

    loop_idx += 1

emailer().send()