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
fpath = pathlib.Path(f'{CCAI_PATH}/data')

loop_idx = 0
prog_id = 'a'
trials_per_save = 5
perception_noise = 0.0
pregrasp_iters = 80
regrasp_iters = 200
delete_imgs()

while True:
    pose_tuples = []
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    for i in tqdm(range(trials_per_save)):

        # img_save_dir = None
        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/regrasp_trial_{i+1}')
        pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        env.frame_fpath = img_save_dir
        env.frame_id = 0

        initialization = get_initialization(config['sim_device'], env, 
                    max_screwdriver_tilt=0.01, screwdriver_noise_mag=0.01, finger_noise_mag=0.3)
        
        pregrasp_pose, planned_pose = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise, 
                        image_path = img_save_dir, initialization = initialization, mode='no_vf', iters = pregrasp_iters)

        print("done pregrasp")

        regrasp_pose, regrasp_traj = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=perception_noise, 
                                image_path = img_save_dir, initialization = pregrasp_pose, mode='no_vf', iters = regrasp_iters)
        
        print("done regrasp")
        
        _, turn_pose, succ, turn_traj = do_turn(regrasp_pose, config, env, 
                        sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                        perception_noise=perception_noise, image_path = img_save_dir)
        
        print("done turn")
        
        pose_tuples.append((pregrasp_pose, regrasp_pose, regrasp_traj, turn_pose, turn_traj))

    if perception_noise == 0:
        savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/regrasp_to_turn_dataset_{prog_id}_{loop_idx}.pkl'
    else:
        savepath = f'{fpath.resolve()}/regrasp_to_turn_datasets/noisy_regrasp_to_turn_dataset_{prog_id}_{loop_idx}.pkl'

    pkl.dump(pose_tuples, open(savepath, 'wb'))

    loop_idx += 1
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

emailer().send()