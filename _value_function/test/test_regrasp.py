import numpy as np
import pickle as pkl
import pathlib
import shutil
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, convert_partial_to_full_config, delete_imgs
from _value_function.test.test_method import get_initialization
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')

if __name__ == '__main__':

    # test_name = 'regrasp'
    n_trials = 8
    img_save_dir = None
    pregrasp_iters = 80
    regrasp_iters = 200
    perception_noise = 0.0

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    
    regrasp_trajs = []
    for q in range(n_trials):

        initialization = get_initialization(config['sim_device'], env, 
                    max_screwdriver_tilt=0.01, screwdriver_noise_mag=0.01, finger_noise_mag=0.3)
        
        pregrasp_pose_vf, plan_vf = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise, 
                                image_path = img_save_dir, initialization = initialization, 
                                model_name = 'ensemble', mode='no_vf', iters = pregrasp_iters,
                                vf_weight = 10.0, other_weight = 0.1, variance_ratio = 5)

        regrasp_pose, regrasp_traj = regrasp(env, config, chain, state2ee_pos_partial, perception_noise=perception_noise, 
                                            initialization = pregrasp_pose_vf, mode='no_vf', iters = regrasp_iters)

        regrasp_trajs.append(regrasp_traj)

    savepath = f'{fpath.resolve()}/test/regrasp.pkl'
    pkl.dump(regrasp_trajs, open(savepath, 'wb'))

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    # emailer().send()