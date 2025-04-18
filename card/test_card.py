import numpy as np
import pickle as pkl
import pathlib
import sys
import time
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import emailer, delete_imgs
from card.card_problem import init_env, pull_index, pull_middle
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
from card.process_card_data import calculate_cost
from card.get_card_dataset import get_initialization
from itertools import product

if len(sys.argv) == 2:
    config_path = f'card{sys.argv[1]}.yaml'
else:
    config_path = 'card0.yaml'

def load_or_create_checkpoint(checkpoint_path, method_names):

    if checkpoint_path.exists():
        print(f"Loading existing checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pkl.load(f)
    else:
        print(f"No checkpoint found. Creating a new one at {checkpoint_path}. Also deleting imgs because we're starting a new test")
        delete_imgs()
       
        checkpoint = {
            'results': {},
            'tested_combinations': set()
        }
        for m in method_names:
            checkpoint['results'][m] = {}

    return checkpoint

def get_initializations(env, config, chain, sim_device, n_samples, card_noise_mag0, card_noise_mag1, finger_noise_mag, save=False, name='test_inits'):

    savepath = f'{fpath.resolve()}/card/test/initializations/{name}.pkl'
   
    initializations = []
    for _ in range(n_samples):
        initialization = get_initialization(env, sim_device, card_noise_mag0, card_noise_mag1, finger_noise_mag)
        initializations.append(initialization)

    if save == True:
        pkl.dump(initializations, open(savepath, 'wb'))
    return initializations

def save_checkpoint(checkpoint):

    with open(checkpoint_path, 'wb') as f:
        pkl.dump(checkpoint, f)

if __name__ == '__main__':

    test_name = 'testswap2'
    checkpoint_path = fpath / 'card' /'test'/'test_method'/f'checkpoint_{test_name}.pkl'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    warmup_iters = 50
    online_iters = 20

    n_trials = 50
    n_repeat = 1
    test_time = False
    vf_times = []
    no_vf_times = []

    calc_vf = True
    calc_novf = True

    method_names = []
    if calc_vf:
        method_names.append("vf")
    if calc_novf:
        method_names.append("no_vf")

    card_noise_mag0, card_noise_mag1, finger_noise_mag = 0.06, 0.2, 0.2

    vf_weight_i = 100.0
    other_weight_i = 1.3
    variance_ratio_i = 32.0

    vf_weight_m = vf_weight_i
    other_weight_m = other_weight_i
    variance_ratio_m = variance_ratio_i

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer = init_env(visualize=True, config_path=config_path)

    init_path = fpath / 'card' /'test'/'initializations'/'test_inits.pkl'

    if init_path.exists() == False or len(pkl.load(open(init_path, 'rb'))) != n_trials:
        print("Generating new pregrasp initializations...")
        get_initializations(env, config, chain, config['sim_device'], n_trials,
                            card_noise_mag0, card_noise_mag1, finger_noise_mag, save=True,
                            name='test_inits')
   
    inits = pkl.load(open(init_path, 'rb'))
    init_indices = list(range(n_trials))
    repeat_indices = list(range(n_repeat))
    args = [init_indices, repeat_indices]

    checkpoint = load_or_create_checkpoint(checkpoint_path=checkpoint_path, method_names=method_names)

    total_vf_cost = 0
    total_no_vf_cost = 0

    for init_index, repeat_index in product(*args):
        combo_tuple = (init_index, repeat_index)

        if combo_tuple in checkpoint['tested_combinations']:
            continue

        print(f"Testing combination {combo_tuple}")

        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/card/imgs/test/trial_{n_repeat*init_index+repeat_index+1}')
        pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        env.frame_fpath = img_save_dir
        env.frame_id = 0
       
        init_pose = inits[init_index]

        if calc_vf:

            env.reset(dof_pos= init_pose)
        
            final_state, full_traj0 = pull_index(env, config, chain, img_save_dir, warmup_iters, online_iters,
                        model_name = 'index_vf', mode='vf', task = 'index1',
                        vf_weight = vf_weight_i, other_weight = other_weight_i, variance_ratio = variance_ratio_i,
                        )
            final_state, full_traj1 = pull_middle(env, config, chain, img_save_dir, warmup_iters, online_iters,
                            model_name = 'middle_vf', mode='vf', task = 'middle_vf',
                            vf_weight = vf_weight_m, other_weight = other_weight_m, variance_ratio = variance_ratio_m,
                            )
            final_state, full_traj2 = pull_index(env, config, chain, img_save_dir, warmup_iters, online_iters,
                            model_name = 'index_vf', mode='vf', task = 'index2',
                            vf_weight = vf_weight_i, other_weight = other_weight_i, variance_ratio = variance_ratio_i,
                            )

            # if test_time:
            #     vf_times.append(compute_time_rg + compute_time_t)

            result_vf = [init_pose, final_state, full_traj0, full_traj1, full_traj2]
            checkpoint['results']['vf'][combo_tuple] = result_vf

            cost = calculate_cost(init_pose, final_state)
            total_vf_cost += cost
            print('---------------------------------')
            print(f"VF cost: {cost}")

        if calc_novf:

            env.reset(dof_pos= init_pose)
        
            final_state, full_traj0 = pull_index(env, config, chain, img_save_dir, warmup_iters, online_iters,
                        mode='no_vf', 
                        )
            final_state, full_traj1 = pull_middle(env, config, chain, img_save_dir, warmup_iters, online_iters,
                            mode='no_vf', 
                            )
            final_state, full_traj2 = pull_index(env, config, chain, img_save_dir, warmup_iters, online_iters,
                            mode='no_vf', 
                            )

            # if test_time:
            #     vf_times.append(compute_time_rg + compute_time_t)

            result_no_vf = [init_pose, final_state, full_traj0, full_traj1, full_traj2]
            checkpoint['results']['no_vf'][combo_tuple] = result_no_vf

            cost = calculate_cost(init_pose, final_state)
            total_no_vf_cost += cost
            print('---------------------------------')
            print(f"no VF cost: {cost}")
        
        # if test_time:
        #     print(f"Mean VF time: {np.mean(vf_times)}")
        #     print(f"Mean No VF time: {np.mean(no_vf_times)}")

        checkpoint['tested_combinations'].add(combo_tuple)
        save_checkpoint(checkpoint)
        savepath = f'{fpath.resolve()}/card/test/test_method/result_{test_name}.pkl'
        pkl.dump(checkpoint['results'], open(savepath, 'wb'))

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    print(f"Avg VF cost: {total_vf_cost/n_trials/n_repeat}")
    print(f"Avg No VF cost: {total_no_vf_cost/n_trials/n_repeat}")
    emailer().send()