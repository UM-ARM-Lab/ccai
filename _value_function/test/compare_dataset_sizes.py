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
from _value_function.test.test_method import get_initializations
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
from itertools import product

def load_or_create_checkpoint(checkpoint_path, method_names):
    """
    Creates or loads a checkpoint whose structure supports
    a flexible number of methods. Each method gets its own 
    dictionary under checkpoint['results'][method_name].
    """
    if checkpoint_path.exists():
        print(f"Loading existing checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pkl.load(f)
    else:
        print(f"No checkpoint found. Creating a new one at {checkpoint_path}. Also deleting imgs because we're starting a new test")
        delete_imgs()
        # Initialize an empty dictionary for each method:
        checkpoint = {
            'results': {},
            'tested_combinations': set()
        }
        for m in method_names:
            checkpoint['results'][m] = {}
    return checkpoint

def save_checkpoint(checkpoint, checkpoint_path):
    with open(checkpoint_path, 'wb') as f:
        pkl.dump(checkpoint, f)

if __name__ == '__main__':

    test_name = '0'
    checkpoint_path = fpath / 'test' / 'compare_sizes' / f'checkpoint_{test_name}.pkl'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # vf_sizes = [1, 2, 4, 8]
    # vf_names = []
    # # (Original loop adjusted slightly so it actually uses vf_sizes[i] for the label)
    # for i in range(len(vf_sizes)):
    #     vf_names.append(f"vf_{vf_sizes[i]}_samples")

    vf_names = ["ensemble", "ensemble_big"]

    n_trials = 5
    n_repeat = 1
    perception_noise = 0.0
    calc_novf = True

    max_screwdriver_tilt = 0.015
    screwdriver_noise_mag = 0.015
    finger_noise_mag = 0.25

    pregrasp_iters = 80
    regrasp_iters = 50
    turn_iters = 100

    vf_weight_rg = 12
    other_weight_rg = 8
    variance_ratio_rg = 2

    vf_weight_t = 12
    other_weight_t = 8
    variance_ratio_t = 2

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    pregrasp_path = fpath / 'test' / 'initializations' / 'compare_sizes_pregrasps.pkl'

    # Create initial pregrasp poses if missing or out of date
    if (not pregrasp_path.exists()) or (len(pkl.load(open(pregrasp_path, 'rb'))) != n_trials):
        print("Generating new pregrasp initializations...")
        get_initializations(env, config, chain, config['sim_device'], n_trials,
                            max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag,
                            save=True, do_pregrasp=True, name='compare_sizes_pregrasps')

    pregrasps = pkl.load(open(pregrasp_path, 'rb'))

    # Build a list of method names. If calc_novf is True, add "no_vf"
    method_names = []
    if calc_novf:
        method_names.append("no_vf")
    # Then add each VF method name
    method_names.extend(vf_names)

    # Load/create checkpoint that can store results for all these methods
    checkpoint = load_or_create_checkpoint(checkpoint_path=checkpoint_path, method_names=method_names)

    pregrasp_indices = list(range(n_trials))
    repeat_indices = list(range(n_repeat))
    args = [pregrasp_indices, repeat_indices]

    for pregrasp_index, repeat_index in product(*args):
        combo_tuple = (pregrasp_index, repeat_index)

        # Skip if we already tested this combination
        if combo_tuple in checkpoint['tested_combinations']:
            continue

        print(f"Testing combination {combo_tuple}")

        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{n_repeat*pregrasp_index+repeat_index+1}')
        img_save_dir.mkdir(parents=True, exist_ok=True)
        env.frame_fpath = img_save_dir
        env.frame_id = 0

        pregrasp_pose = pregrasps[pregrasp_index]

        # --------------------------------------------------------------------
        # 1) "no_vf" approach (only if calc_novf=True)
        # --------------------------------------------------------------------
        if calc_novf:
            method_key = "no_vf"
            # Only run if not stored yet
            if (combo_tuple not in checkpoint['results'][method_key]):
                env.reset(dof_pos=pregrasp_pose)

                regrasp_pose_novf, regrasp_traj_novf = regrasp(
                    env, config, chain, state2ee_pos_partial, perception_noise=0,
                    image_path=img_save_dir, initialization=pregrasp_pose,
                    mode='no_vf', iters=regrasp_iters
                )

                _, turn_pose_novf, succ_novf, turn_traj_novf = do_turn(
                    regrasp_pose_novf, config, env,
                    sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                    perception_noise=0, image_path=img_save_dir, iters=turn_iters,
                    mode='no_vf'  
                )

                result_novf = [
                    pregrasp_pose, regrasp_pose_novf, regrasp_traj_novf,
                    turn_pose_novf, turn_traj_novf
                ]
                # Store to dict
                checkpoint['results'][method_key][combo_tuple] = result_novf
                save_checkpoint(checkpoint, checkpoint_path)

        # --------------------------------------------------------------------
        # 2) Each VF approach
        # --------------------------------------------------------------------
        for vf_index, vf_name in enumerate(vf_names):
            # If not yet stored for this method
            if (combo_tuple not in checkpoint['results'][vf_name]):
                env.reset(dof_pos=pregrasp_pose)

                regrasp_pose_vf, regrasp_traj_vf = regrasp(
                    env, config, chain, state2ee_pos_partial, perception_noise=0,
                    image_path=img_save_dir, initialization=pregrasp_pose,
                    mode='vf', model_name=vf_name, iters=regrasp_iters,
                    vf_weight=vf_weight_rg, other_weight=other_weight_rg, variance_ratio=variance_ratio_rg
                )

                # (Currently using mode='no_vf' for turn, as in original)
                _, turn_pose_vf, succ_vf, turn_traj_vf = do_turn(
                    regrasp_pose_vf, config, env,
                    sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                    perception_noise=0, image_path=img_save_dir, iters=turn_iters,
                    mode='no_vf', model_name=vf_name,
                    vf_weight=vf_weight_t, other_weight=other_weight_t, variance_ratio=variance_ratio_t
                )

                result_vf = [
                    pregrasp_pose, regrasp_pose_vf, regrasp_traj_vf,
                    turn_pose_vf, turn_traj_vf
                ]
                # Store to dict
                checkpoint['results'][vf_name][combo_tuple] = result_vf
                save_checkpoint(checkpoint, checkpoint_path)

        # Mark this (pregrasp_index, repeat_index) done
        checkpoint['tested_combinations'].add(combo_tuple)
        save_checkpoint(checkpoint, checkpoint_path)

    # Finally save the entire results structure to a separate file
    savepath = f'{fpath.resolve()}/test/compare_sizes_{test_name}.pkl'
    pkl.dump(checkpoint['results'], open(savepath, 'wb'))

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    emailer().send()