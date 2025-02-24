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
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost

import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
from itertools import product
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)
if len(sys.argv) == 2:
    config_path = f'allegro_screwdriver_adam{sys.argv[1]}.yaml'
else:
    config_path = 'allegro_screwdriver_adam0.yaml'

def get_initialization(env, sim_device, max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag):

    while True:

        index_noise_mag = torch.tensor([finger_noise_mag]*4)
        index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
        middle_thumb_noise_mag = torch.tensor([finger_noise_mag]*4)
        middle_thumb_noise = middle_thumb_noise_mag * (2 * torch.rand(4) - 1)
        screwdriver_noise = torch.tensor([
            np.random.normal(0, screwdriver_noise_mag),  
            np.random.normal(0, screwdriver_noise_mag),  
            np.random.uniform(0, 2 * np.pi),
            0.0  
        ])
        #fingers=['index', 'middle', 'ring', 'thumb']
        initialization = torch.cat((
            torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + index_noise,
            torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + middle_thumb_noise,
            torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=sim_device),
            torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=sim_device) + middle_thumb_noise,
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float().to(device=sim_device) + screwdriver_noise
        ), dim=1).to(sim_device)



       
        env.reset(dof_pos= initialization)
        for _ in range(64):
            env._step_sim()
        solved_initialization = env.get_state()['q'].reshape(1,16)[:,0:-1].to(device=sim_device)
        solved_initialization = convert_partial_to_full_config(solved_initialization)

        sd = solved_initialization[0, -4:-1]

        if abs(sd[0]) < max_screwdriver_tilt and abs(sd[1]) < max_screwdriver_tilt:
            return solved_initialization

def get_initializations(env, config, chain, sim_device, n_samples, 
                        max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, 
                        save = False, do_pregrasp=False, name = "unnamed"):
    
    savepath = f'{fpath.resolve()}/test/initializations/{name}.pkl'
   
    initializations = []
    for _ in range(n_samples):
        initialization = get_initialization(env, sim_device, max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag)
        initializations.append(initialization)

    if do_pregrasp == False:
        if save == True:
            pkl.dump(initializations, open(savepath, 'wb'))
        return initializations
   
    else:
        pregrasps = []
        for init in initializations:
            pregrasp_pose, _ = pregrasp(
                    env, config, chain, deterministic=True, perception_noise=0,
                    image_path=None, initialization=init, mode='no_vf',
                    iters=80
                )
            pregrasps.append(pregrasp_pose)
        if save == True:
            pkl.dump(pregrasps, open(savepath, 'wb'))
        return pregrasps

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

def save_checkpoint(checkpoint):

    with open(checkpoint_path, 'wb') as f:
        pkl.dump(checkpoint, f)

if __name__ == '__main__':

    test_name = 'compare_epochs'
    checkpoint_path = fpath /'test'/'test_method'/f'checkpoint_{test_name}.pkl'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = 20
    n_repeat = 1
    perception_noise = 0.0

    calc_vf = False
    calc_diffusion_no_contact_cost = True
    calc_diffusion2 = True
    calc_diffusion_w_contact_cost = False
    calc_novf = False
    calc_combined = True

    method_names = []
    if calc_vf:
        method_names.append("vf")
    if calc_novf:
        method_names.append("no_vf")
    if calc_diffusion_no_contact_cost:
        method_names.append("diffusion_no_contact_cost")
    if calc_diffusion_w_contact_cost:
        method_names.append("diffusion_w_contact_cost")
    if calc_combined:
        method_names.append("combined")
    if calc_diffusion2:
        method_names.append("diffusion2")

    max_screwdriver_tilt = 0.015
    screwdriver_noise_mag = 0.015
    finger_noise_mag = 0.05

    regrasp_iters = 40
    turn_iters = 50

    vf_weight_rg = 5.0
    other_weight_rg = 1.9
    variance_ratio_rg = 10.0

    vf_weight_t = 3.3
    other_weight_t = 1.9
    variance_ratio_t = 1.625

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True, config_path=config_path)

    pregrasp_path = fpath /'test'/'initializations'/'test_method_pregrasps.pkl'
    diffusion_path = 'data/training/allegro_screwdriver/adam_diffusion/allegro_screwdriver_diffusion_4999.pt'
    diffusion_path2 = 'data/training/allegro_screwdriver/adam_diffusion/allegro_screwdriver_diffusion_9999.pt'

    if pregrasp_path.exists() == False or len(pkl.load(open(pregrasp_path, 'rb'))) != n_trials:
        print("Generating new pregrasp initializations...")
        get_initializations(env, config, chain, config['sim_device'], n_trials,
                            max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, save=True,
                            do_pregrasp=True, name='test_method_pregrasps')
   
    pregrasps = pkl.load(open(pregrasp_path, 'rb'))
   
    pregrasp_indices = list(range(n_trials))
    repeat_indices = list(range(n_repeat))
    args = [pregrasp_indices, repeat_indices]

    checkpoint = load_or_create_checkpoint(checkpoint_path=checkpoint_path, method_names=method_names)

    for pregrasp_index, repeat_index in product(*args):
        combo_tuple = (pregrasp_index, repeat_index)

        if combo_tuple in checkpoint['tested_combinations']:
            continue

        print(f"Testing combination {combo_tuple}")

        img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{n_repeat*pregrasp_index+repeat_index+1}')
        pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
        env.frame_fpath = img_save_dir
        env.frame_id = 0
       
        pregrasp_pose = pregrasps[pregrasp_index]

        if calc_vf:
            env.reset(dof_pos= pregrasp_pose)
        
            regrasp_pose_vf, regrasp_traj_vf, regrasp_plan, rg_initial_samples = regrasp(
                    env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                    image_path=img_save_dir, initialization=pregrasp_pose, mode='vf', iters=regrasp_iters, model_name = "ensemble_rg",
                    vf_weight=vf_weight_rg, other_weight=other_weight_rg, variance_ratio=variance_ratio_rg
            )
        
            _, turn_pose_vf, succ_vf, turn_traj_vf, turn_plan, t_initial_samples = do_turn(
                regrasp_pose_vf, config, env,
                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='vf',
                model_name="ensemble_t", initial_yaw = regrasp_pose_vf[0, -2],
                vf_weight=vf_weight_t, other_weight=other_weight_t, variance_ratio=variance_ratio_t
            )

            # Store the VF approach result
            result_vf = [pregrasp_pose, regrasp_pose_vf, regrasp_traj_vf, turn_pose_vf, turn_traj_vf]
            checkpoint['results']['vf'][combo_tuple] = result_vf

            turn_cost = calculate_turn_cost(regrasp_pose_vf.numpy(), turn_pose_vf)
            print('---------------------------------')
            print(f"VF cost: {turn_cost}")

        if calc_diffusion_no_contact_cost:

            env.reset(dof_pos= pregrasp_pose)
           
            regrasp_pose_diffusion, regrasp_traj_diffusion, regrasp_plan, rg_initial_samples = regrasp(
                env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                use_diffusion=True, use_contact_cost=False,
                diffusion_path = diffusion_path,
                image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            )
       
            _, turn_pose_diffusion, succ_diffusion, turn_traj_diffusion, turn_plan, t_initial_samples = do_turn(
                regrasp_pose_diffusion, config, env,
                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                use_diffusion=True,
                diffusion_path = diffusion_path,
                perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf',
            )
           
            result_diffusion = [pregrasp_pose, regrasp_pose_diffusion, regrasp_traj_diffusion, turn_pose_diffusion, turn_traj_diffusion, initial_samples]
            checkpoint['results']['diffusion_no_contact_cost'][combo_tuple] = result_diffusion
            turn_cost = calculate_turn_cost(regrasp_pose_diffusion.numpy(), turn_pose_diffusion)
            print('---------------------------------')
            print(f"Diffusion no contact cost: {turn_cost}")

        if calc_diffusion2:

            env.reset(dof_pos= pregrasp_pose)
           
            regrasp_pose_diffusion2, regrasp_traj_diffusion2, regrasp_plan, rg_initial_samples = regrasp(
                env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                use_diffusion=True, use_contact_cost=False,
                diffusion_path = diffusion_path2,
                image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            )
       
            _, turn_pose_diffusion2, succ_diffusion2, turn_traj_diffusion2, turn_plan, t_initial_samples = do_turn(
                regrasp_pose_diffusion2, config, env,
                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                use_diffusion=True,
                diffusion_path = diffusion_path2,
                perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf',
            )
           
            result_diffusion2 = [pregrasp_pose, regrasp_pose_diffusion2, regrasp_traj_diffusion2, turn_pose_diffusion2, turn_traj_diffusion2, initial_samples]
            checkpoint['results']['diffusion2'][combo_tuple] = result_diffusion2
            turn_cost = calculate_turn_cost(regrasp_pose_diffusion2.numpy(), turn_pose_diffusion2)
            print('---------------------------------')
            print(f"Diffusion 2 cost: {turn_cost}")

        if calc_diffusion_w_contact_cost:

            env.reset(dof_pos= pregrasp_pose)
           
            regrasp_pose_diffusion_wc, regrasp_traj_diffusion_wc, regrasp_plan, rg_initial_samples = regrasp(
                env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                use_diffusion=True, use_contact_cost=True,
                diffusion_path = diffusion_path,
                image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            )
       
            _, turn_pose_diffusion_wc, succ_diffusion_wc, turn_traj_diffusion_wc, turn_plan, t_initial_samples = do_turn(
                regrasp_pose_diffusion_wc, config, env,
                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                use_diffusion=True, 
                diffusion_path = diffusion_path,
                perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf',
            )
           
            result_diffusion_wc = [pregrasp_pose, regrasp_pose_diffusion_wc, regrasp_traj_diffusion_wc, turn_pose_diffusion_wc, turn_traj_diffusion_wc, initial_samples]
            checkpoint['results']['diffusion_w_contact_cost'][combo_tuple] = result_diffusion_wc
            turn_cost = calculate_turn_cost(regrasp_pose_diffusion_wc.numpy(), turn_pose_diffusion_wc)
            print('---------------------------------')
            print(f"Diffusion with contact cost: {turn_cost}")

        if calc_combined:

            env.reset(dof_pos= pregrasp_pose)
           
            regrasp_pose_combined, regrasp_traj_combined, regrasp_plan, rg_initial_samples = regrasp(
                env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                use_diffusion=True, use_contact_cost=False,
                diffusion_path = diffusion_path,
                image_path=img_save_dir, initialization=pregrasp_pose, mode='vf', iters=regrasp_iters,
            )
       
            _, turn_pose_combined, succ_combined, turn_traj_combined, turn_plan, t_initial_samples = do_turn(
                regrasp_pose_combined, config, env,
                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                use_diffusion=True, 
                diffusion_path = diffusion_path,
                initial_yaw = regrasp_pose_combined[0, -2],
                perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='vf',
            )
           
            result_combined = [pregrasp_pose, regrasp_pose_combined, regrasp_traj_combined, turn_pose_combined, turn_traj_combined, initial_samples]
            checkpoint['results']['combined'][combo_tuple] = result_combined
            turn_cost = calculate_turn_cost(regrasp_pose_combined.numpy(), turn_pose_combined)
            print('---------------------------------')
            print(f"Combined method cost: {turn_cost}")

        if calc_novf:

            env.reset(dof_pos= pregrasp_pose)
           
            regrasp_pose_novf, regrasp_traj_novf, regrasp_plan, rg_initial_samples = regrasp(
                env, config, chain, state2ee_pos_partial, perception_noise=perception_noise, use_diffusion = False,
                image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            )
       
            _, turn_pose_novf, succ_novf, turn_traj_novf, turn_plan, t_initial_samples = do_turn(
                regrasp_pose_novf, config, env,
                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
                perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf', use_diffusion = False,
            )
           
            result_novf = [pregrasp_pose, regrasp_pose_novf, regrasp_traj_novf, turn_pose_novf, turn_traj_novf]
            checkpoint['results']['no_vf'][combo_tuple] = result_novf

            turn_cost = calculate_turn_cost(regrasp_pose_novf.numpy(), turn_pose_novf)
            print('---------------------------------')
            print(f"No VF cost: {turn_cost}")
            
        checkpoint['tested_combinations'].add(combo_tuple)
        save_checkpoint(checkpoint)
        savepath = f'{fpath.resolve()}/test/test_method_{test_name}.pkl'
        pkl.dump(checkpoint['results'], open(savepath, 'wb'))

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    emailer().send()