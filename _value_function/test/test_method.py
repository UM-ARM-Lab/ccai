import numpy as np
import pickle as pkl
import pathlib
import shutil
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, emailer, convert_partial_to_full_config, delete_imgs
import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def get_initialization(max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag):

        while True:

            config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)
            sim_device = config['sim_device']

            # torch.random.manual_seed(1)
            index_noise_mag = torch.tensor([finger_noise_mag]*4)
            index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
            middle_thumb_noise_mag = torch.tensor([finger_noise_mag]*4)
            middle_thumb_noise = middle_thumb_noise_mag * (2 * torch.rand(4) - 1)
            screwdriver_noise = torch.tensor([
            np.random.uniform(-screwdriver_noise_mag, screwdriver_noise_mag),  # Random value between -0.05 and 0.05
            np.random.uniform(-screwdriver_noise_mag, screwdriver_noise_mag),  # Random value between -0.05 and 0.05
            np.random.uniform(0, 2 * np.pi),  # Random value between 0 and 2π
            0.0  
            ])
            #fingers=['index', 'middle', 'ring', 'thumb']
            initialization = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + index_noise,
                                        torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + middle_thumb_noise,
                                        torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=sim_device),
                                        torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=sim_device) + middle_thumb_noise,
                                        torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float().to(device=sim_device) + screwdriver_noise),
                                        dim=1).to(sim_device)
            
            env.reset(dof_pos= initialization)
            for _ in range(64):
                env._step_sim()
            solved_initialization = env.get_state()['q'].reshape(1,16)[:,0:-1].to(device=sim_device)
            solved_initialization = convert_partial_to_full_config(solved_initialization)

            sd = solved_initialization[0, -4:-1]

            gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            del env, sim_env, viewer
            torch.cuda.empty_cache()

            if abs(sd[0]) < max_screwdriver_tilt and abs(sd[1]) < max_screwdriver_tilt:
                return solved_initialization
        

def get_initializations(n_samples, max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag, save = False):
    
    initializations = []
    for _ in range(n_samples):
        initialization = get_initialization(max_screwdriver_tilt, screwdriver_noise_mag, finger_noise_mag)
        initializations.append(initialization)

    pkl.dump(initializations, open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'wb'))
    return initializations


if __name__ == '__main__':

    test_name = 'multi_step'
    n_trials = 5
    n_repeat = 1
    perception_noise = 0.0
    calc_novf = True

    tuples = []
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    # initializations = get_initializations(config['sim_device'], env, n_trials, 
    #                 max_screwdriver_tilt=0.03, screwdriver_noise_mag=0.03, finger_noise_mag=0.3)
    # pkl.dump(initializations, open('/home/newuser/Desktop/Honda/ccai/_value_function/test/gravity_test/inits.pkl', 'wb'))

    # exit()
    initializations = pkl.load(open('/home/newuser/Desktop/Honda/ccai/_value_function/test/gravity_test/inits.pkl', 'rb'))

    delete_imgs()

    for i in tqdm(range(n_trials)):
        for j in tqdm(range(n_repeat)):

            # img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{i}')
            img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs/trial_{i+1}')
            pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  
            env.frame_fpath = img_save_dir
            env.frame_id = 0
            pregrasp_iters = 200

            # Our method

            pregrasp_pose_vf, planned_pregrasp_pose = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise, 
                            image_path = img_save_dir, initialization = initializations[i], 
                            model_name = 'ensemble', mode='vf', iters = pregrasp_iters,
                            vf_weight = 10.0, other_weight = 0.1, variance_ratio = 5)
            
            _, turn_pose_vf, turn_succ, turn_trajectory_vf = do_turn(pregrasp_pose_vf, config, env, 
                            sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                            perception_noise=perception_noise, image_path = img_save_dir)
            
            # no vf

            if calc_novf:
                pregrasp_pose_no_vf, no_vf_plan = pregrasp(env, config, chain, deterministic=True, perception_noise=perception_noise, 
                                image_path = img_save_dir, initialization = initializations[i], mode='no_vf', iters = pregrasp_iters)
                
                _, turn_pose_no_vf, turn_succ, turn_trajectory_no_vf = do_turn(pregrasp_pose_no_vf, config, env, 
                                sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial, 
                                perception_noise=perception_noise, image_path = img_save_dir)
                
                tuples.append((initializations[i], 
                               pregrasp_pose_vf, planned_pregrasp_pose, turn_pose_vf, turn_trajectory_vf,
                               pregrasp_pose_no_vf, no_vf_plan, turn_pose_no_vf, turn_trajectory_no_vf))

            else:
                tuples.append((initializations[i], pregrasp_pose_vf, planned_pregrasp_pose, turn_pose_vf, turn_trajectory_vf))
            


    savepath = f'{fpath.resolve()}/test/test_method_{test_name}.pkl'
    pkl.dump(tuples, open(savepath, 'wb'))

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    # emailer().send()