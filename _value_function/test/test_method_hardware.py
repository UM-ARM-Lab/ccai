import numpy as np
import pickle as pkl

import pathlib
import shutil
from functools import partial
import yaml
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(CCAI_PATH))
from tqdm import tqdm
from pathlib import Path
from _value_function.screwdriver_problem import init_env, do_turn, pregrasp, regrasp, emailer, convert_partial_to_full_config, delete_imgs
from _value_function.data_collect.process_final_poses_regrasp import calculate_turn_cost
import pytorch_kinematics as pk
from isaac_victor_envs.utils import get_assets_dir

from ccai.utils.allegro_utils import state2ee_pos
from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv

import torch
fpath = pathlib.Path(f'{CCAI_PATH}/data')
from itertools import product
import warnings
warnings.filterwarnings("ignore",message="You are using `torch.load` with `weights_only=False`",category=FutureWarning)
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

if __name__ == '__main__':

    perception_noise = 0.0
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

    # config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True, config_path=config_path)

    pregrasp_path = fpath /'test'/'official_initializations'/'test_method_pregrasps.pkl'
    diffusion_path = 'data/training/allegro_screwdriver/adam_diffusion/allegro_screwdriver_diffusion_4999.pt'
   
    print("input method:")
    method = input()
    # method = "vf"
    print("input trial number:")
    trial_number = int(input())
    # trial_number = 0

    pregrasps = pkl.load(open(pregrasp_path, 'rb'))
    pregrasp_pose = pregrasps[trial_number]

    ########################################################################################################################
    # HARDWARE ENVIRONMENT SETUP ###########################################################################################
    ########################################################################################################################

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{config_path}').read_text())

    from hardware.hardware_env import HardwareEnv
    default_dof_pos = pregrasp_pose[:,:16]
    env = HardwareEnv(default_dof_pos[:, :16], 
                        finger_list=config['fingers'], 
                        kp=config['kp'], 
                        obj='screwdriver',
                        mode='relative',
                        gradual_control=True,
                        num_repeat=10)
    env.initial_dof_pos = default_dof_pos
    env.reset()

    env.get_state()
    for _ in range(5):
        root_coor, root_ori = env.obj_reader.get_state_world_frame_pos()
    print('Root coor:', root_coor)
    print('Root ori:', root_ori)
    root_coor = root_coor
    robot_p = np.array([0, -0.095, 1.33])
    root_coor = root_coor + robot_p

    img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/imgs_official/hardware/{method}/trial_{trial_number}')
    pathlib.Path.mkdir(img_save_dir, parents=True, exist_ok=True)  

    sim_env = RosAllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=2.5,
                                device=config['sim_device'],
                                valve=config['object_type'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                table_pose=None,
                                gravity=False
                                )
    sim_env.frame_fpath = img_save_dir
    sim_env.frame_id = 0
    
    sim_env.reset(dof_pos= pregrasp_pose)

    print("Enter when done setting up screwdriver")
    input()
    
    sim, gym, viewer = sim_env.get_sim()
    assert (np.array(sim_env.robot_p) == robot_p).all()
    assert (sim_env.initial_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
    env.world_trans = sim_env.world_trans
    env.joint_stiffness = sim_env.joint_stiffness
    env.device = sim_env.device
    env.table_pose = sim_env.table_pose

    ros_copy_node = None
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    config['ee_names'] = ee_names
    config['obj_dof_code'] = [0, 0, 0, 1, 1, 1]
    config['obj_dof'] = np.sum(config['obj_dof_code'])
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos_partial = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)

    ########################################################################################################################
    ########################################################################################################################

    print(f"Testing {method} trial {trial_number} ...")


    if method == 'vf':
        env.initial_dof_pos = pregrasp_pose
        env.reset()
    
        regrasp_pose_vf, regrasp_traj_vf, regrasp_plan = regrasp(
                env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
                image_path=img_save_dir, initialization=pregrasp_pose, mode='vf', iters=regrasp_iters, model_name = "ensemble_rg",
                vf_weight=vf_weight_rg, other_weight=other_weight_rg, variance_ratio=variance_ratio_rg,
                sim_viz_env=sim_env
        )
    
        _, turn_pose_vf, succ_vf, turn_traj_vf, turn_plan = do_turn(
            regrasp_pose_vf, config, env,
            sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
            perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='vf',
            model_name="ensemble_t", initial_yaw = regrasp_pose_vf[0, -2],
            vf_weight=vf_weight_t, other_weight=other_weight_t, variance_ratio=variance_ratio_t,
            sim_viz_env=sim_env
        )

        # Store the VF approach result
        result = [pregrasp_pose, regrasp_pose_vf, regrasp_traj_vf, turn_pose_vf, turn_traj_vf]
        turn_cost = calculate_turn_cost(regrasp_pose_vf.numpy(), turn_pose_vf)
        print('---------------------------------')
        print(f"VF cost: {turn_cost}")
        print('---------------------------------')

    elif method == 'diffusion_no_contact_cost':

        env.initial_dof_pos = pregrasp_pose
        env.reset()
        
        regrasp_pose_diffusion, regrasp_traj_diffusion, regrasp_plan = regrasp(
            env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
            use_diffusion=True, use_contact_cost=False,
            diffusion_path = diffusion_path,
            image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            sim_viz_env=sim_env
        )
    
        _, turn_pose_diffusion, succ_diffusion, turn_traj_diffusion, turn_plan = do_turn(
            regrasp_pose_diffusion, config, env,
            sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
            use_diffusion=True,
            diffusion_path = diffusion_path,
            perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf',
            sim_viz_env=sim_env
        )
        
        result = [pregrasp_pose, regrasp_pose_diffusion, regrasp_traj_diffusion, turn_pose_diffusion, turn_traj_diffusion]
        turn_cost = calculate_turn_cost(regrasp_pose_diffusion.numpy(), turn_pose_diffusion)
        print('---------------------------------')
        print(f"Diffusion no contact cost: {turn_cost}")
        print('---------------------------------')

    elif method == 'diffusion_w_contact_cost':

        env.initial_dof_pos = pregrasp_pose
        env.reset()
        
        regrasp_pose_diffusion_wc, regrasp_traj_diffusion_wc, regrasp_plan = regrasp(
            env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
            use_diffusion=True, use_contact_cost=True,
            diffusion_path = diffusion_path,
            image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            sim_viz_env=sim_env
        )
    
        _, turn_pose_diffusion_wc, succ_diffusion_wc, turn_traj_diffusion_wc, turn_plan = do_turn(
            regrasp_pose_diffusion_wc, config, env,
            sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
            use_diffusion=True, 
            diffusion_path = diffusion_path,
            perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf',
            sim_viz_env=sim_env
        )
        
        result = [pregrasp_pose, regrasp_pose_diffusion_wc, regrasp_traj_diffusion_wc, turn_pose_diffusion_wc, turn_traj_diffusion_wc]
        turn_cost = calculate_turn_cost(regrasp_pose_diffusion_wc.numpy(), turn_pose_diffusion_wc)
        print('---------------------------------')
        print(f"Diffusion with contact cost: {turn_cost}")
        print('---------------------------------')

    elif method == 'combined':

        env.initial_dof_pos = pregrasp_pose
        env.reset()
        
        regrasp_pose_combined, regrasp_traj_combined, regrasp_plan = regrasp(
            env, config, chain, state2ee_pos_partial, perception_noise=perception_noise,
            use_diffusion=True,
            diffusion_path = diffusion_path,
            image_path=img_save_dir, initialization=pregrasp_pose, mode='vf', iters=regrasp_iters,
            sim_viz_env=sim_env
        )
    
        _, turn_pose_combined, succ_combined, turn_traj_combined, turn_plan = do_turn(
            regrasp_pose_combined, config, env,
            sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
            use_diffusion=True, 
            diffusion_path = diffusion_path,
            initial_yaw = regrasp_pose_combined[0, -2],
            perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='vf',
            sim_viz_env=sim_env
        )
        
        result = [pregrasp_pose, regrasp_pose_combined, regrasp_traj_combined, turn_pose_combined, turn_traj_combined]
        turn_cost = calculate_turn_cost(regrasp_pose_combined.numpy(), turn_pose_combined)
        print('---------------------------------')
        print(f"Combined method cost: {turn_cost}")
        print('---------------------------------')

    elif method == 'novf':
        env.initial_dof_pos = pregrasp_pose
        env.reset()
        
        regrasp_pose_novf, regrasp_traj_novf, regrasp_plan = regrasp(
            env, config, chain, state2ee_pos_partial, perception_noise=perception_noise, use_diffusion = False,
            image_path=img_save_dir, initialization=pregrasp_pose, mode='no_vf', iters=regrasp_iters,
            sim_viz_env=sim_env
        )
    
        _, turn_pose_novf, succ_novf, turn_traj_novf, turn_plan = do_turn(
            regrasp_pose_novf, config, env,
            sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial,
            perception_noise=perception_noise, image_path=img_save_dir, iters=turn_iters,mode='no_vf', use_diffusion = False,
            sim_viz_env=sim_env
        )
        
        result = [pregrasp_pose, regrasp_pose_novf, regrasp_traj_novf, turn_pose_novf, turn_traj_novf]

        turn_cost = calculate_turn_cost(regrasp_pose_novf.numpy(), turn_pose_novf)
        print('---------------------------------')
        print(f"No VF cost: {turn_cost}")
        print('---------------------------------')

savepath = f'{fpath.resolve()}/test/hardware_results/{method}_trial_{trial_number}.pkl'
pkl.dump(result, open(savepath, 'wb'))

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
