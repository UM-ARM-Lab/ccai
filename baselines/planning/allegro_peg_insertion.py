from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroPegInsertionEnv

import numpy as np
import pickle as pkl

import torch
import time
import copy
import yaml
import pathlib
from functools import partial

import time
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

from utils.allegro_utils import *
from examples.allegro_valve_roll import AllegroContactProblem, PositionControlConstrainedSVGDMPC
from scipy.spatial.transform import Rotation as R
from naive_planner import NaivePlanner


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# device = 'cuda:0'
obj_dof = 6
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')



def do_trial(env, params, fpath):
    peg_goal = params['object_goal'].cpu()
    peg_goal_pos = peg_goal[:3]
    peg_goal_mat = R.from_euler('XYZ', peg_goal[-3:]).as_matrix()
    # step multiple times untile it's stable
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    for i in range(1):
        if len(params['fingers']) == 3:
            action = torch.cat((env.default_dof_pos[:,:8], env.default_dof_pos[:, 12:16]), dim=-1)
        elif len(params['fingers']) == 4:
            action = env.default_dof_pos[:, :16]
        state = env.step(action)

    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []

    start = state[0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
  
    pregrasp_problem = AllegroContactProblem(
        dx=4 * num_fingers,
        du=4 * num_fingers,
        start=start[:4 * num_fingers + obj_dof],
        goal=None,
        T=4,
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.peg_pose,
        object_type='peg',
        world_trans=env.world_trans,
        fingers=params['fingers'],
        obj_dof_code=params['obj_dof_code'],
        obj_joint_dim=0,
        fixed_obj=True
    )
    pregrasp_params = params.copy()
    pregrasp_params['N'] = 10
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)
    pregrasp_planner.warmup_iters = 50
    
    
    start = env.get_state()['q'][0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])


    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        action = action.repeat(params['N'], 1)
        env.step(action)
        action_list.append(action)
    state = env.get_state()
    start = state[0].reshape(1, 4 * num_fingers + obj_dof).to(device=params['device'])

    # get the end effector point in the peg frame
    fk_dict = forward_kinematics(partial_to_full_state(start[:, :12], fingers=params['fingers']))
    fks = [fk_dict[finger] for finger in fk_dict.keys()]
    fk_world = [env.world_trans.compose(fk) for fk in fks]
    peg_state = start[:, -obj_dof:]
    peg_quat = R.from_euler('XYZ', peg_state[0, 3:].detach().cpu().numpy()).as_quat()
    peg_trans = tf.Transform3d(pos=torch.tensor(peg_state[0, :3], device=params['device']).float(),
                                rot=torch.tensor(
                                    [peg_quat[3], peg_quat[0], peg_quat[1], peg_quat[2]],
                                    device=params['device']).float(), device=params['device'])
    fk_peg_frame = [peg_trans.inverse().compose(fk) for fk in fk_world]

    planner = NaivePlanner(chain=params['chain'], fingers=params['fingers'], ee_peg_frame=fk_peg_frame, T=params['T'], world2robot=env.world_trans, goal=params['object_goal'],
                          obj_dof_code=[1, 1, 1, 1, 1, 1], device=params['device'])

    actual_trajectory = []
    duration = 0

    num_fingers_to_plan = num_fingers
    info_list = []
    contact_list = []
    validity_flag = True
    with torch.no_grad():
        for k in range(params['num_steps']):
            state = env.get_state()
            start = state[0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])

            actual_trajectory.append(state[0, :4 * num_fingers + obj_dof].clone())
            start_time = time.time()

            action = planner.step(start[:4 * num_fingers + obj_dof]) # this call will modify the environment
            action = action.unsqueeze(0)
            state = env.step(action)

            if k < params['num_steps'] - 1:
                planner.T = planner.T - 1

            print(f"solve time: {time.time() - start_time}")
            print(f"current theta: {state[0, -obj_dof:].detach().cpu().numpy()}")
            duration += time.time() - start_time

            contacts = gym.get_env_rigid_contacts(env.envs[0])
            for body0, body1 in zip (contacts['body0'], contacts['body1']):
                if body0 == 31 and body1 == 33:
                    print("contact with wall")
                    contact_list.append(True)
                    break
                elif body0 == 33 and body1 == 31:
                    print("contact with wall")
                    contact_list.append(True)
                    break

            peg_state = env.get_state()['q'][:, -obj_dof:].cpu()
            peg_mat = R.from_euler('XYZ', peg_state[:, -3:]).as_matrix()
            distance2goal_ori = tf.so3_relative_angle(torch.tensor(peg_mat), \
            torch.tensor(peg_goal_mat).unsqueeze(0), cos_angle=False).detach().cpu().abs()
            distance2goal_pos = (peg_state[:, :3] - peg_goal_pos.unsqueeze(0)).norm(dim=-1).detach().cpu()
            
            print(distance2goal_pos, distance2goal_ori)
            if not check_peg_validity(peg_state[0]):
                validity_flag = False
            info = {'distance2goal_pos': distance2goal_pos, 'distance2goal_ori': distance2goal_ori, 
            'contact': contact_list,
            'validity_flag': validity_flag}
            info_list.append(info)
            # add trajectory lines to sim

            
            action_list.append(action)

    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)



    state = env.get_state()
    state = state[0].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal_pos = distance2goal_pos
    final_distance_to_goal_ori = distance2goal_ori
    contact_rate = np.array(contact_list).mean()

    print(f'Controller: {params["controller"]} Final distance to goal pos: {final_distance_to_goal_pos.item()}, ori: {final_distance_to_goal_pos.item()}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"])}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            d2goal_pos=final_distance_to_goal_pos.item(),
            d2goal_ori=final_distance_to_goal_ori.item())
    env.reset()
    ret = {'final_distance_to_goal_pos': final_distance_to_goal_pos.item(), 
    'final_distance_to_goal_ori': final_distance_to_goal_ori.item(), 
    'contact_rate': contact_rate,
    'validity_flag': validity_flag,
    'avg_online_time': duration / (params["num_steps"])}
    return ret

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/planning/config/allegro_peg_insertion.yaml').read_text())
    from tqdm import tqdm

    env = AllegroPegInsertionEnv(config['controllers']['planning']['N'], control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                )

    sim, gym, viewer = env.get_sim()


    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass


    results = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'hitosashi_ee',
            'middle': 'naka_ee',
            'ring': 'kusuri_ee',
            'thumb': 'oya_ee',
            }
    config['ee_names'] = ee_names
    config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
    config['obj_dof'] = np.sum(config['obj_dof_code'])

    # screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver_6d.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    # screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])
    for controller in config['controllers'].keys():
        results[controller] = {}
        results[controller]['dist2goal_pos'] = []
        results[controller]['dist2goal_ori'] = []
        results[controller]['contact_rate'] = []
        results[controller]['validity_flag'] = []
        results[controller]['avg_online_time'] = []

    for i in tqdm(range(config['num_trials'])):
        goal = torch.tensor([0, 0, 0, 0, 0, 0])
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            config_file_path = pathlib.PurePath.joinpath(fpath, 'config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(config, f)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['object_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.peg_pose).to(params['device']).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            ret = do_trial(env, params, fpath)
            # final_distance_to_goal = turn(env, params, fpath)

            results[controller]['dist2goal_pos'].append(ret['final_distance_to_goal_pos'])
            results[controller]['dist2goal_ori'].append(ret['final_distance_to_goal_ori'])
            results[controller]['contact_rate'].append(ret['contact_rate'])
            results[controller]['validity_flag'].append(ret['validity_flag'])
            results[controller]['avg_online_time'].append(ret['avg_online_time'])

        print(results)
        for key in results[controller].keys():
            print(f"{controller} {key}: avg: {np.array(results[controller][key]).mean()}, std: {np.array(results[controller][key]).std()}")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
