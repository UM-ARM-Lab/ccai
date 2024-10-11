from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
import numpy as np
import pickle as pkl
import torch
import time
import copy
import yaml
import pathlib
from functools import partial
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
import matplotlib.pyplot as plt
from utils.allegro_utils import *
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from examples.allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from examples.allegro_screwdriver import ALlegroScrewdriverContact
from scipy.spatial.transform import Rotation as R
from baselines.planning.ik import IKSolver
from tqdm import tqdm

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class emailer():
    def __init__(self):
        self.sender_email = "eburner813@gmail.com"
        self.receiver_email = "adamhung@umich.edu"  # You can send it to yourself
        self.password = "yhpffhhnwbhpluty"
    def send(self, *args, **kwargs):
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = "program finished"
        body = "program finished"
        msg.attach(MIMEText(body, 'plain'))
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(self.sender_email, self.password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text)
            print("Email sent")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            server.quit()


if __name__ == "__main__":
    fingers=['index', 'middle', 'thumb']

    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())

    params = config.copy()
    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                #video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gradual_control=True,
                                randomize_obj_start=True,
                                gravity=False
                                )

    sim, gym, viewer = env.get_sim()
    state = env.get_state()
    device = config['sim_device']
    params['device'] = device

    # dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float().to(device=device),
    #                     torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float().to(device=device),
    #                     torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=device),
    #                     torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float().to(device=device)),
    #                     dim=1).to(device)
    # dof_pos = torch.cat((dof_pos, torch.zeros((1, 4)).float().to(device=device)),
    #                                      dim=1).to(device)
    # dof_pos = dof_pos.repeat(1, 1)
    # partial_default_dof_pos = np.concatenate((dof_pos[:, 0:8], dof_pos[:, 12:16]), axis=1)

    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
    params['ee_names'] = ee_names
    params['obj_dof_code'] = [0, 0, 0, 1, 1, 1]
    params['obj_dof'] = np.sum(params['obj_dof_code'])

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    
    chain = pk.build_chain_from_urdf(open(asset).read())
    params['chain'] = chain.to(device=device)
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in fingers]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=fingers, chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    object_location = torch.tensor(env.table_pose).to(params['device']).float() # TODO: confirm if this is the correct location
    params['object_location'] = object_location
    params.update(config['controllers']['csvgd'])

    initial_poses = []
    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    obj_dof = 3
    num_fingers = len(params['fingers'])


    default_dof_pos = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=device),
                                torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=device),
                                torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=device),
                                torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=device),
                                torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float().to(device=device)),
                                dim=1).to(device)
    
    try:
        for i in tqdm(range(10000)):

            #print("iteration: ", i)
            env.reset(dof_pos= default_dof_pos, deterministic=False)
            start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)

            screwdriver = start.clone()[-4:-1]
            #print("start screwdriver: ", screwdriver)
            screwdriver = torch.cat((screwdriver, torch.tensor([0])),dim=0).reshape(1,4)

            if 'index' in params['fingers']:
                contact_fingers = params['fingers']
            else:
                contact_fingers = ['index'] + params['fingers']    
            pregrasp_problem = ALlegroScrewdriverContact(
                dx=4 * num_fingers,
                du=4 * num_fingers,
                start=start[:4 * num_fingers + obj_dof],
                goal=None,
                T=2,
                chain=params['chain'],
                device=device,
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                fingers=contact_fingers,
                obj_dof_code=params['obj_dof_code'],
                obj_joint_dim=1,
                fixed_obj=True,
            )
            pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
            pregrasp_planner.warmup_iters = 80#500 #50
            # 200 -> 20 seconds / grasp

            #start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=device)
            best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])

            # traj_for_viz = best_traj[:, :pregrasp_problem.dx]
            # tmp = start[4 * num_fingers:].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
            # traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)    
            # viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
            # img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            # gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            # pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            # pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            # visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath, 
            #                      pregrasp_problem.fingers, pregrasp_problem.obj_dof+1,
            #                      #points = goal_poses.cpu().numpy(),
            #                      )

            #for x in best_traj[:, :4 * num_fingers]:
            x = best_traj[-1, :4 * num_fingers]
            action = x.reshape(-1, 4 * num_fingers).to(device=env.device) 
            solved_pos = torch.cat((
                    action.clone()[:, :8], 
                    torch.tensor([[0., 0.5, 0.65, 0.65]]).to(device=device), 
                    action.clone()[:, 8:], 
                    screwdriver.to(device=device)
                    #torch.zeros(1,4).to(device=device)
                    ), dim=1).to(device)
            #print("solution screwdriver: ", solved_pos[:, 16:16+2])
            env.reset(dof_pos = solved_pos, deterministic=True)
            initial_poses.append(solved_pos.cpu())
            #env.gym.write_viewer_image_to_file(env.viewer, f'{fpath.resolve()}/initial_pose_frames.pkl/frame_{i}.png')
            #time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted, not saving")
        exit()
        pass


    with open(f'{fpath.resolve()}/initial_poses.pkl', 'wb') as f:
        pkl.dump(initial_poses, f)
    # with open(f'{fpath.resolve()}/screwdriver_poses.pkl', 'wb') as f:
        #  pkl.dump(screwdriver_poses, f)
    print("Saved poses")
    emailer().send()
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)