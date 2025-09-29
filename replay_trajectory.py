#!/usr/bin/env python3

import sys
import pickle
import pathlib
import argparse
import time
import torch
import numpy as np
import pathlib
# Add the parent directory to the path to import modules
sys.path.append('..')

from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv
from isaac_victor_envs.utils import get_assets_dir

import datetime

def create_env(config=None, visualize=True, device='cuda:0'):
    """Create AllegroScrewdriverTurningEnv with default parameters"""
    
    now = datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
    video_path = f'/home/abhinav/Documents/ccai/data/experiments/idto_replay_valve/{now}'
    pathlib.Path(video_path).mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    default_config = {
        'num_envs': 1,
        'control_mode': 'joint_impedance',
        'use_cartesian_controller': False,
        'viewer': visualize,
        'steps_per_action': 6,
        'friction_coefficient': 2.5,
        'device': device,
        'video_save_path': video_path,
        'joint_stiffness': 3,  # Default kp value
        'fingers': ['index', 'middle', 'thumb'],
        'gradual_control': False,
        'gravity': True,
        'randomize_obj_start': False,
        'randomize_rob_start': False,
        'external_wrench_perturb': False,
        'force_sensors': False
    }
    
    # Override with provided config if available
    # if config:
    #     default_config.update(config)
    
    # Create environment
    # env = AllegroScrewdriverTurningEnv(
    #     default_config['num_envs'], 
    #     control_mode=default_config['control_mode'],
    #     use_cartesian_controller=default_config['use_cartesian_controller'],
    #     viewer=default_config['viewer'],
    #     steps_per_action=default_config['steps_per_action'],
    #     friction_coefficient=default_config['friction_coefficient'],
    #     device=default_config['device'],
    #     video_save_path=default_config['video_save_path'],
    #     joint_stiffness=default_config['joint_stiffness'],
    #     fingers=default_config['fingers'],
    #     gradual_control=default_config['gradual_control'],
    #     gravity=default_config['gravity'],
    #     randomize_obj_start=default_config['randomize_obj_start'],
    #     randomize_rob_start=default_config['randomize_rob_start'],
    #     external_wrench_perturb=default_config['external_wrench_perturb'],
    #     force_sensors=default_config['force_sensors']
    # )
    
    env = AllegroValveTurningEnv(
        default_config['num_envs'], 
        control_mode=default_config['control_mode'],
        use_cartesian_controller=default_config['use_cartesian_controller'],
        viewer=default_config['viewer'],
        steps_per_action=default_config['steps_per_action'],
        friction_coefficient=.1,
        device=default_config['device'],
        video_save_path=default_config['video_save_path'],
        joint_stiffness=3,
        fingers=default_config['fingers'],
        gravity=default_config['gravity'],
        randomize_obj_start=False,
        randomize_rob_start=False,
    )
    
    return env

def replay_trajectory(pickle_file_path, visualize=True, step_delay=0.1, device='cuda:0', save_video=False, video_path=None, line_data_file=None, trial_index=0):
    """
    Replay a trajectory from a pickle file containing a torch tensor of shape T x dx.
    
    Args:
        pickle_file_path (str): Path to the pickle file containing the trajectory tensor
        visualize (bool): Whether to show the visualization
        step_delay (float): Delay between steps in seconds
        device (str): Device to use for computation
        save_video (bool): Whether to save video frames
        video_path (str): Path to save video frames
        line_data_file (str): Path to the pickle file containing line visualization data
        trial_index (int): Index of the trial to visualize (default: 0)
    """
    
    # Load the trajectory tensor from pickle file
    print(f"Loading trajectory from: {pickle_file_path}")
    
    trial_index = 0
    with open(pickle_file_path, 'rb') as f:
        trajectory_data = pickle.load(f)[9]
    
    # Load line visualization data if provided
    single_lines_data = None
    multiple_lines_data = None
    if line_data_file:
        print(f"Loading line visualization data from: {line_data_file}")
        with open(line_data_file, 'rb') as f:
            line_data = pickle.load(f)
            if len(line_data) >= 2 and trial_index < len(line_data[0]):
                single_lines_data = line_data[0][trial_index]  # Element 0: single lines (green)
                multiple_lines_data = line_data[1][trial_index]  # Element 1: multiple lines (blue)
                print(f"Loaded line data for trial {trial_index}: {len(single_lines_data)} single line entries, {len(multiple_lines_data)} multiple line entries")
    
    # Handle different possible formats
    if isinstance(trajectory_data, torch.Tensor):
        trajectory = trajectory_data
    elif isinstance(trajectory_data, list):
        # If it's a list of tensors, stack them
        if all(isinstance(x, torch.Tensor) for x in trajectory_data):
            trajectory = torch.stack(trajectory_data)
        else:
            # If it's a list of numpy arrays, convert to tensor
            trajectory = torch.tensor(np.array(trajectory_data))
    elif isinstance(trajectory_data, np.ndarray):
        trajectory = torch.tensor(trajectory_data)
    else:
        trajectory = torch.stack(trajectory_data['states'])[:, :13]

    print(f"Loaded trajectory with shape: {trajectory.shape}")
    
    # Ensure trajectory is on the correct device
    trajectory = trajectory.to(device)
    
    T, dx = trajectory.shape
    print(f"Trajectory has {T} timesteps and {dx} dimensions")
    
    # Create environment
    config = {
        'device': device,
        'video_save_path': video_path if save_video else None
    }
    
    env = create_env(config=config, visualize=visualize, device='cpu')
    # time.sleep(5)
    
    try:
        # Reset environment
        env.reset()
        # time.sleep(5)
        print("Environment reset successfully")
        
        # If saving video, set up frame saving
        if save_video and video_path:
            pathlib.Path(video_path).mkdir(parents=True, exist_ok=True)
            env.frame_fpath = pathlib.Path(video_path)
            env.frame_id = 0
        
        print(f"Starting trajectory replay with {step_delay}s delay between steps...")
        print("Press Ctrl+C to stop")
        
        # Loop through trajectory and set poses
        for t in range(0,T, 1):

            pose = trajectory[t]
            
            if pose.shape[0] == 15:
                pose = torch.cat((pose, torch.zeros(1)), dim=0)
            
            # Set the pose in the environment
            env.set_pose(pose)
            
            # Clear previous lines before stepping simulation
            env.gym.clear_lines(env.viewer)
                            
            env._step_sim()
            
            # Draw lines if line data is available
            if single_lines_data and len(single_lines_data) > 0:
                # Check if the first element matches current timestep
                if single_lines_data[0][0] == t:
                    last_matching_line_data = None
                    # Loop through and find the last matching timestep, discarding elements
                    while len(single_lines_data) > 0 and single_lines_data[0][0] == t:
                        timestep, line_data = single_lines_data.pop(0)  # Remove and get the first element
                        last_matching_line_data = line_data
                    
                    # Draw the last matching single line in green
                    if last_matching_line_data is not None and len(last_matching_line_data) > 0:
                        env.gym.add_lines(env.viewer, env.envs[0], 1, 
                                        last_matching_line_data,
                                        np.array([[0, 1., 0]], dtype=np.float32))
            
            if multiple_lines_data and len(multiple_lines_data) > 0:
                # Check if the first element matches current timestep
                if multiple_lines_data[0][0] == t:
                    last_matching_lines_data = None
                    # Loop through and find the last matching timestep, discarding elements
                    while len(multiple_lines_data) > 0 and multiple_lines_data[0][0] == t:
                        timestep, lines_data = multiple_lines_data.pop(0)  # Remove and get the first element
                        last_matching_lines_data = lines_data
                    
                    # Draw the last matching multiple lines in blue
                    if last_matching_lines_data is not None and len(last_matching_lines_data) > 0:
                        env.gym.add_lines(env.viewer, env.envs[0], 6, 
                                        last_matching_lines_data,
                                        np.array([[0/255, 0/255, 255/255], [0/255, 0/255, 255/255], [0/255, 0/255, 255/255]], dtype=np.float32))
            
            # Write frame if saving video
            if save_video:
                env.write_image()
            
            print(f"Step {t+1}/{T} - Set pose: {pose.cpu().numpy()[-4:]}..." + 
                    f" (showing last 4 dims)")
            
            # Add delay between steps
            if step_delay > 0:
                time.sleep(step_delay)

        
        print("Trajectory replay completed!")
        
        # Keep environment open for a moment if visualizing
        if visualize:
            print("Environment will close in 5 seconds...")
            time.sleep(5)
    
    finally:
        # Clean up
        try:
            sim, gym, viewer = env.get_sim()
            if viewer:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
        except:
            pass

def main():
    # parser = argparse.ArgumentParser(description='Replay trajectory from pickle file in AllegroScrewdriverTurningEnv')
    # parser.add_argument('pickle_file', type=str, help='Path to pickle file containing trajectory tensor')
    # parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    # parser.add_argument('--step-delay', type=float, default=0.1, help='Delay between steps in seconds (default: 0.1)')
    # parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')
    # parser.add_argument('--save-video', action='store_true', help='Save video frames')
    # parser.add_argument('--video-path', type=str, default='./replay_video', help='Path to save video frames')
    
    # args = parser.parse_args()
    
    # Check if pickle file exists
    # if not pathlib.Path(args.pickle_file).exists():
    #     print(f"Error: Pickle file '{args.pickle_file}' not found")
    #     return
    
    # Run trajectory replay
    replay_trajectory(
        pickle_file_path='state_trajectories.pkl',
        # visualize=not args.no_visualize,
        # step_delay=args.step_delay,
        # device=args.device,
        # save_video=args.save_video,
        # video_path=args.video_path
        visualize=True,
        step_delay=0.0,
        device='cpu',
        save_video=False,
        video_path=None,
        line_data_file=None,#'allegro_perturbations_screwdriver_hora_new.pkl',
        trial_index=0
    )

if __name__ == "__main__":
    main()
