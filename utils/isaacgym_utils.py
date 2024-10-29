from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv, AllegroPegTurningEnv, AllegroPegAlignmentEnv

def get_env(task, img_save_dir, config, num_envs=1):
    if task == 'screwdriver_turning':
        env = AllegroScrewdriverTurningEnv(num_envs=num_envs, 
                                           control_mode='joint_impedance',
                                            use_cartesian_controller=False,
                                            viewer=True,
                                            steps_per_action=60,
                                            friction_coefficient=1.0,
                                            device=config['sim_device'],
                                            video_save_path=img_save_dir,
                                            joint_stiffness=config['kp'],
                                            fingers=config['fingers'],
                                            gradual_control=config['gradual_control'],
                                            arm_type=config['arm_type'],
                                            gravity=config['gravity'],
                                            )
    elif task == 'valve_turning':
        env = AllegroValveTurningEnv(num_envs=num_envs, 
                                    control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                valve_type=config['object_type'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                random_robot_pose=config['random_robot_pose'],
                                )
    elif task == 'peg_turning':
        env = AllegroPegTurningEnv(num_envs=num_envs,
                                control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                )
    elif task == 'peg_alignment':
        env = AllegroPegAlignmentEnv(num_envs=num_envs,
                                     control_mode='joint_impedance',
                                     use_cartesian_controller=False,
                                     viewer=True,
                                     steps_per_action=60,
                                     friction_coefficient=1.0,
                                     device=config['sim_device'],
                                     video_save_path=img_save_dir,
                                     joint_stiffness=config['kp'],
                                     fingers=config['fingers'],
                                     gravity=config['gravity'],
                                     )
    return env