from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv, AllegroPegTurningEnv, AllegroPegAlignmentEnv, AllegroReorientationEnv
import torch
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
                                gradual_control=config['gradual_control'],
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
                                    gradual_control=config['gradual_control'],
                                     )
    elif task == 'reorientation':
        env = AllegroReorientationEnv(num_envs=num_envs,
                                control_mode='joint_impedance',
                                use_cartesian_controller=False,
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=2.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                gradual_control=config['gradual_control'],
                                )
    return env

def get_screwdriver_setup(config_dict, z_trans):
    ret = {}
    ret['body_r'] = config_dict['handle_radius']
    ret['body_h'] = config_dict['handle_height']
    ret['body_link_z'] = z_trans
    ret['stick_r'] = config_dict['shaft_radius']
    ret['stick_h'] = config_dict['shaft_height']
    ret['stick_link_z'] = config_dict['shaft_height'] / 2.0
    ret['cap_joint_z'] = z_trans + config_dict['handle_height'] / 2.0
    ret['cap_r'] = config_dict['handle_radius']
    return ret

def swap_joint_angle(q):
    # tensor([[-0.4854,  0.5381,  0.1845, -1.7260, -1.1532, -1.6625, -1.3796,  0.1000,
    #       0.0000, -0.1000,  1.1000,  0.5000,  0.5000,  0.4000,  0.3000,  0.6000,
    #       0.6500,  0.9000,  0.3000,  0.6000,  0.6500,  0.9000,  1.1000]],
    index_0, ring_0, middle_0, thumb_0, index_1, ring_1, middle_1, thumb_1, index_2, ring_2, middle_2, thumb_2, index_3, ring_3, middle_3, thumb_3 = q
    return torch.tensor([index_0, index_1, index_2, index_3,
                        middle_0, middle_1, middle_2, middle_3,
                        ring_0, ring_1, ring_2, ring_3,
                        thumb_0, thumb_1, thumb_2, thumb_3]).float().to(q.device)