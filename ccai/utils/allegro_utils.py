import torch
from functools import wraps
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pytorch_kinematics.transforms as tf
import time

full_finger_list = ['index', 'middle', 'ring', 'thumb']

def get_model_input_state(state, env, obj_dof):
    all_idx = []
    for finger in ['index', 'middle', 'thumb']:
        this_finger_idx = env.finger_to_joint_index[finger]
        all_idx += this_finger_idx
    finger_states = state[..., all_idx]
    obj_state = state[..., 16:16 + obj_dof]
    return torch.cat((finger_states, obj_state), dim=-1)

def get_arm_dof(arm_type):
    if arm_type == 'robot':
        arm_dof = 7
    elif arm_type == 'floating_3d':
        arm_dof = 3
    elif arm_type == 'floating_6d':
        arm_dof = 6
    elif arm_type == 'None':
        arm_dof = 0
    else:
        raise ValueError('Invalid arm type')
    return arm_dof

def partial_to_full_state(partial, fingers):
    """
    fingers: which fingers are in the partial state
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    num_fingers = len(fingers)
    partial_fingers = torch.chunk(partial, chunks=num_fingers, dim=-1)
    partial_dict = dict(zip(fingers, partial_fingers))
    full = []
    for i, finger in enumerate(full_finger_list):
        if finger in fingers:
            full.append(partial_dict[finger])
        if finger not in fingers:
            full.append(torch.zeros_like(partial_fingers[0]))
    full = torch.cat(full, dim=-1)
    return full


def full_to_partial_state(full, fingers):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, mid, ring, thumb = torch.chunk(full, chunks=4, dim=-1)
    full_dict = dict(zip(full_finger_list, [index, mid, ring, thumb]))
    partial = []
    for finger in fingers:
        partial.append(full_dict[finger])
    partial = torch.cat(partial, dim=-1)
    return partial


def finger_constraint_wrapper(self, *args, **kwargs):
    #xu = kwargs.pop('xu', None)
    #if xu is None:
    #    xu = args[0]
    fingers = kwargs.pop('fingers', None)
    if fingers is None:
        raise ValueError("fingers must be specified")
    func = kwargs.pop('func', None)
    if func is None:
        raise ValueError("func must be specified")

    compute_grads = kwargs.pop('compute_grads', True)
    compute_hess = kwargs.pop('compute_hess', False)
    # compute contact constraints for index finger
    g_list, grad_g_list, hess_g_list, t_mask_list = [], [], [], []
    for finger in fingers:
        g, grad_g, hess_g, t_mask = func(self, finger_name=finger,
                                 compute_grads=compute_grads, compute_hess=compute_hess, **kwargs)
        g_list.append(g)
        grad_g_list.append(grad_g)
        hess_g_list.append(hess_g)
        t_mask_list.append(t_mask)
    g = torch.cat(g_list, dim=1)
    t_mask = torch.cat(t_mask_list, dim=1)
    if compute_grads:
        grad_g = torch.cat(grad_g_list, dim=1)
    else:
        return g, None, None, t_mask

    if compute_hess:
        hess_g = torch.cat(hess_g_list, dim=1)
        return g, grad_g, hess_g, t_mask

    return g, grad_g, None, t_mask


def all_finger_constraints(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        fingers = self.fingers
        return finger_constraint_wrapper(self, fingers=fingers, func=func, *args, **kwargs)

    return wrapper


def regrasp_finger_constraints(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        fingers = self.regrasp_fingers
        return finger_constraint_wrapper(self, fingers=fingers, func=func, *args, **kwargs)

    return wrapper


def contact_finger_constraints(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        fingers = self.contact_fingers
        return finger_constraint_wrapper(self, fingers=fingers, func=func, *args, **kwargs)

    return wrapper


def state2ee_pos(state, finger_name, fingers, chain, frame_indices, world_trans):
    """
    :params state: B x 8 joint configuration for full hand
    :return ee_pos: B x 3 position of ee

    """
    fk_dict = chain.forward_kinematics(partial_to_full_state(state.to(device=chain.device), fingers), frame_indices=frame_indices)
    m = world_trans.compose(fk_dict[finger_name].to(world_trans.device))
    points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=m.device).unsqueeze(0)
    ee_p = m.transform_points(points_finger_frame).squeeze(-2)
    return ee_p


is_visible = False
def visualize_trajectory(trajectory, scene, scene_fpath, fingers, obj_dof, headless=False, task='screwdriver', pcd=None):
    num_fingers = len(fingers)
    # for a single trajectory
    T, dxu = trajectory.shape
    # set up visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=int(800), height=int(600), visible=not headless)
    # update camera
    vis.get_render_option().mesh_show_wireframe = True
    vis.get_render_option().point_show_normal = True
    for t in range(T):
        vis.clear_geometries()
        q = trajectory[t, : 4 * num_fingers]
        theta = trajectory[t, 4 * num_fingers: 4 * num_fingers + obj_dof]
        # scene.visualize_robot(partial_to_full_state(q.unsqueeze(0), fingers).to(device=scene.device),
        #                      theta.unsqueeze(0).to(device=scene.device))
        rob_mesh, meshes = scene.get_visualization_meshes(partial_to_full_state(q.unsqueeze(0), fingers).to(device=scene.device),
                                                theta.unsqueeze(0).to(device=scene.device), pcd=pcd)
        # o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True, width=800, height=600)
        meshes += rob_mesh
        for mesh in meshes:
            vis.add_geometry(mesh)
        # vis.add_geometry(rob_mesh)
        
        # def toggle_visibility(vis):
        #     global is_visible
        #     if is_visible:
        #         for m in rob_mesh:
        #             vis.remove_geometry(m, reset_bounding_box=False)
        #     else:
        #         for m in rob_mesh:
        #             vis.add_geometry(m, reset_bounding_box=False)
            
        #     # Update visibility status
        #     is_visible = not is_visible
        #     vis.update_renderer()


        # vis.register_key_callback(ord('T'), toggle_visibility)
        # vis.run()

        ctr = vis.get_view_control()
        if task == 'screwdriver':
            import os
            #Get current working directory
            cwd = os.getcwd()

            parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-10-02-14-35-33.json")
            # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-08-07-10-49-00.json")
        elif task == 'card':
            parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_card.json")
        ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(False)
        plt.imsave(f'{scene_fpath}/img/im_{t:04d}.png',
                   np.asarray(img),
                   dpi=1)

    vis.destroy_window()

    # convert to GIF
    import subprocess
    output_dir = f'{scene_fpath}/gif/trajectory.gif'
    cmd = f"ffmpeg -y -i {scene_fpath}/img/im_%4d.png -vf palettegen ~/palette.png"
    subprocess.call(cmd, shell=True)
    cmd = f"ffmpeg -y -framerate 2 -i {scene_fpath}/img/im_%4d.png -i ~/palette.png " \
          f"-lavfi paletteuse {output_dir}"
    subprocess.call(cmd, shell=True)


def visualize_trajectories(trajectories, scene, fpath, headless=False):
    for n, trajectory in enumerate(trajectories):
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/gif'), parents=True, exist_ok=True)
        visualize_trajectory(trajectory, scene, f'{fpath}/trajectory_{n + 1}/kin', headless=headless)
        # Visualize what happens if we execute the actions in the trajectory in the simulator
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/gif'), parents=True, exist_ok=True)
        # visualize_trajectory_in_sim(trajectory, config['env'], f'{fpath}/trajectory_{n + 1}/sim')
        # save the trajectory
        np.save(f'{fpath}/trajectory_{n + 1}/traj.npz', trajectory.cpu().numpy())


def axis_angle_to_euler(axis_angle):
    matrix = tf.axis_angle_to_matrix(axis_angle)
    euler = tf.matrix_to_euler_angles(matrix, convention='XYZ')
    return euler

def get_screwdriver_top_in_world(env_q, object_chain, world2robot_trans, object_asset_pos):
    """
    env_q: 1 dimension without batch
    """
    env_q = torch.cat((env_q, torch.zeros(1, device=env_q.device)), dim=-1) # add the screwdriver cap dim
    screwdriver_top_obj_frame = object_chain.forward_kinematics(env_q.unsqueeze(0).to(object_chain.device))['screwdriver_cap']
    screwdriver_top_obj_frame = screwdriver_top_obj_frame.get_matrix().reshape(4, 4)[:3, 3]
    world2obj_trans = tf.Transform3d(pos=torch.tensor(object_asset_pos, device=object_chain.device).float(),
                                        rot=torch.tensor([1, 0, 0, 0], device=object_chain.device).float(), device=object_chain.device)
    screwdriver_top_world_frame = world2obj_trans.transform_points(screwdriver_top_obj_frame.unsqueeze(0)).squeeze(0)
    return screwdriver_top_world_frame


def euler_diff(euler1, euler2, representation='XYZ'):
    """
    :params euler1: B x 3
    :params euler2: B x 3
    :return diff: B x 1
    compute the difference between two orientations represented by euler angles
    """
    ori1_mat = R.from_euler(representation, euler1).as_matrix()
    ori2_mat = R.from_euler(representation, euler2).as_matrix()
    diff = tf.so3_relative_angle(torch.tensor(ori1_mat), torch.tensor(ori2_mat), cos_angle=False).detach().cpu()
    return diff

def convert_yaw_to_sine_cosine(xu, yaw_idx=14):
    """
    xu is shape (N, T, 36)
    Replace the yaw in xu with sine and cosine and return the new xu
    """
    yawp1 = yaw_idx + 1
    yaw = xu[..., yaw_idx]
    sine = torch.sin(yaw)
    cosine = torch.cos(yaw)
    xu_new = torch.cat([xu[..., :yaw_idx], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[..., yawp1:]], dim=-1)
    return xu_new

def convert_sine_cosine_to_yaw(xu, yaw_idx=14):
    """
    xu is shape (N, T, 37)
    Replace the sine and cosine in xu with yaw and return the new xu
    """
    yawp1 = yaw_idx + 1
    yawp2 = yaw_idx + 2
    orig_type = torch.is_tensor(xu)
    if not orig_type:
        xu = torch.tensor(xu)
    sine = xu[..., yawp1]
    cosine = xu[..., yaw_idx]
    yaw = torch.atan2(sine, cosine)
    xu_new = torch.cat([xu[..., :yaw_idx], yaw.unsqueeze(-1), xu[..., yawp2:]], dim=-1)
    if not orig_type:
        xu_new = xu_new.numpy()
    return xu_new
