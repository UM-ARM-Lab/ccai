import torch
from functools import wraps
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pytorch_kinematics.transforms as tf

full_finger_list = ['index', 'middle', 'ring', 'thumb']


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
    g_list, grad_g_list, hess_g_list = [], [], []
    for finger in fingers:
        g, grad_g, hess_g = func(self, finger_name=finger,
                                 compute_grads=compute_grads, compute_hess=compute_hess, **kwargs)
        g_list.append(g)
        grad_g_list.append(grad_g)
        hess_g_list.append(hess_g)
    g = torch.cat(g_list, dim=1)
    if compute_grads:
        grad_g = torch.cat(grad_g_list, dim=1)
    else:
        return g, None, None

    if compute_hess:
        hess_g = torch.cat(hess_g_list, dim=1)
        return g, grad_g, hess_g

    return g, grad_g, None


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


def visualize_trajectory(trajectory, scene, scene_fpath, fingers, obj_dof, headless=False, task='screwdriver'):
    num_fingers = len(fingers)
    # for a single trajectory
    T, dxu = trajectory.shape
    # set up visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(800), height=int(600), visible=not headless)
    # update camera
    vis.get_render_option().mesh_show_wireframe = True
    for t in range(T):
        vis.clear_geometries()
        q = trajectory[t, : 4 * num_fingers]
        theta = trajectory[t, 4 * num_fingers: 4 * num_fingers + obj_dof]
        # scene.visualize_robot(partial_to_full_state(q.unsqueeze(0), fingers).to(device=scene.device),
        #                      theta.unsqueeze(0).to(device=scene.device))
        meshes = scene.get_visualization_meshes(partial_to_full_state(q.unsqueeze(0), fingers).to(device=scene.device),
                                                theta.unsqueeze(0).to(device=scene.device))
        for mesh in meshes:
            vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        if task == 'screwdriver':
            import os
            #Get current working directory
            cwd = os.getcwd()

            parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-08-07-10-49-00.json")
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
