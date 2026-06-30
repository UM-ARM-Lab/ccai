import torch
from functools import wraps
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pytorch_kinematics import transforms as tf
import time
import pickle

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


def _partial_to_visualization_full_state(
    partial,
    fingers,
    full_dof_reference=None,
    joint_index=None,
    controlled_joint_index=None,
):
    if full_dof_reference is None and joint_index is None:
        return partial_to_full_state(partial, fingers)
    if full_dof_reference is None or joint_index is None:
        raise ValueError("full_dof_reference and joint_index must be provided together.")

    active_joint_index = controlled_joint_index
    if active_joint_index is None:
        active_joint_index = sum([list(joint_index[finger]) for finger in fingers], [])
    expected_dof = len(active_joint_index)
    if partial.shape[-1] != expected_dof:
        raise ValueError(
            f"Expected {expected_dof} controlled DOFs for visualization, got {partial.shape[-1]}."
        )

    reference = torch.as_tensor(full_dof_reference, device=partial.device, dtype=partial.dtype).reshape(-1)
    active_joint_index = torch.as_tensor(active_joint_index, device=partial.device, dtype=torch.long)
    scatter_index = active_joint_index.expand(partial.shape[:-1] + (active_joint_index.numel(),))
    full_shape = partial.shape[:-1] + (reference.numel(),)
    return reference.expand(full_shape).scatter(dim=-1, index=scatter_index, src=partial)


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
CAMERA_PRESET_FILENAMES = {
    "screwdriver": "ScreenCamera_2024-10-02-14-35-33.json",
    "card": "ScreenCamera_card.json",
}


def _get_camera_parameters(task, camera_parameters_path=None):
    if camera_parameters_path is not None:
        camera_path = pathlib.Path(camera_parameters_path).expanduser()
    else:
        try:
            preset_name = CAMERA_PRESET_FILENAMES[task]
        except KeyError as exc:
            raise ValueError(f"Unsupported visualization task: {task}") from exc
        camera_path = pathlib.Path(__file__).resolve().with_name(preset_name)
    if not camera_path.exists():
        raise FileNotFoundError(f"Missing camera parameters for task '{task}': {camera_path}")
    return o3d.io.read_pinhole_camera_parameters(str(camera_path))


def _write_window_camera_parameters(vis, camera_parameters_path):
    camera_path = pathlib.Path(camera_parameters_path).expanduser()
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(str(camera_path), parameters)
    print(f"Saved Open3D camera parameters to {camera_path}")
    return False


def _register_camera_save_callback(vis, save_camera_parameters_path):
    if save_camera_parameters_path is None:
        return

    def save_callback(callback_vis):
        return _write_window_camera_parameters(callback_vis, save_camera_parameters_path)

    vis.register_key_callback(ord("S"), save_callback)


def _collect_visualization_geometry(
    trajectory_step,
    scene,
    fingers,
    obj_dof,
    pcd=None,
    full_dof_reference=None,
    joint_index=None,
    controlled_joint_index=None,
):
    robot_dof = len(controlled_joint_index) if controlled_joint_index is not None else 4 * len(fingers)
    q = trajectory_step[:robot_dof]
    theta = trajectory_step[robot_dof: robot_dof + obj_dof]
    full_q = _partial_to_visualization_full_state(
        q.unsqueeze(0),
        fingers,
        full_dof_reference=full_dof_reference,
        joint_index=joint_index,
        controlled_joint_index=controlled_joint_index,
    )
    rob_mesh, meshes = scene.get_visualization_meshes(
        full_q.to(device=scene.device),
        theta.unsqueeze(0).to(device=scene.device),
        pcd=pcd,
    )
    return meshes + rob_mesh


def _make_offscreen_material(geometry):
    material = o3d.visualization.rendering.MaterialRecord()
    if isinstance(geometry, o3d.geometry.PointCloud):
        material.shader = "defaultUnlit"
        material.point_size = 5.0
    else:
        material.shader = "defaultLit"
    return material


def _compute_auto_camera_from_geometries(geometries):
    bounds = []
    for geometry in geometries:
        if not hasattr(geometry, "get_axis_aligned_bounding_box"):
            continue
        bbox = geometry.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.get_min_bound(), dtype=np.float64)
        max_bound = np.asarray(bbox.get_max_bound(), dtype=np.float64)
        if min_bound.shape != (3,) or max_bound.shape != (3,):
            continue
        if not (np.isfinite(min_bound).all() and np.isfinite(max_bound).all()):
            continue
        bounds.append((min_bound, max_bound))

    if not bounds:
        return None

    combined_min = np.min(np.stack([bound[0] for bound in bounds], axis=0), axis=0)
    combined_max = np.max(np.stack([bound[1] for bound in bounds], axis=0), axis=0)
    center = 0.5 * (combined_min + combined_max)
    extent = np.maximum(combined_max - combined_min, 1e-3)
    view_direction = np.array([0.65, -0.75, 0.45], dtype=np.float64)
    view_direction = view_direction / np.linalg.norm(view_direction)
    distance = max(float(np.max(extent)) * 2.75, 0.35)
    eye = center + view_direction * distance
    return {
        "center": center,
        "eye": eye,
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "extent": extent,
    }


def _apply_auto_window_camera(view_control, geometries):
    camera = _compute_auto_camera_from_geometries(geometries)
    if camera is None:
        return False
    view_control.set_lookat(camera["center"])
    front = camera["center"] - camera["eye"]
    front = front / np.linalg.norm(front)
    view_control.set_front(front)
    view_control.set_up(camera["up"])
    view_control.set_zoom(0.65)
    return True


def _apply_auto_offscreen_camera(renderer, geometries):
    camera = _compute_auto_camera_from_geometries(geometries)
    if camera is None:
        return False
    renderer.setup_camera(
        60.0,
        camera["center"].astype(np.float32),
        camera["eye"].astype(np.float32),
        camera["up"].astype(np.float32),
    )
    return True


def _visualize_trajectory_window(
    trajectory,
    scene,
    scene_path,
    fingers,
    obj_dof,
    headless=False,
    task='screwdriver',
    pcd=None,
    full_dof_reference=None,
    joint_index=None,
    controlled_joint_index=None,
    camera_mode="preset",
    camera_parameters_path=None,
    save_camera_parameters_path=None,
    camera_setup_only=False,
):
    parameters = _get_camera_parameters(task, camera_parameters_path=camera_parameters_path)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=600, visible=not headless)
    _register_camera_save_callback(vis, save_camera_parameters_path)
    vis.get_render_option().mesh_show_wireframe = True
    vis.get_render_option().point_show_normal = True
    vis.get_render_option().background_color = np.ones(3)

    for t in range(trajectory.shape[0]):
        vis.clear_geometries()
        meshes = _collect_visualization_geometry(
            trajectory[t],
            scene,
            fingers,
            obj_dof,
            pcd=pcd,
            full_dof_reference=full_dof_reference,
            joint_index=joint_index,
            controlled_joint_index=controlled_joint_index,
        )
        for mesh in meshes:
            vis.add_geometry(mesh)

        ctr = vis.get_view_control()
        if camera_mode == "auto":
            if not _apply_auto_window_camera(ctr, meshes):
                ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        elif camera_mode == "preset":
            ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        else:
            raise ValueError(f"Unsupported camera mode: {camera_mode}")
        vis.poll_events()
        vis.update_renderer()
        if camera_setup_only:
            if save_camera_parameters_path is not None:
                print(
                    "Adjust the Open3D view, press S to save the camera, "
                    "then close the viewer."
                )
            else:
                print("Adjust the Open3D view, then close the viewer.")
            vis.run()
            vis.destroy_window()
            return False
        img = vis.capture_screen_float_buffer(False)
        plt.imsave(scene_path / 'img' / f'im_{t:04d}.png', np.asarray(img), dpi=1)

    vis.destroy_window()
    return True


def _visualize_trajectory_offscreen(
    trajectory,
    scene,
    scene_path,
    fingers,
    obj_dof,
    task='screwdriver',
    pcd=None,
    full_dof_reference=None,
    joint_index=None,
    controlled_joint_index=None,
    camera_mode="preset",
    camera_parameters_path=None,
):
    parameters = _get_camera_parameters(task, camera_parameters_path=camera_parameters_path)
    renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

    for t in range(trajectory.shape[0]):
        renderer.scene.clear_geometry()
        geometries = _collect_visualization_geometry(
            trajectory[t],
            scene,
            fingers,
            obj_dof,
            pcd=pcd,
            full_dof_reference=full_dof_reference,
            joint_index=joint_index,
            controlled_joint_index=controlled_joint_index,
        )
        for idx, geometry in enumerate(geometries):
            renderer.scene.add_geometry(
                f"geometry_{idx}",
                geometry,
                _make_offscreen_material(geometry),
            )

        if camera_mode == "auto":
            if not _apply_auto_offscreen_camera(renderer, geometries):
                renderer.setup_camera(parameters.intrinsic, parameters.extrinsic)
        elif camera_mode == "preset":
            renderer.setup_camera(parameters.intrinsic, parameters.extrinsic)
        else:
            raise ValueError(f"Unsupported camera mode: {camera_mode}")
        img = renderer.render_to_image()
        o3d.io.write_image(str(scene_path / 'img' / f'im_{t:04d}.png'), img)
    return True


def visualize_trajectory(
    trajectory,
    scene,
    scene_fpath,
    fingers,
    obj_dof,
    headless=False,
    task='screwdriver',
    pcd=None,
    render_backend='window',
    full_dof_reference=None,
    joint_index=None,
    controlled_joint_index=None,
    camera_mode="preset",
    camera_parameters_path=None,
    save_camera_parameters_path=None,
    camera_setup_only=False,
):
    scene_path = pathlib.Path(scene_fpath)
    with open(scene_path / 'traj.pkl', 'wb') as f:
        pickle.dump(trajectory.cpu().numpy(), f)

    if render_backend == 'offscreen':
        rendered_frames = _visualize_trajectory_offscreen(
            trajectory,
            scene,
            scene_path,
            fingers,
            obj_dof,
            task=task,
            pcd=pcd,
            full_dof_reference=full_dof_reference,
            joint_index=joint_index,
            controlled_joint_index=controlled_joint_index,
            camera_mode=camera_mode,
            camera_parameters_path=camera_parameters_path,
        )
    elif render_backend == 'window':
        rendered_frames = _visualize_trajectory_window(
            trajectory,
            scene,
            scene_path,
            fingers,
            obj_dof,
            headless=headless,
            task=task,
            pcd=pcd,
            full_dof_reference=full_dof_reference,
            joint_index=joint_index,
            controlled_joint_index=controlled_joint_index,
            camera_mode=camera_mode,
            camera_parameters_path=camera_parameters_path,
            save_camera_parameters_path=save_camera_parameters_path,
            camera_setup_only=camera_setup_only,
        )
    else:
        raise ValueError(f"Unsupported render backend: {render_backend}")

    if not rendered_frames:
        return

    # convert to GIF
    import subprocess
    output_dir = scene_path / 'gif' / 'trajectory.gif'
    cmd = f"ffmpeg -y -i {scene_path}/img/im_%4d.png -vf palettegen ~/palette.png"
    subprocess.call(cmd, shell=True)
    cmd = f"ffmpeg -y -framerate 2 -i {scene_path}/img/im_%4d.png -i ~/palette.png " \
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

def vector_cos(a, b):
    """Compute cosine similarity between two vectors."""
    return torch.dot(a.reshape(-1), b.reshape(-1)) / (torch.norm(a.reshape(-1)) * torch.norm(b.reshape(-1)))

def euler_to_quat(euler):
    """Convert Euler angles to quaternion."""
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat

def euler_to_angular_velocity(current_euler, next_euler):
    """Convert Euler angles to angular velocity."""
    current_quat = euler_to_quat(current_euler)
    next_quat = euler_to_quat(next_euler)
    dquat = next_quat - current_quat
    con_quat = - current_quat  # conjugate
    con_quat[..., 0] = current_quat[..., 0]
    omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:]
    return omega

def dec2bin(x, bits):
    """Convert decimal to binary representation."""
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits):
    """Convert binary to decimal representation."""
    mask = 2 ** torch.arange(bits).to(b.device, b.dtype)
    return torch.sum(b * mask, dim=-1)

def extract_state_vector(state, num_fingers, device, obj_dof=3, slice_end=None, hardcoded_dim=None):
    """
    Helper function to extract and process state vector from environment state.
    
    Args:
        state: Environment state dictionary
        num_fingers: Number of fingers 
        device: Target device for the tensor
        obj_dof: Object degrees of freedom (default: 3)
        slice_end: Optional end index for slicing (e.g., 15)
        hardcoded_dim: Optional hardcoded dimension instead of calculating from num_fingers
    
    Returns:
        Processed state tensor
    """
    if hardcoded_dim is not None:
        state_tensor = state['q'].reshape(-1, hardcoded_dim)
    else:
        # Calculate dimension based on fingers and object DOF
        if obj_dof == 3:
            dim = 4 * num_fingers + 4  # Most common case
        else:
            dim = 4 * num_fingers + obj_dof
        state_tensor = state['q'].reshape(-1, dim)
    
    # Extract first element
    state_tensor = state_tensor[0]
    
    # Apply slicing if specified
    if slice_end is not None:
        state_tensor = state_tensor[:slice_end]
    
    # Move to device
    return state_tensor.to(device=device)
