from scipy.spatial.transform import Rotation as R
import numpy as np

def quaternion_to_intrinsic_xyz_euler(q):
    """
    Convert quaternion (w, x, y, z) to intrinsic XYZ Euler angles using scipy.
    
    Args:
        q: A tuple or list (w, x, y, z)
    
    Returns:
        A tuple (roll, pitch, yaw) in radians
    """
    q = q.cpu().detach().numpy()
    w, x, y, z = q

    # scipy expects quaternions in (x, y, z, w) order
    quat_xyzw = [x, y, z, w]
    
    # Create rotation object
    rotation = R.from_quat(quat_xyzw)
    
    # Get intrinsic XYZ Euler angles
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    return tuple(euler_angles)

from scipy.spatial.transform import Rotation as R

def transform_robot_to_object_frame(q_robot_world, q_object_world):
    """
    Given robot and object quaternions in world frame (wxyz),
    compute the robot quaternion in object frame (still wxyz output).
    """
    # Rearrange from (w, x, y, z) to (x, y, z, w) for scipy
    q_robot_world_xyzw = [q_robot_world[1], q_robot_world[2], q_robot_world[3], q_robot_world[0]]
    q_object_world_xyzw = [q_object_world[1], q_object_world[2], q_object_world[3], q_object_world[0]]
    
    # Create rotation objects
    rot_robot_world = R.from_quat(q_robot_world_xyzw)
    rot_object_world = R.from_quat(q_object_world_xyzw)
    
    # Invert object rotation
    rot_world_object = rot_object_world.inv()
    
    # Compute robot rotation in object frame
    rot_robot_object = rot_world_object * rot_robot_world
    # rot_robot_object = rot_robot_world * rot_object_world.inv()

    # Convert back to quaternion (x, y, z, w)
    q_robot_object_xyzw = rot_robot_object.as_quat()
    
    return q_robot_object_xyzw

def transform_pose_to_object_aligned_frame(hand_pos_world, hand_quat_world, object_pos_world, object_quat_world):
    """
    Transform hand and object poses into a frame aligned with the object's orientation.

    Args:
        hand_pos_world: np.ndarray of shape (3,), hand position in world frame.
        hand_quat_world: np.ndarray of shape (4,), hand orientation [w, x, y, z] in world frame.
        object_pos_world: np.ndarray of shape (3,), object position in world frame.
        object_quat_world: np.ndarray of shape (4,), object orientation [w, x, y, z] in world frame.

    Returns:
        hand_pos_new: np.ndarray of shape (3,), hand position in new frame.
        hand_quat_new: np.ndarray of shape (4,), hand quaternion [w, x, y, z] in new frame.
        object_pos_new: np.ndarray of shape (3,), object position in new frame.
        object_quat_new: np.ndarray of shape (4,), object quaternion [w, x, y, z] in new frame.
    """
    # Convert quaternions from [w, x, y, z] to [x, y, z, w] for scipy
    r_hand_world = R.from_quat([hand_quat_world[1], hand_quat_world[2], hand_quat_world[3], hand_quat_world[0]])
    r_object_world = R.from_quat([object_quat_world[1], object_quat_world[2], object_quat_world[3], object_quat_world[0]])

    # Inverse of object rotation
    r_world_to_object = r_object_world.inv()

    # Transform positions (no subtraction — just rotate both)
    hand_pos_new = r_world_to_object.apply(hand_pos_world)
    object_pos_new = r_world_to_object.apply(object_pos_world)

    # Transform orientations
    r_hand_new = r_world_to_object * r_hand_world
    r_object_new = r_world_to_object * r_object_world

    hand_quat_new = r_hand_new.as_quat()  # [x, y, z, w]

    object_quat_new = r_object_new.as_quat()  # [x, y, z, w]

    return hand_pos_new, hand_quat_new, object_pos_new, object_quat_new
if __name__ == "__main__":
    q_robot_world = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    q_object_world = np.array([0.7071, 0.0, 0.7071, 0.0])  # [w, x, y, z]
    # q_object_world = np.array([1, 0, 0, 0])  # [w, x, y, z]

    q_robot_in_object = transform_robot_to_object_frame(q_robot_world, q_object_world)

    print("Robot quaternion in object frame:", q_robot_in_object)