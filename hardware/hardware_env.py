import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
# from .allegro_ros import RosNode
import rospy
from shapely.geometry import Polygon
import pathlib
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
import torch
import yaml
from lightweight_vicon_bridge.msg import MocapState
from tf.transformations import euler_from_quaternion

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class ObjectPoseReader:
    def __init__(self, obj='valve', mode='relative') -> None:
        rospy.init_node('object_pose_reader')
        self.mode = mode
        self.obj = obj

        self.mocap_sub = rospy.Subscriber('/mocap_tracking', MocapState, self.mocap_callback)

    def euler_trans_from_segment(self, segment):
        transform = segment.transform
        obj_quat = (transform.rotation)
        # obj_euler = np.array(euler_from_quaternion([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w], axes='rxyz'))
        obj_euler = np.array(euler_from_quaternion([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w], axes='rxyz'))
        obj_trans = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        return obj_euler, obj_trans

    def mocap_callback(self, data):
        self.arm_base = [i for i in data.tracked_objects if i.name == 'right_arm_base'][0]
        self.mocap_obj = [i for i in data.tracked_objects if i.name == self.obj][0]
        self.obj_euler, self.obj_trans = self.euler_trans_from_segment(self.mocap_obj.segments[0])
        self.arm_base_euler, self.arm_base_trans = self.euler_trans_from_segment(self.arm_base.segments[0])

        # Transform obj_euler and obj_trans into origin frame
        rotation_matrix = R.from_euler('xyz', self.arm_base_euler).as_matrix()
        self.obj_trans_robot_frame = rotation_matrix @ self.obj_trans + self.arm_base_trans
        self.obj_euler_robot_frame = R.from_matrix(rotation_matrix @ R.from_euler('xyz', self.obj_euler).as_matrix()).as_euler('xyz')

    def get_state(self):
        return self.obj_trans, self.obj_euler

if __name__ == "__main__":
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_cnf_only.yaml').read_text())
    default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                dim=1)
    sim_env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=config['visualize'],
                                        steps_per_action=60,
                                        friction_coefficient=config['friction_coefficient'] * 2.5,
                                        # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                        device=config['sim_device'],
                                        video_save_path=img_save_dir,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        gradual_control=False,
                                        gravity=True, # For data generation only
                                        randomize_obj_start=config.get('randomize_obj_start', False),
                                        )
    obj_reader = ObjectPoseReader(obj='screwdriver', mode='relative')
    # rospy.spin()
    import time
    time.sleep(.1)
    while True:
        root_coor, root_ori = obj_reader.get_state()
        cur_pose = sim_env.get_state()['q'].reshape(-1)

        root_ori = torch.tensor(root_ori, dtype=cur_pose.dtype)

        print(root_ori)

        cur_pose[-4:-1] = root_ori

        sim_env.set_pose(cur_pose.reshape(1,-1))