import rospy
import pathlib
import argparse
import numpy as np

# deprecated, but we need it for ros_numpy
np.float = np.float64
import ros_numpy
import torch
import time
from sensor_msgs.msg import JointState

from arm_rviz.rviz_animation_controller import RvizAnimationController
from rviz_voxelgrid_visuals import conversions
from arm_rviz_msgs.msg import VoxelgridStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3

from victor_hardware_interface_msgs.msg import MotionCommand, ControlMode

import pytorch_kinematics as pk
import os, sys

from isaac_victor_envs.utils import get_assets_dir

from ccai.env import generate_random_sphere_world


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset_dir = get_assets_dir()
asset = asset_dir + '/victor/victor_mallet.urdf'

ee_name = 'victor_left_gripper_palm'
VICTOR_URDF_FILE = '/home/tpower/catkin_ws/src/kuka_iiwa_interface/victor_description/urdf/victor.urdf'

chain = pk.build_serial_chain_from_urdf(open(VICTOR_URDF_FILE).read(), ee_name)

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', type=str, required=True)
    parser.add_argument('--mode', type=str, default='reach', choices=['reach', 'table'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # load data to start
    data = np.load(f'{args.exp_folder}/data_vis_9.npz', allow_pickle=True)[args.mode].item()
    actual_trajectory = data['trajectories']
    planned_trajectories = data['sampled_trajectories']
    sdf_grid = data['constraints']

    # how many trials are there?
    import pathlib

    path = pathlib.Path(f'{args.exp_folder}')
    print(path)
    files = list(path.rglob('*.npz'))
    num_epochs = len(files)

    VICTOR_BASE_POSITION = np.array([0, 0, 0])
    rospy.init_node('victor_visualiser')
    with HiddenPrints():
        chain = pk.build_serial_chain_from_urdf(
            open(VICTOR_URDF_FILE).read(),
            'victor_left_gripper_palm')

    pub_vox = rospy.Publisher('/voxelgrid_1', VoxelgridStamped, queue_size=1)
    trajectory_pubs = rospy.Publisher('trajectory', MarkerArray, queue_size=10)
    current_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    victor_right_arm_pub = rospy.Publisher('/victor/right_arm/motion_command', MotionCommand, queue_size=10)
    victor_left_arm_pub = rospy.Publisher('/victor/left_arm/motion_command', MotionCommand, queue_size=10)
    goal_pub = rospy.Publisher('goal', Marker, queue_size=10)

    while pub_vox.get_num_connections() == 0:
        rospy.sleep(0.1)

    ve = RvizAnimationController(n_time_steps=num_epochs, ns='epoch')
    vt = RvizAnimationController(n_time_steps=actual_trajectory.shape[0], ns='batch')


    def convert_config_to_ee(config):
        config_shape = config.shape
        ee_transform = chain.forward_kinematics(torch.from_numpy(config.reshape(-1, 7)))
        ee_location = ee_transform.transform_points(torch.zeros(1, 3))
        ee_shape = list(config_shape)
        ee_shape[-1] = 3
        return ee_location.reshape(ee_shape)


    def set_joinstate(arm, config):
        assert len(config) == 7

        if arm == 'left':
            prefix = 'victor_left_arm_joint_'
        elif arm == 'right':
            prefix = 'victor_right_arm_joint_'
        else:
            raise ValueError('invalid arm')

        """
        msg = JointState()
        msg.header.stamp = rospy.Time.now()

        msg.name = ['']*7
        msg.position = [0]*7
        msg.velocity = [0] * 7
        msg.effort = [0] * 7
        for i in range(7):
            msg.name[i] = prefix + str(i+1)
            msg.position[i] = config[i]
        """
        msg = MotionCommand()
        msg.header.stamp = rospy.Time.now()
        msg.control_mode.mode = 0
        msg.joint_position.joint_1 = config[0]
        msg.joint_position.joint_2 = config[1]
        msg.joint_position.joint_3 = config[2]
        msg.joint_position.joint_4 = config[3]
        msg.joint_position.joint_5 = config[4]
        msg.joint_position.joint_6 = config[5]
        msg.joint_position.joint_7 = config[6]

        # current_state_pub.publish(msg)

        if arm == 'left':
            victor_left_arm_pub.publish(msg)
        else:
            victor_right_arm_pub.publish(msg)


    set_joinstate('left', np.zeros(7))
    set_joinstate('right', np.zeros(7))

    while not ve.done:
        e = ve.t()
        fileno = (e + 1) * 10 - 1
        # load files
        data = np.load(f'{args.exp_folder}/data_vis_{fileno}.npz', allow_pickle=True)[args.mode].item()
        actual_trajectory = data['trajectories']
        planned_trajectories = data['sampled_trajectories']
        sdf = data['constraints']
        goals = data['goals']

        vt.reset()

        while not vt.done:
            t = vt.t()
            if False:#args.mode == 'reach':
                # Publish environment sdf
                sdf_centre = np.array([0.5, 0.0, 1.0])
                voxel_grid = np.where(sdf[t] < 0, 1, 0)
                voxel_grid = voxel_grid.reshape((64, 64, 64))
                scale = 1.0 / 64.0
                pub_vox.publish(conversions.vox_to_voxelgrid_stamped(voxel_grid, scale=scale, frame_id='world',
                                                                     origin=np.array([0.25, -0.25, 0.5])))

            print(actual_trajectory.shape)
            print(planned_trajectories.shape)
            print(sdf.shape)
            print(goals.shape)

            current_state = actual_trajectory[t, 0]
            plans = planned_trajectories[t]

            plans_ee = convert_config_to_ee(plans)
            print(plans_ee.shape)
            set_joinstate('left', current_state)

            print(plans[:, 0])
            print('--')
            print(actual_trajectory[t, 0])
            trajectories_msg = MarkerArray()

            # publish plans
            for n in range(plans.shape[0]):
                traj_msg = Marker()
                traj_msg.header.frame_id = 'world'
                traj_msg.type = Marker.SPHERE_LIST
                traj_msg.scale.x = 0.01
                traj_msg.scale.y = 0.01
                traj_msg.scale.z = 0.01
                # randomise colour
                traj_msg.color.b, traj_msg.color.g, traj_msg.color.r = 1.0, 0.0, 0.0  # np.random.rand(3)
                traj_msg.color.a = 1.0
                traj_msg.pose.orientation.w = 1.0
                traj_msg.id = n
                for m in range(plans_ee.shape[1]):
                    point = ros_numpy.msgify(Vector3, plans_ee[n, m, :3])
                    traj_msg.points.append(point)
                trajectories_msg.markers.append(traj_msg)
            actual_ee = convert_config_to_ee(actual_trajectory)
            print(actual_ee.shape)
            # publish actual trajectory
            traj_msg = Marker()
            traj_msg.header.frame_id = 'world'
            traj_msg.type = Marker.SPHERE_LIST
            traj_msg.scale.x = 0.01
            traj_msg.scale.y = 0.01
            traj_msg.scale.z = 0.01
            # randomise colour
            traj_msg.color.b, traj_msg.color.g, traj_msg.color.r = 0.0, 1.0, 0.0  # np.random.rand(3)
            traj_msg.color.a = 1.0
            traj_msg.pose.orientation.w = 1.0
            traj_msg.id = plans.shape[0] + 1
            print('--')
            for m in range(actual_trajectory.shape[1]):
                print(actual_trajectory[t, m])
                print(actual_ee[t, m, :3])
                point = ros_numpy.msgify(Vector3, actual_ee[t, m, :3])
                traj_msg.points.append(point)
            trajectories_msg.markers.append(traj_msg)

            trajectory_pubs.publish(trajectories_msg)

            goal_msg = Marker()
            goal_msg.header.frame_id = 'world'
            goal_msg.type = Marker.POINTS
            goal_msg.scale.x = 0.1
            goal_msg.scale.y = 0.1
            goal_msg.scale.z = 0.1
            goal_msg.color.b = 1.0
            goal_msg.color.a = 1.0
            goal_msg.points = [ros_numpy.msgify(Vector3, goals[t])]
            goal_pub.publish(goal_msg)

            vt.step()
        ve.step()
