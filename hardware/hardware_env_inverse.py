import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import pathlib
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv
from isaac_victor_envs.tasks.allegro_ros import RosAllegroValveTurningEnv
if __name__ == "__main__":
    from allegro_ros import RosNode
else:
    from .allegro_ros import RosNode
import torch
import yaml
from lightweight_vicon_bridge.msg import MocapState
from tf.transformations import euler_from_quaternion
import pytorch_kinematics as pk

urdf_path = "/home/abhinav/Documents/git_packages/isaacgym-arm-envs/isaac_victor_envs/assets/xela_models/victor_allegro_stalk.urdf"
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class InverseObjectPoseCalculator:
    """
    Inverse of ObjectPoseReader - takes arm configuration and outputs where the screwdriver 
    should be in mocap world frame.
    """
    def __init__(self, obj='valve', device='cpu') -> None:
        self.obj = obj
        self.device = device

        # Same transformation matrices as in ObjectPoseReader
        self.object_to_hand_matrix = torch.tensor([[[ 5.9605e-08,  1.0000e+00,  0.0000e+00,  9.5000e-02],
         [-7.6604e-01,  5.9605e-08,  6.4279e-01, -9.3431e-03],
         [ 6.4279e-01,  0.0000e+00,  7.6604e-01, -1.1135e-02],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]], device=device) #scene_trans
        
        self.arm_mocap_to_arm_victor_matrix = torch.tensor([[[ 0.99973467,  0.01829301,  0.0139986 , -0.01603915],
                                                            [-0.01838437,  0.99981035,  0.00642524,  0.00957037],
                                                            [-0.01387841, -0.00668089,  0.99988137,  0.03222477],
                                                            [ 0.        ,  0.        ,  0.        ,  1.        ]]])
                                                                        
        self.object_to_hand_trans = pk.Transform3d(matrix=self.object_to_hand_matrix)
        self.hand_to_object_trans = self.object_to_hand_trans.inverse()
        self.arm_mocap_to_arm_victor_trans = pk.Transform3d(matrix=self.arm_mocap_to_arm_victor_matrix)

        # Initialize kinematics chain
        self.chain = pk.build_serial_chain_from_urdf(open(urdf_path, mode='rb').read(), 
                                                   'allegro_hand_base_link', 
                                                   root_link_name='victor_right_arm_link_1')
        self.chain = self.chain.to(device=device)

    def get_screwdriver_pose_from_arm_config(self, arm_config, arm_base_trans=None, arm_base_euler=None):
        """
        Given an arm configuration, calculate where the screwdriver should be in mocap world frame.
        
        Args:
            arm_config: torch.Tensor of shape (7,) - arm joint angles in radians
            arm_base_trans: np.array of shape (3,) - arm base translation in mocap world frame
            arm_base_euler: np.array of shape (3,) - arm base rotation in mocap world frame
        
        Returns:
            tuple: (screwdriver_trans, screwdriver_euler) in mocap world frame
        """
        # Ensure arm_config is the right shape and device
        if len(arm_config.shape) == 1:
            arm_config = arm_config.unsqueeze(0)  # Add batch dimension
        arm_config = arm_config.to(self.device)
        
        # Ensure arm_base_trans and arm_base_euler are numpy arrays on CPU
        if arm_base_trans is not None:
            arm_base_trans = np.array(arm_base_trans, dtype=np.float32)
        if arm_base_euler is not None:
            arm_base_euler = np.array(arm_base_euler, dtype=np.float32)
        
        # Forward kinematics to get hand pose in arm victor frame
        hand_pose_arm_victor = self.chain.forward_kinematics(arm_config)
        
        # Transform from arm victor frame to arm mocap frame
        # hand_to_arm_mocap = arm_victor_to_arm_mocap * hand_to_arm_victor
        hand_to_arm_mocap = self.arm_mocap_to_arm_victor_trans.inverse().compose(hand_pose_arm_victor)
        
        # If arm base pose is provided, transform to mocap world frame
        if arm_base_trans is not None and arm_base_euler is not None:
            # Create arm mocap to mocap world transformation
            arm_mocap_to_mocap_world_trans = pk.Transform3d(
                pos=torch.tensor(arm_base_trans, device=self.device, dtype=torch.float32), 
                rot=torch.tensor(arm_base_euler, device=self.device, dtype=torch.float32)
            )
            
            # Transform hand pose to mocap world frame
            # hand_to_mocap_world = arm_mocap_to_mocap_world * hand_to_arm_mocap
            hand_to_mocap_world_trans = arm_mocap_to_mocap_world_trans.compose(hand_to_arm_mocap)
            
            # Get screwdriver pose in mocap world frame
            # screwdriver_to_mocap_world = hand_to_mocap_world * screwdriver_to_hand
            screwdriver_to_hand_trans = self.object_to_hand_trans
            screwdriver_to_mocap_world_trans = hand_to_mocap_world_trans.compose(screwdriver_to_hand_trans)
            
            # Extract position and orientation
            screwdriver_matrix = screwdriver_to_mocap_world_trans.get_matrix()[0]
            screwdriver_trans = screwdriver_matrix[:3, 3].cpu().numpy()
            screwdriver_rot_matrix = screwdriver_matrix[:3, :3].cpu().numpy()
            screwdriver_euler = R.from_matrix(screwdriver_rot_matrix).as_euler('xyz')
            
            return screwdriver_trans, screwdriver_euler
        else:
            # Return hand pose in arm mocap frame if no arm base pose provided
            hand_matrix = hand_to_arm_mocap.get_matrix()[0]
            hand_trans = hand_matrix[:3, 3].cpu().numpy()
            hand_rot_matrix = hand_matrix[:3, :3].cpu().numpy()
            hand_euler = R.from_matrix(hand_rot_matrix).as_euler('xyz')
            
            return hand_trans, hand_euler

    def get_screwdriver_pose_with_mocap_subscriber(self, arm_config):
        """
        Get screwdriver pose using mocap data for arm base pose.
        This requires ROS to be running and mocap data to be available.
        
        Args:
            arm_config: torch.Tensor of shape (7,) - arm joint angles in radians
            
        Returns:
            tuple: (screwdriver_trans, screwdriver_euler) in mocap world frame, or None if mocap data unavailable
        """
        # This would need to be called from a ROS node context
        # For now, return None and let the caller handle mocap data
        return None

class ObjectPoseReader:
    def __init__(self, obj='valve', mode='relative', device='cpu') -> None:
        # if __name__ == '__main__':
        #     rospy.init_node('object_pose_reader')
        self.mode = mode
        self.obj = obj
        self.device=device

        self.mocap_sub = rospy.Subscriber('/mocap_tracking', MocapState, self.mocap_callback)

        self.object_to_hand_matrix = torch.tensor([[[ 5.9605e-08,  1.0000e+00,  0.0000e+00,  9.5000e-02],
         [-7.6604e-01,  5.9605e-08,  6.4279e-01, -9.3431e-03],
         [ 6.4279e-01,  0.0000e+00,  7.6604e-01, -1.1135e-02],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]], device=device) #scene_trans
        
        self.arm_mocap_to_arm_victor_matrix = torch.tensor([[[ 0.99973467,  0.01829301,  0.0139986 , -0.01603915],
                                                            [-0.01838437,  0.99981035,  0.00642524,  0.00957037],
                                                            [-0.01387841, -0.00668089,  0.99988137,  0.03222477],
                                                            [ 0.        ,  0.        ,  0.        ,  1.        ]]])
                                                                        
        self.object_to_hand_trans = pk.Transform3d(matrix=self.object_to_hand_matrix)
        self.hand_to_object_trans = self.object_to_hand_trans.inverse()

        self.arm_mocap_to_arm_victor_trans = pk.Transform3d(matrix=self.arm_mocap_to_arm_victor_matrix)

    def euler_trans_from_segment(self, segment):
        transform = segment.transform
        obj_quat = (transform.rotation)
        # obj_euler = np.array(euler_from_quaternion([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w], axes='rxyz'))
        obj_euler = np.array(euler_from_quaternion([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w], axes='rxyz'))
        obj_trans = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        return obj_euler, obj_trans

    def mocap_callback(self, data):
        self.arm_base = [i for i in data.tracked_objects if i.name == 'right_arm_base']
        if len(self.arm_base) > 0:
            self.arm_base = self.arm_base[0]
        else:
            self.arm_base = None
        self.mocap_obj = [i for i in data.tracked_objects if i.name == self.obj][0]
        self.obj_euler_, self.obj_trans_ = self.euler_trans_from_segment(self.mocap_obj.segments[0])
        self.obj_trans_[2] += .02
        # self.obj_euler_[0] += .02
        
        # self.obj_euler_[0] -= .03

    def get_state(self):
        if self.obj == 'valve':
            return self.obj_euler_[1]
        return self.obj_trans_, self.obj_euler_
    
    def get_state_world_frame_pos(self):
        
        if hasattr(self, 'arm_base') and self.arm_base is not None:
            self.obj_to_mocap_world_trans = pk.Transform3d(pos=torch.tensor(self.obj_trans_, device=self.device, dtype=self.object_to_hand_trans.dtype), rot=torch.tensor(np.zeros_like(self.obj_euler_), device=self.device, dtype=self.object_to_hand_trans.dtype))
            self.arm_base_euler, self.arm_base_trans = self.euler_trans_from_segment(self.arm_base.segments[0])

            self.arm_mocap_to_mocap_world_trans = pk.Transform3d(pos=torch.tensor(self.arm_base_trans, device=self.device, dtype=self.object_to_hand_trans.dtype), rot=torch.tensor(self.arm_base_euler, device=self.device, dtype=self.object_to_hand_trans.dtype))
        
        #    C to A                 = B to A * C to B
        #    obj to arm_mocap        = mocap_world_to_arm_mocap * obj to mocap_world
            self.obj_to_arm_mocap_trans = self.arm_mocap_to_mocap_world_trans.inverse().compose(self.obj_to_mocap_world_trans)
            self.obj_trans_robot_frame = self.obj_to_arm_mocap_trans.get_matrix()[0][:3, 3]
            self.obj_euler_robot_frame_for_IK = self.obj_to_arm_mocap_trans.get_matrix()[0][:3, :3]
            self.hand_to_arm_mocap = self.obj_to_arm_mocap_trans.compose(self.hand_to_object_trans)

            self.hand_to_mocap_world_trans = self.arm_mocap_to_mocap_world_trans.compose(self.hand_to_arm_mocap)
            self.hand_to_mocap_world_position = self.hand_to_mocap_world_trans.get_matrix()[0][:3, 3].cpu().numpy()

            # Object trans is difference between object position and hand position

            # Skip below to get world position of object
            
            # print('obj_trans pre', self.obj_trans_)
            self.obj_trans_[-1] -= .1 + 0.412*2.54/100
            # print('obj_trans post z offset', self.obj_trans_)
            self.obj_trans_ = self.obj_trans_ - self.hand_to_mocap_world_position
            # print('obj_trans post hand offset', self.obj_trans_)
            # print('hand to mocap world position', self.hand_to_mocap_world_position)

        return self.obj_trans_, self.obj_euler_
    
    def get_target_IK_pose(self):
        # Create pk Transform3D from self obj_trans and obj_euler

        #    C to A                 = B to A * C to B
        # C=hand, A=arm_victor, B=arm_mocap
        # C to A = hand_to_arm_victor, B to A = arm_mocap_to_arm_victor, C to B = hand_to_arm_mocap
        hand_to_arm_victor = self.arm_mocap_to_arm_victor_trans.compose(self.hand_to_arm_mocap)

        return (hand_to_arm_victor)


class HardwareEnv:
    def __init__(self, default_pos, num_repeat=1, gradual_control=False, finger_list=['index', 'middle', 'ring', 'thumb'], kp=4, obj='valve', ori_only=True, mode='relative', device='cuda:0', node_name='allegro_hand_viz'):
        self.__all_finger_list = ['index', 'middle', 'ring', 'thumb']
        self.obj = obj
        self.__finger_list = finger_list
        self.__ros_node = RosNode(node_name=node_name, kp=kp, num_repeat=num_repeat, gradual_control=gradual_control)

        self.obj_reader = ObjectPoseReader(obj=obj, mode=mode)

        self.device = device
        self.default_dof_pos = default_pos.clone()
        self.ori_only = ori_only
    
    def get_state(self):
        # rospy.sleep(0.5)
        robot_state = self.__ros_node.allegro_joint_pos.float()
        robot_state = robot_state.to(self.device)
        index, mid, ring, thumb = torch.chunk(robot_state, chunks=4, dim=-1)
        state = {}
        state['index'] = index
        state['middle'] = mid
        state['ring'] = ring
        state['thumb'] = thumb
        q = []
        for finger_name in self.__finger_list:
            q.append(state[finger_name])
        if self.obj == 'valve':
            ori = self.obj_reader.get_state()
            ori = torch.tensor([ori]).float().to(self.device)
            q.append(ori)
        # elif self.obj == 'screwdriver':
        elif 'screwdriver' in self.obj:
            pos, ori = self.obj_reader.get_state()
            pos = torch.tensor(pos).float().to(self.device)
            ori = torch.tensor(ori).float().to(self.device)
            # ori = ori * 0 # debug
            if self.ori_only:
                q.append(ori)
                q.append(torch.zeros(1).float().to(self.device)) # add the screwdriver cap angle
            else:
                raise NotImplementedError
        all_state = torch.cat((robot_state, ori), dim=-1)
        state['all_state'] = all_state
        state['q'] = torch.cat(q).unsqueeze(0)
        state['theta'] = ori
        return state
    def step(self, action):
        action = self.partial_to_full_state(action)
        # action[:, -2] += 0.25
        if len(action.shape) == 2:
            action = action.squeeze(0)
        if len(action.shape) == 2:
            action = action[0]
        self.__ros_node.apply_action(action)
        return self.get_state()
    def reset(self):
        return self.__ros_node.apply_action(self.default_dof_pos.squeeze(0))
    def partial_to_full_state(self, partial):
        """
        :params partial: B x 8 joint configurations for index and thumb
        :return full: B x 16 joint configuration for full hand

        # assume that default is zeros, but could change
        """
        finger_data = torch.chunk(partial, chunks=len(self.__finger_list), dim=-1)
        full = []
        ctr = 0
        for finger_name in self.__all_finger_list:
            if finger_name not in self.__finger_list:
                full.append(torch.zeros_like(finger_data[0]))
            else:
                full.append(finger_data[ctr])
                ctr += 1
        full = torch.cat(full, dim=-1)
        return full

def regularized_ik(n_tgts):
    """
    Generate an initial guess for the IK solver that keeps the arm configuration close to previously used config
    """
    init_dof = torch.zeros((n_tgts, 7), device=device)
    # The arm config from ICRA 25 in degrees. To avoid strange local minima
    init_dof += torch.tensor([[50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78]], device=device)
    init_dof = init_dof / 180 * np.pi

    init_dof += torch.randn_like(init_dof) * 0.01

    return init_dof

if __name__ == "__main__":
    """
    Example usage of the inverse functionality
    """
    device = 'cuda:0'
    
    # Initialize the inverse calculator
    inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver_catching', device=device)
    
    # Example arm configuration (7 joint angles in radians)
    # This is the configuration from the original code
    arm_config = torch.tensor([50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78], device=device) / 180 * np.pi
    
    # Example arm base pose in mocap world frame (you would get this from mocap data)
    arm_base_trans = np.array([0.0, 0.0, 0.0])  # Replace with actual mocap data
    arm_base_euler = np.array([0.0, 0.0, 0.0])  # Replace with actual mocap data
    
    # Calculate where the screwdriver should be
    screwdriver_trans, screwdriver_euler = inverse_calc.get_screwdriver_pose_from_arm_config(
        arm_config, arm_base_trans, arm_base_euler
    )
    
    print(f"Screwdriver position in mocap world frame: {screwdriver_trans}")
    print(f"Screwdriver orientation in mocap world frame: {screwdriver_euler}")
    
    # Test without arm base pose (returns hand pose in arm mocap frame)
    hand_trans, hand_euler = inverse_calc.get_screwdriver_pose_from_arm_config(arm_config)
    print(f"Hand position in arm mocap frame: {hand_trans}")
    print(f"Hand orientation in arm mocap frame: {hand_euler}")
