#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import torch
from copy import deepcopy

LIB_CMD_TOPIC = '/allegroHand/lib_cmd'

class RosNode(object):
    '''
    Ros Node for communication with the hardware
    '''
    def __init__(self, node_name='run_policy', num_repeat=1, gradual_control=False, kp=4, use_grav_comp=False):
        try:
            rospy.init_node('allegro_hand_node')
        except:
            pass
        self.node_name = node_name
        self.num_repeat = num_repeat
        self.get_allegro_bounds()
        # rospy.init_node(self.node_name)
        self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.set_allegro_states_subscriber()
        self.set_allegro_cmd_publisher()
        self.set_allegro_grav_comp_subscriber()
        self.lib_cmd_publisher = rospy.Publisher(LIB_CMD_TOPIC, String, queue_size=-1)
        self.lib_cmd_publisher.publish('gravcomp')
        rospy.sleep(2)
        print('ros listner initialized.')
        self.current_joint_pose = None
        self.kp = kp
        self.use_grav_comp = use_grav_comp
        self.gradual_control = gradual_control

        
    
    def get_allegro_bounds(self, allegro_lb=None, allegro_ub=None):
        '''
        Get allegro hand movement bound
        Args:
            1. allegro_lb: [16] lower bound
            2. allegro_ub: [16] upper bound
        '''
        if allegro_lb is None:
            self.allegro_lb = torch.tensor([-0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270,
                                            -0.4700, -0.1960, -0.1740, -0.2270,  0.2630, -0.1050, -0.1890, -0.1620])
        else:
            self.allegro_lb = allegro_lb

        if allegro_ub is None:
            self.allegro_ub = torch.tensor([0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180, 0.4700,
                                            1.6100, 1.7090, 1.6180, 1.3960, 1.1630, 1.6440, 1.7190])
        else:
            self.allegro_ub = allegro_ub
    
    def allegro_joint_callback(self, data):
        '''
        Function called each time we recieve a joint_state
        Save the latest Joint State in current_joint_pose for use at the moment of publish 
        Transform the Joint State positions into a torch tensor to be used when needed
        '''
        self.current_joint_pose = data
        # self.allegro_joint_pos_scaled = unscale(torch.tensor(data.position), self.allegro_lb, self.allegro_ub)
        self.allegro_joint_pos = torch.tensor(data.position)

    def set_allegro_states_subscriber(self, topic_name='/allegroHand/joint_states'):
        '''
        Subscribe to a rostopic and update the corresponding values
        Args:
            1. topic_name: str of rostopic to subscribe
        '''
        self.subscriber = rospy.Subscriber(topic_name, JointState, self.allegro_joint_callback)
    def set_allegro_grav_comp_subscriber(self, topic_name='/allegroHand/grav_comp_torques'):
        '''
        Subscribe to a rostopic and update the corresponding values
        Args:
            1. topic_name: str of rostopic to subscribe
        '''
        self.grav_comp_subscriber = rospy.Subscriber(topic_name, JointState, self.allegro_grav_comp_callback)

    def allegro_grav_comp_callback(self, data):
        '''
        Function called each time we recieve a joint_state
        Save the latest Joint State in current_joint_pose for use at the moment of publish 
        Transform the Joint State positions into a torch tensor to be used when needed
        '''
        self.grav_comp_data = data
        self.grav_comp_torque = torch.tensor(data.effort)
    def set_allegro_cmd_publisher(self, topic_name='/allegroHand/joint_cmd'):
        '''
        Publish to a rostopic and update the corresponding values
        Args:
            1. topic_name: str of rostopic to subscribe
        '''
        self.joint_comm_publisher = rospy.Publisher(topic_name, JointState, queue_size=-1)
        print('joint command publisher initialized.')
        
    def apply_action(self, action, weight=0.5):
        "the action has to be full action (16 dimensional)"

        action_sequence = np.linspace(torch.tensor(self.current_joint_pose.position), action.cpu(), int(self.num_repeat * 0.75) + 1)[1:]
        action_sequence = np.concatenate([action_sequence, np.tile(action.cpu().numpy(), (self.num_repeat - len(action_sequence), 1))])
        if len(action.shape) == 2:
            action = action.squeeze(0)
        # if self.num_repeat == 1:
        # for i in range(self.num_repeat):
        #     desired_js = deepcopy(self.current_joint_pose) # We copy the message type from the last current joint pose recieved to have the format
        #     if self.use_grav_comp:
        #         desired_js.position = action + self.grav_comp_torque.to(action.device) / self.kp
        #     else:
        #         desired_js.position = action
        #     desired_js.effort = list([]) # We set the effort command to zero because we are doing position control and not torque control
        #     self.joint_comm_publisher.publish(desired_js) # Publish the desired command
        #     rospy.sleep(0.1)
        for i in range(self.num_repeat):
            # action = list(action.detach().cpu().numpy())
            action = list(action_sequence[i])
            desired_js = deepcopy(self.current_joint_pose) # We copy the message type from the last current joint pose recieved to have the format
            desired_js.position = action # We change the position to have the commanded joint angles
            desired_js.effort = list([]) # We set the effort command to zero because we are doing position control and not torque control
            self.joint_comm_publisher.publish(desired_js) # Publish the desired command
            # rospy.sleep(0.05)

def main():
    from isaac_victor_envs.utils import get_assets_dir
    import pytorch_kinematics as pk
    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
        'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
        'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
        'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
        'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
    }
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'
    # action_open = torch.zeros(16)
    # action_close = torch.tensor([
    #     0.0, 0.8, 0.8, 0.8,
    #     0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0
    # ])
    action_open = torch.tensor([0.0, 0.5727, 0.4090, 0.4090, 
                            0.0, 0.5727, 0.4090, 0.4090, 
                            0.0, 0.5727, 0.4090, 0.4090, 
                            0.7745, 0.4745, 0.5728, 0.5227])
    action_close = torch.tensor([0.0, 0.8727, 1.3090, 1.3090, 
                            0.0, 0.8727, 1.3090, 1.3090, 
                            0.0, 0.5000, 1.3484, 1.3484, 
                            1.3745, 0.4745, 1.1728, 1.7227])
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in ee_names.keys()]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    fk_dict = chain.forward_kinematics(action_close, frame_indices=frame_indices)
    desired_ee_pos = fk_dict['allegro_hand_hitosashi_finger_finger_0_aftc_base_link'].get_matrix()[0, :3, 3]

    num_interpolation_steps = 1000
    total_steps = 5000
    
    
    ros_node = RosNode(num_repeat=1)
    rospy.sleep(0.5)
    ctr = 0
    
    robot_state = []
    while not rospy.is_shutdown():
        ctr += 1
        
        if ctr <= 180:
            action = action_open
                
        else:
            if num_interpolation_steps > 1:
                if ctr == 181:  # Initialize interpolation when switching to close
                    current_pos = ros_node.allegro_joint_pos
                    interpolated_actions = interpolate_tensors(current_pos, action_close, num_interpolation_steps)
                    step_idx = 0
                    
                if step_idx < num_interpolation_steps:
                    action = interpolated_actions[step_idx]
                    step_idx += 1
                else:
                    action = action_close
            else:
                action = action_close
                
        ros_node.apply_action(action, weight=1)
        robot_state.append(ros_node.allegro_joint_pos.clone())
        if ctr >= 5000:
            break
    robot_state = torch.stack(robot_state)
    index_ee_pos = chain.forward_kinematics(robot_state, frame_indices=frame_indices)['allegro_hand_hitosashi_finger_finger_0_aftc_base_link'].get_matrix()[0, :3, 3]
    data = {'robot_state': robot_state, 'index_ee_pos': index_ee_pos, 'target_pos': desired_ee_pos}
    # torch.save(data, f'robot_state_{num_interpolation_steps}.pt')
        # if ctr <= 180:
        #     action = torch.tensor([0.0, 0.5727, 0.4090, 0.4090, 
        #                            0.0, 0.5727, 0.4090, 0.4090, 
        #                            0.0, 0.5727, 0.4090, 0.4090, 
        #                            0.7745, 0.4745, 0.5728, 0.5227])
        # else:
        #     action = torch.tensor([0.0, 0.8727, 1.3090, 1.3090, 
        #                            0.0, 0.8727, 1.3090, 1.3090, 
        #                            0.0, 0.5000, 1.3484, 1.3484, 
        #                            1.3745, 0.4745, 1.1728, 1.7227])

def interpolate_tensors(start_tensor, end_tensor, num_steps):
    """
    Linearly interpolate between two tensors
    
    Args:
        start_tensor (torch.Tensor): Starting tensor of shape (16,)
        end_tensor (torch.Tensor): Ending tensor of shape (16,)
        num_steps (int): Number of interpolation steps
        
    Returns:
        torch.Tensor: Interpolated tensors of shape (num_steps, 16)
    """
    return torch.stack([
        start_tensor + (end_tensor - start_tensor) * (step / (num_steps - 1))
        for step in range(num_steps)
    ])
if __name__ == '__main__':
   main()
   rospy.spin()