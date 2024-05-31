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
            rospy.sleep(0.05)

def main():
    ros_node = RosNode()
    while not rospy.is_shutdown():
        # The rospy sleep set the seconds before running another iteration. During this time, we will recieve multiple joint state, but we will used the last one
        # recieved for the ros_node.step(obs) and ros_node.apply_action. If we want to compute the action after each joint state update, we can put the apply_action inside 
        # the allegro_joint_callback function.
        rospy.sleep(1)
        
        # ros_node.get_object_pose()
        # obs = ros_node.get_observation()
        action = torch.randn(16) / 3
        print(ros_node.allegro_joint_pos)
        ros_node.apply_action(action, weight=1)
        
        

if __name__ == '__main__':
   main()
   rospy.spin()
