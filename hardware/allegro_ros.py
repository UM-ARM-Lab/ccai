#!/usr/bin/env python3
import json
import numpy as np
import re
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
from copy import deepcopy

LIB_CMD_TOPIC = '/allegroHand/lib_cmd'
TIMED_JOINT_CMD_TOPIC = '/allegroHand/timed_joint_cmd'
COMMAND_MODE_REPEAT = 'repeat'
COMMAND_MODE_TIMED_HOLD_CURRENT = 'timed_hold_current'
COMMAND_MODE_CONTROLLER_TIMED_HOLD_CURRENT = 'controller_timed_hold_current'
COMMAND_MODE_VALVE_TIMED_REPEAT = 'valve_timed_repeat'

def allegro_joint_state_order_from_names(names):
    if names is None:
        return None
    names = list(names)
    if len(names) != 16:
        return None
    order = []
    for name in names:
        match = re.search(r"(?:joint_)?(\d+)(?:\.0)?$", str(name))
        if match is None:
            return None
        joint_index = int(match.group(1))
        if joint_index < 0 or joint_index >= 16:
            return None
        order.append(joint_index)
    if sorted(order) != list(range(16)):
        return None
    return order


def canonicalize_allegro_joint_positions(positions, names):
    positions = np.asarray(positions, dtype=np.float32).reshape(-1)
    order = allegro_joint_state_order_from_names(names)
    if order is None or positions.shape[0] != len(order):
        return positions
    canonical = np.empty_like(positions)
    for message_index, canonical_index in enumerate(order):
        canonical[canonical_index] = positions[message_index]
    return canonical


def reorder_canonical_allegro_target(target, names):
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    order = allegro_joint_state_order_from_names(names)
    if order is None or target.shape[0] != len(order):
        return target
    return target[np.asarray(order, dtype=np.int64)]

class RosNode(object):
    '''
    Ros Node for communication with the hardware
    '''
    def __init__(
        self,
        node_name='run_policy',
        num_repeat=1,
        gradual_control=False,
        kp=4,
        use_grav_comp=False,
        command_mode=COMMAND_MODE_REPEAT,
        command_duration_s=1.0 / 12.0,
        initialize_control=True,
    ):
        try:
            rospy.init_node(node_name)
        except:
            print('failed to initialize node')
            pass
        self.node_name = node_name
        self.num_repeat = num_repeat
        self.command_mode = str(command_mode)
        if self.command_mode not in (
            COMMAND_MODE_REPEAT,
            COMMAND_MODE_TIMED_HOLD_CURRENT,
            COMMAND_MODE_CONTROLLER_TIMED_HOLD_CURRENT,
            COMMAND_MODE_VALVE_TIMED_REPEAT,
        ):
            raise ValueError(
                f"Unsupported Allegro command_mode={self.command_mode!r}; "
                f"expected {COMMAND_MODE_REPEAT!r}, {COMMAND_MODE_TIMED_HOLD_CURRENT!r}, "
                f"or {COMMAND_MODE_CONTROLLER_TIMED_HOLD_CURRENT!r}."
            )
        self.command_duration_s = float(command_duration_s)
        if not np.isfinite(self.command_duration_s) or self.command_duration_s < 0.0:
            raise ValueError(f"command_duration_s must be finite and non-negative, got {command_duration_s}.")
        if self.command_mode == COMMAND_MODE_CONTROLLER_TIMED_HOLD_CURRENT and self.command_duration_s <= 0.0:
            raise ValueError(
                f"command_duration_s must be positive for {COMMAND_MODE_CONTROLLER_TIMED_HOLD_CURRENT!r}."
            )
        self.get_allegro_bounds()
        # rospy.init_node(self.node_name)
        self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self._joint_state_order_diagnostic_printed = False
        self.current_joint_pose = None
        self._joint_state_seq = 0
        self.set_allegro_states_subscriber()
        self.set_allegro_cmd_publisher()
        self.set_allegro_timed_cmd_publisher()
        self.set_allegro_grav_comp_subscriber()
        self.lib_cmd_publisher = rospy.Publisher(LIB_CMD_TOPIC, String, queue_size=-1)
        if initialize_control:
            self.lib_cmd_publisher.publish('gravcomp')
        rospy.sleep(2)
        print('ros listner initialized.')
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
        self._joint_state_seq += 1
        # self.allegro_joint_pos_scaled = unscale(torch.tensor(data.position), self.allegro_lb, self.allegro_ub)
        names = getattr(data, 'name', None)
        order = allegro_joint_state_order_from_names(names)
        if not self._joint_state_order_diagnostic_printed:
            if order is None:
                print(
                    'ALLEGRO_JOINT_STATE_ORDER_FALLBACK '
                    + json.dumps({'reason': 'missing_or_unparsable_names', 'names': list(names or [])}),
                    flush=True,
                )
            else:
                print(
                    'ALLEGRO_JOINT_STATE_ORDER '
                    + json.dumps({'message_index_to_canonical_index': order, 'names': list(names)}),
                    flush=True,
                )
            self._joint_state_order_diagnostic_printed = True
        self.allegro_joint_pos = torch.tensor(canonicalize_allegro_joint_positions(data.position, names))

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

    def set_allegro_timed_cmd_publisher(self, topic_name=TIMED_JOINT_CMD_TOPIC):
        self.timed_joint_comm_publisher = rospy.Publisher(topic_name, JointTrajectory, queue_size=-1)
        print('timed joint command publisher initialized.')
        
    def _publish_canonical_target(self, target_canonical):
        target = list(reorder_canonical_allegro_target(
            target_canonical,
            getattr(self.current_joint_pose, 'name', None),
        ))
        desired_js = deepcopy(self.current_joint_pose)
        desired_js.position = target
        desired_js.effort = list([])
        self.joint_comm_publisher.publish(desired_js)

    def _publish_canonical_timed_target(self, target_canonical):
        target = list(reorder_canonical_allegro_target(
            target_canonical,
            getattr(self.current_joint_pose, 'name', None),
        ))
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start = rospy.Duration.from_sec(self.command_duration_s)

        timed_cmd = JointTrajectory()
        timed_cmd.header.stamp = rospy.Time.now()
        timed_cmd.joint_names = list(getattr(self.current_joint_pose, 'name', []))
        timed_cmd.points = [point]
        self.timed_joint_comm_publisher.publish(timed_cmd)

    def apply_action(self, action, weight=0.5, hold_action=None):
        "the action has to be full action (16 dimensional)"

        if len(action.shape) == 2:
            action = action.squeeze(0)
        action_np = action.detach().cpu().numpy() if hasattr(action, 'detach') else np.asarray(action, dtype=np.float32)
        if self.command_mode == COMMAND_MODE_VALVE_TIMED_REPEAT:
            from hardware.valve_timed_client import ValveTimedClient
            if not hasattr(self, '_valve_client'):
                self._valve_client = ValveTimedClient(self)
            self.last_valve_result = self._valve_client.execute(action_np, self.num_repeat, self.command_duration_s)
            return self.last_valve_result
        if self.command_mode == COMMAND_MODE_CONTROLLER_TIMED_HOLD_CURRENT:
            self._publish_canonical_timed_target(action_np)
            return
        if self.command_mode == COMMAND_MODE_TIMED_HOLD_CURRENT:
            self._publish_canonical_target(action_np)
            rospy.sleep(self.command_duration_s)
            if hold_action is None:
                hold_canonical = canonicalize_allegro_joint_positions(
                    self.current_joint_pose.position,
                    getattr(self.current_joint_pose, 'name', None),
                )
            else:
                if len(hold_action.shape) == 2:
                    hold_action = hold_action.squeeze(0)
                hold_canonical = (
                    hold_action.detach().cpu().numpy()
                    if hasattr(hold_action, 'detach')
                    else np.asarray(hold_action, dtype=np.float32)
                )
            self._publish_canonical_target(hold_canonical)
            return
        current_canonical = canonicalize_allegro_joint_positions(
            self.current_joint_pose.position,
            getattr(self.current_joint_pose, 'name', None),
        )
        action_sequence = np.linspace(current_canonical, action_np, int(self.num_repeat * 0.75) + 1)[1:]
        action_sequence = np.concatenate([action_sequence, np.tile(action_np, (self.num_repeat - len(action_sequence), 1))])
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
            self._publish_canonical_target(action_sequence[i]) # Publish the desired command
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
