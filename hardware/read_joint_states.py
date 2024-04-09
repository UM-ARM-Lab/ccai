from allegro_ros import RosNode
import time

if __name__ == '__main__':
    ros_node = RosNode()
    while True:
        robot_state = ros_node.allegro_joint_pos.float()
        thumb_state = robot_state[-4:]
        print(thumb_state)
        time.sleep(0.1)