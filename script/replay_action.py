import pickle
from utils.allegro_utils import partial_to_full_state
from hardware.hardware_env import RosNode
import rospy
if __name__ == "__main__":
    ros_copy_node = RosNode()
    with open('/home/fanyang/github/ccai/data/action.pkl', 'rb') as f:
        actions = pickle.load(f)
    fingers = ['index', 'middle', 'thumb']
    for i, action in enumerate(actions):
        if i == 4:
            print("start turning")
        ros_copy_node.apply_action(partial_to_full_state(action, fingers))
        rospy.sleep(1)