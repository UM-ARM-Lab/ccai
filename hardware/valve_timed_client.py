"""Valve-only actionlib adapter; legacy Allegro command modes are unchanged."""
import math
import time


class ValveTimedClient:
    def __init__(self, node, endpoint="/allegroHand/execute_valve_action", timeout_s=2.0):
        import actionlib
        import rospy
        from allegro_hand_controllers.msg import ExecuteValveActionAction, ExecuteValveActionGoal
        self.node, self.rospy, self.goal_type = node, rospy, ExecuteValveActionGoal
        self.timeout_s = float(timeout_s)
        self.client = actionlib.SimpleActionClient(endpoint, ExecuteValveActionAction)
        if not self.client.wait_for_server(rospy.Duration(self.timeout_s)):
            raise RuntimeError("Valve timed-repeat controller unavailable; rebuild/source controller workspace")

    def execute(self, target, repeats, duration_s):
        values = [float(v) for v in target]
        if len(values) != 16 or not all(math.isfinite(v) for v in values):
            raise ValueError("Expected 16 finite canonical joint targets")
        if isinstance(repeats, bool) or int(repeats) != repeats or not 1 <= repeats <= 1000000:
            raise ValueError("Invalid Valve repeats")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Invalid Valve action duration")
        goal = self.goal_type()
        goal.target = values
        goal.repeats, goal.duration_s = int(repeats), float(duration_s)
        self.client.send_goal(goal)
        deadline = time.monotonic() + duration_s + self.timeout_s
        while self.client.get_state() in (0, 1, 6, 7):
            if self.rospy.is_shutdown() or time.monotonic() >= deadline:
                self.client.cancel_goal()
                raise RuntimeError("Valve action completion timed out; no next action will be sent")
            time.sleep(0.002)
        if self.client.get_state() != 3:
            raise RuntimeError("Valve action interrupted: " + self.client.get_goal_status_text())
        result = self.client.get_result()
        if result is None or not math.isfinite(result.elapsed_s) or result.elapsed_s < duration_s:
            raise RuntimeError("Invalid Valve completion result")
        # Require a subsequently received state with a source stamp after completion.
        before = self.node._joint_state_seq
        deadline = time.monotonic() + self.timeout_s
        while True:
            pose = self.node.current_joint_pose
            if (self.node._joint_state_seq > before and pose is not None and
                    pose.header.stamp >= result.completed_at):
                break
            if self.rospy.is_shutdown() or time.monotonic() >= deadline:
                raise RuntimeError("No fresh post-completion Allegro state")
            time.sleep(0.002)
        return result
