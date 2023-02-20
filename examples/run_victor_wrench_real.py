import numpy as np
import rospy
from arc_utilities import ros_init
from arm_robots.victor import Victor
from victor_hardware_interface_msgs.msg import ControlMode
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt

import torch
import pytorch_kinematics as pk
from victor_wrench import VictorWrenchProblem
from isaac_victor_envs.utils import get_assets_dir


def stop_condition(feedback):
    time_error = (feedback.actual.time_from_start - feedback.desired.time_from_start).secs
    if time_error > 0.05:
        return True


@ros_init.with_ros("Constrained_SVGDTrajOpt_victor_wrench")
def main():
    M = 4
    T = 10
    dt = 0.1
    online_iters = 5
    device = 'cuda:0'
    asset = f'{get_assets_dir()}/victor/victor_gripper.urdf'
    ee_name = 'l_palm'
    victor_dt = 2
    alpha_J = 0.5
    chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
    chain.to(device=device)

    num_cranks = 1
    victor = Victor()
    victor.set_control_mode(control_mode=ControlMode.JOINT_IMPEDANCE, vel=0.05)
    victor.connect()
    rospy.sleep(1)
    start = victor.get_joint_positions()[:7]
    start.append(0.0)
    start.append(0.0)

    start = torch.tensor(start).to(device=device, dtype=torch.float32).repeat(M, 1)
    start_ee_pos = chain.forward_kinematics(start[0, :7]).reshape(4, 4)[:3, -1]

    wrench_length = 0.13
    wrench_centre = [start_ee_pos[0].item(), start_ee_pos[1].item() - wrench_length, start_ee_pos[2].item()]

    goal = torch.tensor([
        [-2 * torch.pi / 4.0],
    ], device=device)

    particles = 0.1 * torch.randn(M, T, 9, device=device)
    particles = torch.cumsum(particles, dim=1) + start.reshape(M, 1, 9)
    trajectory = particles

    for crank in range(num_cranks):
        # first plan ahead
        problem = VictorWrenchProblem(start=start[0, :7], goal=goal, T=T,
                                      wrench_centre=wrench_centre,
                                      wrench_length=wrench_length, device=device,
                                      chain=chain)

        solver = ConstrainedSteinTrajOpt(problem, dt=dt, alpha_C=1, alpha_J=alpha_J)
        if crank == 0:
            solver.iters = 200
        else:
            solver.iters = online_iters
        trajectory = solver.solve(trajectory)
        if torch.any(torch.isnan(trajectory)):
            print('Got NAN, exiting')
            exit(0)

        if crank == 0:
            y = input('got trajectory, waiting for OK to execute, y for yes, n for no')
            if y != 'y':
                exit(0)

        partial_trajectory = trajectory.clone()
        for t in range(T):
            start = victor.get_joint_positions()[:7]
            start = torch.tensor(start).to(device=device, dtype=torch.float32).repeat(M, 1)
            problem = VictorWrenchProblem(start=start[0, :7], goal=goal, T=T - t,
                                          wrench_centre=wrench_centre,
                                          wrench_length=wrench_length, device=device,
                                          chain=chain)
            solver = ConstrainedSteinTrajOpt(problem, dt=dt, alpha_C=1, alpha_J=alpha_J)
            solver.iters = online_iters
            partial_trajectory = solver.solve(partial_trajectory)
            costs = problem.cost(partial_trajectory)
            idx = torch.argmin(costs)

            if torch.any(torch.isnan(partial_trajectory)):
                print('Got NAN, exiting')
                exit(0)

            # send to robot
            q = partial_trajectory[idx, 0, :7].detach().cpu().numpy().astype(np.float64)
            victor.plan_to_joint_config(victor.left_arm_group, q.tolist(), stop_condition=stop_condition)
            if t < T - 1:
                partial_trajectory = partial_trajectory[:, 1:]

        partial_trajectory = torch.flip(trajectory.clone(), dims=(1,))

        for t in range(T):
            start = victor.get_joint_positions()[:7]
            start = torch.tensor(start).to(device=device, dtype=torch.float32).repeat(M, 1)
            problem = VictorWrenchProblem(start=start[0, :7], goal=0 * goal, T=T - t,
                                          wrench_centre=wrench_centre,
                                          wrench_length=wrench_length, device=device,
                                          chain=chain)
            solver = ConstrainedSteinTrajOpt(problem, dt=dt, alpha_C=1, alpha_J=0.5)
            solver.iters = online_iters
            partial_trajectory = solver.solve(partial_trajectory)
            costs = problem.cost(partial_trajectory)
            idx = torch.argmin(costs)
            if torch.any(torch.isnan(partial_trajectory)):
                print('Got NAN, exiting')
                exit(0)
            # send to robot
            q = partial_trajectory[idx, 0, :7].detach().cpu().numpy().astype(np.float64)
            victor.plan_to_joint_config(victor.left_arm_group, q.tolist(), stop_condition=stop_condition)
            if t < T - 1:
                partial_trajectory = partial_trajectory[:, 1:]

    exit(0)


if __name__ == "__main__":
    main()
