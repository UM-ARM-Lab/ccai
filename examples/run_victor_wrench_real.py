import numpy as np
np.float = np.float64
import rospy
from arc_utilities import ros_init
from arm_robots.victor import Victor
from victor_hardware_interface_msgs.msg import ControlMode
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.ipopt import IpoptMPC
from ccai.mpc.svgd import SVMPC
from ccai.mpc.mppi import MPPI
import torch
import yaml

import pytorch_kinematics as pk
import pathlib

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
from isaac_victor_envs.utils import get_assets_dir

from victor_wrench import VictorWrenchProblem, VictorWrenchIpoptProblem, VictorUnconstrainedPenaltyProblem


def stop_condition(feedback):
    time_error = (feedback.actual.time_from_start - feedback.desired.time_from_start).secs
    if time_error > 1:
        print('Timeout')
        return True


def robot_get_state(robot, chain, wrench_centre):
    q = robot.get_joint_positions()[:7]
    # compute theta?
    q = torch.tensor(q).to(device=chain.device, dtype=torch.float32)
    ee_pos = chain.forward_kinematics(q).get_matrix().reshape(4, 4)[:3, 3]
    theta = - torch.atan2(ee_pos[0] - wrench_centre[0], ee_pos[1] - wrench_centre[1])
    print(theta)
    state = torch.cat((q, torch.zeros(1, device=chain.device), theta.reshape(-1)), dim=0).reshape(9)
    return state


def robot_set_state(robot, state):
    q = state[:7].detach().cpu().numpy().astype(np.float64)
    print('Desired state')
    print(q)
    robot.plan_to_joint_config(robot.left_arm_group, q.tolist(), stop_condition=stop_condition)
    print('Achieved state:')
    print(robot.get_joint_positions()[:7])

def do_trial(robot, params, fpath):
    start = robot_get_state(robot, params['chain'], [0, 0, 0])
    goal = torch.tensor([
        [-2 * torch.pi / 4.0],
    ], device=params['device'])

    # get wrench configs
    start_ee_pos = params['chain'].forward_kinematics(start[:7]).get_matrix().reshape(4, 4)[:3, -1]
    wrench_length = 0.13
    wrench_centre = [start_ee_pos[0].item(), start_ee_pos[1].item() - wrench_length, start_ee_pos[2].item()]

    if params['controller'] == 'csvgd':
        problem = VictorWrenchProblem(start, goal, params['T'], wrench_centre=wrench_centre,
                                      wrench_length=wrench_length, device=params['device'],
                                      chain=params['chain'])
        controller = Constrained_SVGD_MPC(problem, params)
    elif params['controller'] == 'ipopt':
        problem = VictorWrenchIpoptProblem(start, goal, params['T'], wrench_centre=wrench_centre,
                                           wrench_length=wrench_length,
                                           chain=params['chain'])
        controller = IpoptMPC(problem, params)
    elif 'mppi' in params['controller']:
        problem = VictorUnconstrainedPenaltyProblem(start, goal, params['T'],
                                                    wrench_centre=wrench_centre,
                                                    wrench_length=wrench_length,
                                                    chain=params['chain'],
                                                    penalty=params['penalty'])
        controller = MPPI(problem, params)
    elif 'svgd' in params['controller']:
        problem = VictorUnconstrainedPenaltyProblem(start, goal, params['T'],
                                                    wrench_centre=wrench_centre,
                                                    wrench_length=wrench_length,
                                                    chain=params['chain'],
                                                    penalty=params['penalty'])
        controller = SVMPC(problem, params)
    else:
        raise ValueError('Invalid controller')

    y = input('Ready to go? (y/n)')
    if y != 'y':
        exit(0)

    ref_traj = None
    for _ in range(5):
        goal = torch.tensor([
            [-torch.pi / 2.0],
        ], device=params['device'])

        # reset controller with new goal
        state = robot_get_state(robot, params['chain'], wrench_centre).to(device=params['device'])
        controller.reset(state, goal=goal, initial_x=ref_traj, T=params['T'])
        actual_trajectory = []

        for k in range(params['num_steps']):
            print(k)
            state = robot_get_state(robot, params['chain'], wrench_centre).to(device=params['device'])
            actual_trajectory.append(state)

            # compute trajectories
            best_traj, trajectories = controller.step(state)

            if ref_traj is None:
                ref_traj = trajectories.clone()
                #print(best_traj)
                #exit(0)

            # Safety check
            if torch.any(torch.isnan(trajectories)):
                print('Got NAN, exiting')
                exit(0)
            # send to robot
            robot_set_state(robot, best_traj[0])

        goal = torch.tensor([
            [0.0],
        ], device=params['device'])

        # reset controller with new goal
        state = robot_get_state(robot, params['chain'], wrench_centre).to(device=params['device'])
        print(ref_traj.shape)
        controller.reset(state, goal=goal, initial_x=torch.flip(ref_traj, dims=(1,)), T=params['T'])

        for k in range(params['num_steps']):
            state = robot_get_state(robot, params['chain'], wrench_centre).to(device=params['device'])
            actual_trajectory.append(state)

            # compute trajectories
            best_traj, trajectories = controller.step(state)

            # Safety check
            if torch.any(torch.isnan(trajectories)):
                print('Got NAN, exiting')
                exit(0)

            # send to robot
            robot_set_state(robot, best_traj[0])

        state = robot_get_state(robot, params['chain'], wrench_centre).to(device=params['device'])

        actual_trajectory.append(state)
        actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy())
    # return final wrench angle error
    return (actual_trajectory[-1, -1] - goal[0, 0]).abs().item()


@ros_init.with_ros("cstvo_wrench")
def main():
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/config/victor_wrench_real.yaml').read_text())
    from tqdm import tqdm

    asset = f'{get_assets_dir()}/victor/victor_grippers.urdf'
    ee_name = 'l_palm'

    chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
    config['chain'] = chain

    # Initialize robot interface
    victor = Victor()
    victor.set_control_mode(control_mode=ControlMode.JOINT_IMPEDANCE, vel=0.05)
    victor.connect()
    rospy.sleep(1)

    results = {}

    # robot_set_state(victor, torch.tensor([2.9501, -0.2883, -2.5463, -1.5888, -1.6216, -1.1591, 0.0]))
    # exit(0)
    for i in tqdm(range(config['num_trials'])):
        for controller in config['controllers'].keys():
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['chain'] = config['chain'].to(device=params['device'])
            final_distance_to_goal = do_trial(victor, params, fpath)
            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)


if __name__ == "__main__":
    main()
