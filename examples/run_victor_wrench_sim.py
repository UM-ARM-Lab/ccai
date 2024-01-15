import numpy as np
from isaac_victor_envs.tasks.victor_wrench import VictorWrenchEnv
from isaac_victor_envs.utils import get_assets_dir
from ccai.mpc import Constrained_SVGD_MPC, IpoptMPC, SVMPC, MPPI, SQPMPC, iCEM

import torch
import time
import yaml
import pathlib
import pytorch_kinematics as pk

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
from isaac_victor_envs.utils import get_assets_dir

from victor_wrench import VictorWrenchProblem, VictorWrenchIpoptProblem, VictorUnconstrainedPenaltyProblem, \
    VictorWrenchSQPProblem


def do_trial(env, params, fpath):
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    start = torch.cat((state['q'], state['offset'], state['theta']), dim=1).to(device=params['device']).reshape(9)
    start = start.to(device=params['device'], dtype=torch.float32)
    goal = torch.tensor([
        [-2 * torch.pi / 4.0],
    ], device=params['device'])

    wrench_length = 0.2
    wrench_centre = [0.6, 0.15, 0.975]
    if 'csvgd' in params['controller']:
        problem = VictorWrenchProblem(start, goal, params['T'], wrench_centre=wrench_centre,
                                      wrench_length=wrench_length, device=params['device'],
                                      chain=params['chain'])
        controller = Constrained_SVGD_MPC(problem, params)
    elif 'ipopt' in params['controller']:
        problem = VictorWrenchIpoptProblem(start, goal, params['T'], wrench_centre=wrench_centre,
                                           wrench_length=wrench_length,
                                           chain=params['chain'])
        controller = IpoptMPC(problem, params)
    elif 'sqp' in params['controller']:
        problem = VictorWrenchSQPProblem(start, goal, params['T'], wrench_centre=wrench_centre,
                                         wrench_length=wrench_length,
                                         chain=params['chain'])
        controller = SQPMPC(problem, params)
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
    elif 'icem' in params['controller']:
        problem = VictorUnconstrainedPenaltyProblem(start, goal, params['T'],
                                                    wrench_centre=wrench_centre,
                                                    wrench_length=wrench_length,
                                                    chain=params['chain'],
                                                    penalty=params['penalty'])
        controller = iCEM(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    duration = 0
    initial_duration = 0
    planned_trajectories = []
    for k in range(params['num_steps']):
        # print(k)
        state = env.get_state()
        state = torch.cat((state['q'], state['offset'], state['theta']), dim=1).to(device=params['device']).reshape(9)
        actual_trajectory.append(state)
        # if k > 0:
        torch.cuda.synchronize()
        start_time = time.time()
        best_traj, trajectories = controller.step(state)

        if torch.any(torch.isnan(best_traj)):
            print('NAN in best traj')
            break

        planned_trajectories.append(trajectories)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time
        else:
            initial_duration = time.time() - start_time

        M = len(trajectories)
        K = len(trajectories[0])
        # add trajectories to plot
        traj_line_colors = np.random.random((1, M)).astype(np.float32)
        ee_mat = params['chain'].forward_kinematics(trajectories[:, :, :7].reshape(-1, 7)).get_matrix().reshape(M, K, 4,
                                                                                                                4)
        ee_pos = ee_mat[:, :, :3, 3]

        for e in env.envs:
            for t in range(K - 1):
                p = torch.stack((ee_pos[:, t], ee_pos[:, t + 1]), dim=1).reshape(2 * M,
                                                                                 3).cpu().numpy()
                p[:, 2] += 0.005
                gym.add_lines(viewer, e, M, p, traj_line_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

        x = best_traj[0, :7]
        env.step(x.reshape(1, 7).to(device=env.device))
        gym.clear_lines(viewer)

    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    state = env.get_state()
    state = torch.cat((state['q'], state['offset'], state['theta']), dim=1).to(device=params['device']).reshape(9)
    actual_trajectory.append(state)
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)
    planned_trajectories = [plan.cpu().numpy() for plan in planned_trajectories]
    problem.T = len(actual_trajectory)
    g = problem._con_eq(actual_trajectory.unsqueeze(0), False)[0].reshape(-1)
    # h = problem._con_ineq(actual_trajectory.unsqueeze(0), False)[0].reshape(-1)
    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             g=g.cpu().numpy(),
             initial_time=initial_duration,
             online_time=duration / (params['num_steps'] - 1),
             )
    final_distance_to_goal = (actual_trajectory[-1, -1] - goal[0, 0]).abs().item()
    print(f'Final distance to goal: {final_distance_to_goal}')
    # return final wrench angle error
    return final_distance_to_goal


if __name__ == "__main__":

    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/config/victor_wrench.yaml').read_text())
    from tqdm import tqdm

    asset = f'{get_assets_dir()}/victor/victor_grippers.urdf'
    ee_name = 'l_palm'

    chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
    config['chain'] = chain

    # instantiate environment
    env = VictorWrenchEnv(1, control_mode='joint_impedance',
                          viewer=config['visualize'])
    sim, gym, viewer = env.get_sim()

    results = {}

    for i in tqdm(range(config['num_trials'])):
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            final_distance_to_goal = do_trial(env, params, fpath)
            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
