import numpy as np
np.float = np.float64
from isaac_victor_envs.utils import get_assets_dir
import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, UnconstrainedPenaltyProblem, IpoptProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
from ccai.mpc.ipopt import IpoptMPC
from ccai.mpc.diffusion_mpc import Diffusion_MPC
import time
import pytorch_kinematics as pk

from quadrotor_learn_to_sample import TrajectorySampler
from ccai.models.constraint_embedding.vae import Conv3DEncoder
from ccai.models.helpers import SinusoidalPosEmb

from victor_table_surface_jointspace import VictorTableProblem
import rospy
from arc_utilities import ros_init
from arm_robots.victor import Victor
from victor_hardware_interface_msgs.msg import ControlMode

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset_dir = get_assets_dir()
asset = asset_dir + '/victor/victor_mallet.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
chain_cc = pk.build_chain_from_urdf(open(asset).read())
collision_check_links = [
    'victor_left_arm_link_2',
    'victor_left_arm_link_3',
    'victor_left_arm_link_4',
    'victor_left_arm_link_5',
    'victor_left_arm_link_6',
    'victor_left_arm_link_7',
    'victor_left_arm_striker_base',
    'victor_left_arm_striker_mallet'
]


def stop_condition(feedback):
    time_error = (feedback.actual.time_from_start - feedback.desired.time_from_start).secs
    if time_error > 0.05:
        print('Timeout')
        return True


def robot_get_state(robot):
    q = robot.get_joint_positions()[:7]
    q = torch.tensor(q).to(device=chain.device, dtype=torch.float32).reshape(1, -1)
    return q


def robot_set_state(robot, state):
    q = state[:7].detach().cpu().numpy().astype(np.float64)
    print('Desired state')
    print(q)
    robot.plan_to_joint_config(robot.left_arm_group, q.tolist(), stop_condition=stop_condition)
    print('Achieved state:')
    print(robot.get_joint_positions()[:7])


def do_trial(robot, params, fpath):
    start = robot_get_state(robot)

    chain.to(device=params['device'])
    chain_cc.to(device=params['device'])

    # assume it starts at the table height
    ee_pos = chain.forward_kinematics(start).get_matrix()[0, :3, 3]
    table_height = ee_pos[2]

    flow_model = None
    env_encoder = None
    height_encoder = None
    constr_params = None
    if 'csvgd' in params['controller'] or 'diffmpc' in params['controller']:
        if params['flow_model'] != 'none':
            if 'diffusion' in params['flow_model']:
                flow_type = 'diffusion'
            elif 'cnf' in params['flow_model']:
                flow_type = 'cnf'
            else:
                raise ValueError('Invalid flow model type')

            if 'obstacle' in params['controller']:
                env_encoder = Conv3DEncoder(64)
                env_encoder.load_state_dict(torch.load(f'{CCAI_PATH}/{params["flow_model"]}')['env_encoder'])
                env_encoder.to(device=params['device'])
            if 'table' in params['controller']:
                height_encoder = SinusoidalPosEmb(64)
                height_encoder.to(device=params['device'])

            # parameterize constraints
            constr_params = []
            constr_codes = []
            if 'floating_spheres' in config['obstacle_type'] or config['obstacle_type'] == 'tabletop_ycb':
                sdf_grid = np.load(f'{CCAI_PATH}/{config["obstacle_type"]}.npz')['sdf_grid']
                sdf_grid = torch.from_numpy(sdf_grid).to(device=params['device']).reshape(1, 1, 64, 64, 64).to(
                    dtype=torch.float32)
                if env_encoder is not None:
                    env_embedding = env_encoder(sdf_grid)
                    constr_params.append(env_embedding)
                    constr_codes.append(torch.tensor([[1.0, 0.0]]).to(device=params['device']))
            if height_encoder is not None:
                htensor = torch.tensor([table_height]).to(device=params['device']).reshape(-1)
                height_embedding = height_encoder(htensor)
                constr_params.append(height_embedding)
                constr_codes.append(torch.tensor([[0.0, 1.0]]).to(device=params['device']))
            if len(constr_params) > 0:
                constr_params = torch.stack(constr_params, dim=0).reshape(1, -1, 64)
                constr_codes = torch.stack(constr_codes, dim=0).reshape(1, -1, 2)
                constr_params = torch.cat((constr_params, constr_codes), dim=-1)

        obstacle_poses = {
            'pitcher': torch.tensor([[1.0, 0.0, 0.75],
                                      [0.0, 1.0, 0.0, 0.25],
                                      [0.0, 0.0, 1.0, table_height],
                                      [0.0, 0.0, 0.0, 1.0]], device=params['device']),
            'mustard_bottle': torch.tensor([[1.0, 0.0, 0.75],
                                      [0.0, 1.0, 0.0, 0.55],
                                      [0.0, 0.0, 1.0, table_height],
                                      [0.0, 0.0, 0.0, 1.0]], device=params['device']),
            'cracker_box': torch.tensor([[1.0, 0.0, 0.55],
                                      [0.0, 1.0, 0.0, 0.25],
                                      [0.0, 0.0, 1.0, table_height],
                                      [0.0, 0.0, 0.0, 1.0]], device=params['device'])
        }

        problem = VictorTableProblem(start, params['goal'], params['T'], device=params['device'],
                                     obstacle_poses=obstacle_poses,
                                     table_height=table_height,
                                     obstacle_type=params['obstacle_type'],
                                     flow_model=None,
                                     constr_params=constr_params)

        problem.robot_scene.visualize_robot(start)
        exit(0)

        if 'diffmpc' in params['controller']:
            flow_problem = problem
            constrain = params['constrained']
        else:
            flow_problem = None
            constrain = None

        if params['flow_model'] != 'none':
            flow_model = TrajectorySampler(T=params['T'] + 1, dx=7, du=0, context_dim=7 + 3 + 64 + 2, type=flow_type,
                                           timesteps=params['timesteps'], hidden_dim=params['hidden_dim'],
                                           problem=flow_problem, constrain=constrain,
                                           unconditional=params['unconditional'])

            flow_model.load_state_dict(torch.load(f'{CCAI_PATH}/{params["flow_model"]}')['ema'])
            flow_model.to(device=params['device'])
            flow_model.send_norm_constants_to_submodels()

            params['flow_model'] = flow_model
        else:
            params['flow_model'] = None

        if 'csvgd' in params['controller']:
            controller = Constrained_SVGD_MPC(problem, params)
        else:
            controller = Diffusion_MPC(problem, params)

    else:
        raise ValueError('invalid controller type')

    actual_trajectory = []
    planned_trajectories = []
    duration = 0

    for k in range(params['num_steps']):
        state = robot_get_state(robot, params['chain'], wrench_centre).to(device=params['device'])

        actual_trajectory.append(start.clone())
        if k > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        best_traj, trajectories = controller.step(start, constr_params)
        planned_trajectories.append(trajectories)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time

        x = best_traj[0, :7]

    state = env.get_state()
    # obs1_pos = state['obs1_pos'][0, :2]
    # obs2_pos = state['obs2_pos'][0, :2]
    state = state['q'].reshape(7).to(device=params['device'])

    # obs = torch.stack((obs1_pos, obs2_pos), dim=0).cpu().numpy()
    # if not params['include_obstacles']:
    #    obs = None
    obs = None
    actual_trajectory.append(state.clone())

    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 7)
    planned_trajectories = torch.stack(planned_trajectories, dim=0)

    problem.T = actual_trajectory.shape[0]
    if params['include_table']:
        constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0).cpu().numpy()
    else:
        constraint_val = None

    if params['include_obstacles']:
        obs_constraint_val = problem._con_ineq(actual_trajectory.unsqueeze(0), False)[0].squeeze(0).cpu().numpy()
    else:
        obs_constraint_val = None

    final_distance_to_goal = torch.linalg.norm(
        chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).get_matrix().reshape(-1, 4, 4)[:, :3, 3] -
        params[
            'goal'].unsqueeze(0),
        dim=1
    )
    # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    # print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    if params['visualize']:
        env.gym.write_viewer_image_to_file(env.viewer, f'{env.frame_fpath}/frame_{env.frame_id + 1:06d}.png')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val,
             d2goal=final_distance_to_goal.cpu().numpy(),
             traj=planned_trajectories.cpu().numpy(),
             obs=obs,
             height=table_height,
             goal=params['goal'].cpu().numpy(),
             obs_constr=obs_constraint_val,
             )
    return torch.min(final_distance_to_goal).cpu().numpy()

@ros_init.with_ros("cstvo_table")
def main():
    torch.set_float32_matmul_precision('high')
    # get config
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/planning_configs/victor_table_jointspace.yaml').read_text())
    config['obstacle_type'] = 'tabletop_ycb_2'
    from tqdm import tqdm

    # Initialize robot interface
    victor = Victor()
    victor.set_control_mode(control_mode=ControlMode.JOINT_IMPEDANCE, vel=0.05)
    victor.connect()
    rospy.sleep(1)

    # instantiate environment


    """
    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    try:
        while True:
            start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(1, 7)
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    """
    results = {}

    for i in tqdm(range(config['num_trials']), initial=config['start_trial']):
        i += config['start_trial']
        goal = torch.tensor([0.85, 0.0]) + 0.025 * torch.randn(2)
        g = torch.zeros(3)
        g[:2] = goal
        goal = g
        for controller in config['controllers'].keys():
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['goal'] = goal.to(device=params['device'])
            print(goal)
            final_distance_to_goal = do_trial(victor, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        # print(results)

if __name__ == "__main__":
    main()
