from isaac_victor_envs.tasks.victor import VictorPuckObstacleEnv2, VictorPuckObstacleEnv3, orientation_error, quat_change_convention, \
    VictorFloatingSpheresTableEnv, VictorPuckObstacleEnv
from victor_table_surface_jointspace import do_trial


import torch
import yaml
import pathlib
from isaacgym.torch_utils import quat_apply

import pytorch_kinematics as pk
from isaac_victor_envs.utils import get_assets_dir

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
goal_offsets = torch.tensor([[-9.5686e-01, 3.0024e-01, -1.2078e+00],
                             [2.1461e-01, -6.2871e-01, -5.9032e-01],
                             [1.0945e-01, 2.7702e-01, 8.6784e-01],
                             [2.1709e+00, -7.8493e-01, 7.4112e-02],
                             [4.0734e-02, -8.4159e-01, -5.4897e-02],
                             [8.1381e-01, 3.9453e-01, -1.5701e-01],
                             [-1.1987e+00, -5.3608e-01, -1.0481e-01],
                             [1.1764e+00, -2.6621e-02, 4.0178e-01],
                             [-7.2742e-01, 1.1604e+00, -6.1000e-01],
                             [2.2506e+00, -1.3803e+00, -8.2207e-01],
                             [7.4789e-01, -8.3746e-01, -8.4314e-01],
                             [2.0148e-01, -2.0982e-01, -1.3465e+00],
                             [5.3815e-01, -9.4940e-01, 2.6272e+00],
                             [1.3516e+00, 7.6650e-01, -8.8203e-01],
                             [4.7398e-02, -5.0036e-01, 1.5218e-03],
                             [-2.2953e+00, -1.2354e+00, 6.4950e-01],
                             [1.1182e+00, 9.5924e-01, 7.4377e-01],
                             [-7.9049e-01, -9.1182e-01, 5.7457e-01],
                             [1.0378e+00, -1.3220e-01, -8.9577e-01],
                             [2.0970e-01, 5.2175e-01, 5.8444e-01]]
                            )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # get config
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/victor_table_jointspace.yaml').read_text())
    from tqdm import tqdm

    # instantiate environment
    if config['include_obstacles']:
        if config['obstacle_type'] == 'tabletop_ycb':
            env = VictorPuckObstacleEnv2(1, control_mode='joint_impedance',
                                         viewer=config['visualize'])
        if config['obstacle_type'] == 'tabletop_ycb2':
            env = VictorPuckObstacleEnv3(1, control_mode='joint_impedance',
                                         viewer=config['visualize'])
        elif 'floating_spheres' in config['obstacle_type']:
            env = VictorFloatingSpheresTableEnv(1, control_mode='joint_impedance',
                                                viewer=config['visualize'], obs_name=config['obstacle_type'])
    else:
        env = VictorPuckObstacleEnv(1, control_mode='joint_impedance',
                                    viewer=config['visualize'], randomize_start=True,
                                    randomize_obstacles=config['random_env'])

    sim, gym, viewer = env.get_sim()

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
        # table_height = None if config['include_table'] else 0.1
        # obstacles_1 = None if config['include_obstacles'] else [3, 3]
        # obstacles_2 = None if config['include_obstacles'] else [-3, -3]
        # env.reset(table_height, obstacles_1, obstacles_2, start_on_table=config['include_table'])
        if not config['include_obstacles']:
            env.reset(table_height=config['table_height'], start_on_table=config['include_table'],
                      obstacles_1=[3, 3], obstacles_2=[-3, -3])
        else:
            env.reset(table_height=config['table_height'])
        # set goal
        if config['random_env']:
            ct = 0
            goal = torch.tensor([0.45, -0.1]) + torch.rand(2) * torch.tensor([0.5, 0.7])
            state = env.get_state()
            obs1, obs2 = state['obs1_pos'][0, :2].cpu(), state['obs2_pos'][0, :2].cpu()
            while (torch.linalg.norm(goal - obs1) < 0.1 or
                   torch.linalg.norm(goal - obs2) < 0.1):
                goal = torch.tensor([0.45, -0.1]) + torch.rand(2) * torch.tensor([0.5, 0.7])
                ct += 1
                if ct > 100:
                    break
        else:
            if config['obstacle_type'] == 'tabletop_ycb':
                goal = torch.tensor([0.8, 0.1]) + 0.05 * goal_offsets[i, :2]
            elif config['obstacle_type'] == 'tabletop_ycb2':
                goal = torch.tensor([0.9, 0.15]) + 0.05 * goal_offsets[i, :2]
            elif config['obstacle_type'] == 'floating_spheres_1':
                goal = torch.tensor([0.65, 0.05])
                goal = goal + 0.05 * goal_offsets[i, :2]  # torch.tensor([0.25, 0.1]) * torch.rand(2)
            elif config['obstacle_type'] == 'floating_spheres_2':
                goal = torch.tensor([0.75, 0.05]) + 0.025 * goal_offsets[i, :2]
            elif config['obstacle_type'] == 'floating_spheres_5':
                goal = torch.tensor([0.85, 0.0]) + 0.025 * goal_offsets[i, :2]
        g = torch.zeros(3)
        g[:2] = goal
        g[2] = env.table_height
        goal = g
        for controller in config['controllers'].keys():
            # env.reset()
            env.reset_arm_only()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['goal'] = goal.to(device=params['device'])
            print(goal)
            final_distance_to_goal = do_trial(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
