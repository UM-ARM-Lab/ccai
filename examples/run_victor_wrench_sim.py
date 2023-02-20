import numpy as np
from isaac_victor_envs.tasks.victor_wrench import VictorWrenchEnv
from isaac_victor_envs.utils import get_assets_dir
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
import torch
import time
import pytorch_kinematics as pk

from victor_wrench import VictorWrenchProblem

if __name__ == "__main__":

    DEVICE = 'cuda:0'
    asset = f'{get_assets_dir()}/victor/victor_grippers.urdf'
    ee_name = 'l_palm'

    chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
    chain.to(device=DEVICE)

    wrench_length = 0.2
    wrench_centre = [0.6, 0.15, 0.975]

    M = 4
    T = 10
    env = VictorWrenchEnv(M, control_mode='joint_impedance')
    sim, gym, viewer = env.get_sim()
    state = env.get_state()
    q, theta = state['q'], state['theta']

    start = torch.cat((q, theta), dim=1)
    try:
        while True:
            q, theta = state['q'], state['theta']
            start = torch.cat((q, torch.zeros_like(theta), theta), dim=1)
            env.step(q)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    device = DEVICE
    goal = torch.tensor([
        [-2 * torch.pi / 4.0],
    ], device=device)

    particles = 0.1 * torch.randn(M, T, 9, device=device)
    particles = torch.cumsum(particles, dim=1) + start.reshape(M, 1, 9)
    trajectory = particles

    for k in range(T):
        state = env.get_state()
        start = state['q']

        problem = VictorWrenchProblem(start=start[0], goal=goal, T=T - k,
                                      wrench_centre=wrench_centre,
                                      wrench_length=wrench_length,
                                      device=device,
                                      chain=chain)

        solver = ConstrainedSteinTrajOpt(problem, dt=0.1, alpha_C=1, alpha_J=0.5)

        if k == 0:
            solver.iters = 200
        else:
            solver.iters = 10
        s = time.time()
        trajectory = solver.solve(trajectory)
        torch.cuda.synchronize()
        e = time.time()
        print(e - s)
        # input('Optimization finished')

        # add goal lines to sim
        # trajectory_colors
        # traj_line_colors = np.array([[0.5, 0., 0.5]*M], dtype=np.float32)
        traj_line_colors = np.random.random((1, M)).astype(np.float32)
        ee_mat = chain.forward_kinematics(trajectory[:, :, :7].reshape(-1, 7)).reshape(M, T - k, 4, 4)
        ee_pos = ee_mat[:, :, :3, 3]

        for e in env.envs:
            # p = torch.stack((start[:, :3], trajectory[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
            # p[:, 2] += 0.005
            # gym.add_lines(viewer, e, M, p, traj_line_colors)
            for t in range(T - 1 - k):
                p = torch.stack((ee_pos[:, t], ee_pos[:, t + 1]), dim=1).reshape(2 * M,
                                                                                 3).cpu().numpy()
                p[:, 2] += 0.005
                gym.add_lines(viewer, e, M, p, traj_line_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

        costs = problem.cost(trajectory)
        print(costs)
        idx = torch.argmin(costs)
        x = trajectory[idx, 0, :7].reshape(1, -1).repeat(M, 1)
        env.step(x)
        trajectory = trajectory[:, 1:]

        gym.clear_lines(viewer)

    while not gym.query_viewer_has_closed(viewer):
        time.sleep(0.1)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
