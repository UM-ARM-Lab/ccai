from screwdriver_problem import init_env
import numpy as np
import pickle as pkl
import time
import pathlib
import matplotlib.pyplot as plt

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


if __name__ == "__main__":

    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)

    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    with open(f'{fpath.resolve()}/initial_poses/initial_poses_10k.pkl', 'rb') as file:
        initial_poses  = pkl.load(file)
    
    for i in range(len(initial_poses)):
        env.reset(initial_poses[i], deterministic=True)
        time.sleep(0.1)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)