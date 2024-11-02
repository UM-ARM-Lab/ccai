from screwdriver_problem import init_env, pregrasp, emailer
import numpy as np
import pickle as pkl
import pathlib
import sys
from tqdm import tqdm
import torch

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))

if __name__ == "__main__":

    initial_poses = []
    fpath = pathlib.Path(f'{CCAI_PATH}/data')
    params, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)
    
    batch = 0
    try:
        while True:
            initial_poses = []
            for i in range(10000):
                print(f"Generating pose {i}")
                pose = pregrasp(env, params, chain)
                initial_poses.append(pose)

            with open(f'{fpath.resolve()}/initial_poses/initial_poses_{batch}.pkl', 'wb') as f:
                pkl.dump(initial_poses, f)

            batch += 1

    except KeyboardInterrupt:
        print("Process interrupted. Cleaning up and saving current state.")

    finally:
        print("Saved poses")
        emailer().send()
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
