import numpy as np
import pickle as pkl
import pathlib
import sys
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(CCAI_PATH))
from card.card_problem import init_env
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import time
import torch

visualize = True
config, env, sim_env, ros_copy_node, chain, sim, gym, viewer = init_env(visualize=visualize)
trajs = pkl.load(open(fpath / 'card' / 'full_trajs.pkl', 'rb'))

for i in range(len(trajs)):
    print(f'showing traj {i}')
    traj = trajs[i]
    for j in range(27):
        pos = traj[j]
        env.reset(dof_pos=pos)
        time.sleep(0.2)