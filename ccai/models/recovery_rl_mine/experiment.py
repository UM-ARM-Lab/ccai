import datetime
# import gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import moviepy.editor as mpy
# import cv2
from torch import nn, optim

from recovery_rl.replay_memory import ReplayMemory, ConstraintReplayMemory
from recovery_rl.model import VisualEncoderAttn, TransitionModel, VisualReconModel
from recovery_rl.utils import linear_schedule, recovery_config_setup

from env.make_utils import register_env, make_env

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x): return torch.FloatTensor(x).to('cuda')


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


# Process observation for CNN
def process_obs(obs, env_name):
    if 'extraction' in env_name:
        obs = cv2.resize(obs, (64, 48), interpolation=cv2.INTER_AREA)
    im = np.transpose(obs, (2, 0, 1))
    return im


class Experiment:
    def __init__(self, exp_cfg, dset):
        self.exp_cfg = exp_cfg
        # Logging setup
        self.logdir = os.path.join(
            self.exp_cfg['logdir'], '{}_SAC_{}_{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.exp_cfg['env_name'], self.exp_cfg['policy'],
                self.exp_cfg['logdir_suffix']))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg,
                    open(os.path.join(self.logdir, "args.pkl"), "wb"))

        # Experiment setup
        self.experiment_setup()

        # Memory
        self.memory = ReplayMemory(self.exp_cfg['replay_size'], self.exp_cfg['seed'])
        self.recovery_memory = ConstraintReplayMemory(
            self.exp_cfg['safe_replay_size'], self.exp_cfg['seed'])
        self.all_ep_data = []

        self.total_numsteps = 0
        self.updates = 0
        self.num_constraint_violations = 0
        self.num_unsafe_transitions = 0

        self.num_viols = 0
        self.num_successes = 0
        self.viol_and_recovery = 0
        self.viol_and_no_recovery = 0
        # Get demos
        self.task_demos = self.exp_cfg['task_demos']
        self.constraint_demo_dataset = dset

    def experiment_setup(self):
        torch.manual_seed(self.exp_cfg['seed'])
        np.random.seed(self.exp_cfg['seed'])

    def pretrain_critic_recovery(self):
        # Get data for recovery policy and safety critic training
        self.num_unsafe_transitions = 0
        for transition in self.constraint_demo_data:
            self.recovery_memory.push(*transition)
            self.num_constraint_violations += int(transition[2])
            self.num_unsafe_transitions += 1
            if self.num_unsafe_transitions == self.exp_cfg['num_unsafe_transitions']:
                break
        print("Number of Constraint Transitions: ",
                self.num_unsafe_transitions)
        print("Number of Constraint Violations: ",
                self.num_constraint_violations)

        # Train DDPG recovery policy
        for i in range(self.exp_cfg['critic_safe_pretraining_steps']):
            if i % 100 == 0:
                print("CRITIC SAFE UPDATE STEP: ", i)
            self.agent.safety_critic.update_parameters(
                memory=self.recovery_memory,
                policy=self.agent.policy,
                batch_size=min(self.exp_cfg['batch_size'],
                                len(self.constraint_demo_data)))

    def run(self):

        self.pretrain_critic_recovery()
        # Optionally initialize task policy with demos


    def dump_logs(self, train_rollouts, test_rollouts):
        data = {"test_stats": test_rollouts, "train_stats": train_rollouts}
        with open(osp.join(self.logdir, "run_stats.pkl"), "wb") as f:
            pickle.dump(data, f)
