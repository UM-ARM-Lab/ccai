from screwdriver_problem import init_env, pregrasp, do_turn, convert_partial_to_full_config
from train_value_function import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import time
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def get_initialization(sim_device, env):
        # torch.random.manual_seed(1)
        m = 0.3
        index_noise_mag = torch.tensor([m]*4)
        index_noise = index_noise_mag * (2 * torch.rand(4) - 1)
        middle_thumb_noise_mag = torch.tensor([m]*4)
        middle_thumb_noise = middle_thumb_noise_mag * (2 * torch.rand(4) - 1)
        screwdriver_noise = torch.tensor([
        np.random.uniform(-0.05, 0.05),  # Random value between -0.05 and 0.05
        np.random.uniform(-0.05, 0.05),  # Random value between -0.05 and 0.05
        np.random.uniform(0, 2 * np.pi),  # Random value between 0 and 2π
        0.0  
        ])
        #fingers=['index', 'middle', 'ring', 'thumb']
        initialization = torch.cat((torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + index_noise,
                                    torch.tensor([[0., 0.5, 0.7, 0.7]]).float().to(device=sim_device) + middle_thumb_noise,
                                    torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=sim_device),
                                    torch.tensor([[1.3, 0.3, 0.2, 1.1]]).float().to(device=sim_device) + middle_thumb_noise,
                                    torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float().to(device=sim_device) + screwdriver_noise),
                                    dim=1).to(sim_device)
        
        env.reset(dof_pos= initialization)
        solved_initialization = env.get_state()['q'].reshape(1,16)[:,0:-1].to(device=sim_device)
        return convert_partial_to_full_config(solved_initialization)

def test(test_name=''):

    initializations = []
    while len(initializations) < n_samples:
        initialization = get_initialization(sim_device, env)
        sd = initialization[0, -4:-1]
        if abs(sd[0]) < 0.05 and abs(sd[1]) < 0.05:
            initializations.append(initialization)

    # for initialization in initializations:
    #     env.reset(initialization)
    #     time.sleep(1)

    poses_vf = []
    poses_novf = []
    vis_plan = True
    for i in range(n_samples):
        pose_vf = pregrasp(env, config, chain, deterministic=True, 
                           initialization = initializations[i], useVFgrads=True, vis_plan=vis_plan, iters = 500)
        pose_novf = pregrasp(env, config, chain, deterministic=True, 
                             initialization = initializations[i], useVFgrads=False, vis_plan = vis_plan, iters = 50)
        poses_vf.append(pose_vf)
        poses_novf.append(pose_novf)

    # construct a list of tuples of the form (initialization, pose_vf, pose_novf)
    tuples = [(initializations[i], poses_vf[i], poses_novf[i]) for i in range(n_samples)]

    pkl.dump(tuples, open(f'{fpath}/test/test{test_name}.pkl', 'wb'))

if __name__ == "__main__":
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=True)
    sim_device = config['sim_device']
    n_samples = 20
    test_name = 'vf_only_20'

    # test(test_name)
    tuples = pkl.load(open(f'{fpath}/test/test{test_name}.pkl', 'rb'))
    initializations, poses_vf, poses_novf = zip(*tuples)

    while True:
        for i in range(n_samples):
            print("Initialization")
            env.reset(initializations[i])
            time.sleep(1)
            print("No VF gradients")
            env.reset(poses_novf[i])
            time.sleep(1)
            print("With VF gradients")
            env.reset(poses_vf[i])
            time.sleep(2)
        break