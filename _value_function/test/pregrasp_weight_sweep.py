from _value_function.screwdriver_problem import init_env, pregrasp, do_turn, convert_partial_to_full_config, convert_full_to_partial_config
from _value_function.train_value_function import Net, query_ensemble, load_ensemble
import pathlib
import numpy as np
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import time
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
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
        initialization = torch.cat((torch.tensor([[0.1000,  0.6000,  0.6000,  0.6000]]).float().to(device=sim_device) + index_noise,
                                    torch.tensor([[-0.1000,  0.5000,  0.9000,  0.9000]]).float().to(device=sim_device) + middle_thumb_noise,
                                    torch.tensor([[0.0000,  0.5000,  0.6500,  0.6500]]).float().to(device=sim_device),
                                    torch.tensor([[1.2000,  0.3000,  0.3000,  1.2000]]).float().to(device=sim_device) + middle_thumb_noise,
                                    torch.tensor([[0.0000,  0.0000,  0.0000,  0.0000]]).float().to(device=sim_device) + screwdriver_noise),
                                    dim=1).to(sim_device)
        
        env.reset(dof_pos= initialization)
        solved_initialization = env.get_state()['q'].reshape(1,16)[:,0:-1].to(device=sim_device)
        return convert_partial_to_full_config(solved_initialization)

def get_initializations():
     
    initializations = []
    while len(initializations) < n_samples:
        initialization = get_initialization(sim_device, env)
        sd = initialization[0, -4:-1]
        if abs(sd[0]) < 0.05 and abs(sd[1]) < 0.05:
            initializations.append(initialization)
    pkl.dump(initializations, open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'wb'))

def test():

    initializations = pkl.load(open(f'{fpath}/vf_weight_sweep/initializations.pkl', 'rb'))
    models, poses_mean, poses_std, cost_mean, cost_std = load_ensemble(model_name="ensemble")
    
    vf_weights = [10.0, 100.0, 500.0, 1600.0]
    other_weights = [0.06143, 1.0]
    lowest_total_cost = 1e10
    results = []

    for vf_weight in vf_weights:
         for other_weight in other_weights:
            total_pred_cost = 0
            for i in range(n_samples):
                pose_vf = pregrasp(env, config, chain, deterministic=True, 
                            initialization = initializations[i], mode='vf', 
                            vf_weight = vf_weight, other_weight = other_weight,
                            vis_plan=False, iters = 200)
                
                vf_pose = convert_full_to_partial_config(pose_vf.reshape(1,20))
                vf_pose_norm = (vf_pose - poses_mean) / poses_std
                vf_pose_norm = vf_pose_norm.float()
                vf = query_ensemble(vf_pose_norm, models)
                prediction_vf_norm = vf.mean(dim=0)
                prediction_vf = prediction_vf_norm * cost_std + cost_mean
                total_pred_cost += prediction_vf.item()
            print(f'vf_weight: {vf_weight}, other_weight: {other_weight}, total_pred_cost: {total_pred_cost}')
            results.append((vf_weight, other_weight, total_pred_cost))
            if total_pred_cost < lowest_total_cost:
                lowest_total_cost = total_pred_cost
                best_vf_weight = vf_weight
                best_other_weight = other_weight

    pkl.dump(results, open(f'{fpath}/vf_weight_sweep/results.pkl', 'wb'))
    print("results:")
    print(results)
    print(f'best_vf_weight: {best_vf_weight}, best_other_weight: {best_other_weight}, lowest_total_cost: {lowest_total_cost}')


if __name__ == "__main__":
    config, env, sim_env, ros_copy_node, chain, sim, gym, viewer, state2ee_pos_partial = init_env(visualize=False)
    sim_device = config['sim_device']
    n_samples = 3
    get_initializations()
    test()
