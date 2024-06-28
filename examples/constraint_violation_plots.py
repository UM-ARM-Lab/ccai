import yaml
import pathlib
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_sim_eval.yaml').read_text())
goal = - 0.5 * torch.tensor([0, 0, np.pi])

data_path = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/csvgd')
random_or_diffusion = config['experiment_name'].split('_')[2]
with open(data_path / f'constraint_violations.p', 'rb') as f:
    constraint_violations_all = pkl.load(f)

def gen_plot(constraint_violations_all, key):
    plan_cv = constraint_violations_all[key]

    keys = ['pregrasp', 'index', 'thumb_middle', 'turn']
    # violation_by_mode_g = {
    #     key: [] for key in keys
    # }

    # violation_by_mode_h = {
    #     key: [] for key in keys
    # }
    g_all = []
    h_all = []
    for trial_ind in range(len(plan_cv)):
        print([i['c_state'] for i in plan_cv[trial_ind]])
        g_trial_ind = []
        h_trial_ind = []
        for contact_mode_seq in plan_cv[trial_ind]:
            g = torch.abs(contact_mode_seq['g']).mean().cpu().item()
            h = torch.relu(contact_mode_seq['h']).mean().cpu().item()
            g_trial_ind.append(g)
            h_trial_ind.append(h)
        g_all.append(g_trial_ind)
        h_all.append(h_trial_ind)
    g_all = np.stack(g_all, axis=0)
    h_all = np.stack(h_all, axis=0)
    g_all_mean = g_all.mean(axis=0)
    h_all_mean = h_all.mean(axis=0)
    g_all_std = g_all.std(axis=0)
    h_all_std = h_all.std(axis=0)
            # mode = contact_mode_seq['c_state']

            # violation_by_mode_g[mode].append(contact_mode_seq['g'])
            # violation_by_mode_h[mode].append(contact_mode_seq['h'])

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # plot g_all_mean with error bars
    ax.plot(g_all_mean, label='g_all_mean')
    ax.fill_between(np.arange(g_all_mean.shape[0]), g_all_mean - g_all_std, g_all_mean + g_all_std, alpha=0.5)
    ax.plot(h_all_mean, label='h_all_mean')
    ax.fill_between(np.arange(h_all_mean.shape[0]), h_all_mean - h_all_std, h_all_mean + h_all_std, alpha=0.5)
    ax.legend()

    plt.title(f'Constraint Violation: {key}')
    plt.savefig(f'/home/abhinav/Pictures/constraint_violation_analysis/{random_or_diffusion}_init_{key}.png')

    # 4 columns, 2 rows subplots
    # fig, axs = plt.subplots(2, 4, figsize=(12, 12))

    # for ind in range(4):
    #     key = keys[ind]
    #     print(violation_by_mode_g[key])
    #     this_mode_g = torch.abs(violation_by_mode_g[key][t])
    #     print(this_mode_g.shape)

    #     this_mode_h = torch.relu(violation_by_mode_h[key][t])
    #     print(this_mode_h.shape)

gen_plot('inits')
gen_plot('plans')