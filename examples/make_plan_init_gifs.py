import os
import sys
import yaml
import pathlib

import imageio

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{sys.argv[1]}.yaml').read_text())

for trial_num in range(config['num_trials']):
    fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/csvgd/trial_{trial_num + 1}')

    # Get names of directories in fpath
    dirs = [str(d) for d in fpath.iterdir() if d.is_dir()]

    c_imgs = sorted([(int(dir_n.split('_')[-1]), dir_n) for dir_n in dirs if dir_n[-2] == '_' or dir_n[-3] == '_'])

    gif_imgs_inits, gif_imgs_plans = [], []

    for iter, dir_n in c_imgs:
        init_dir = fpath / dir_n / 'init' / 'timestep_0' / 'img'
        plan_dir = fpath / dir_n / 'plan' / 'timestep_0' / 'img'

        # Read in images in init_dir and plan_dir and append to gif_imgs_inits and gif_imgs_plans
        gif_imgs_inits.append([init_dir / img for img in sorted(os.listdir(init_dir))])
        gif_imgs_plans.append([plan_dir / img for img in sorted(os.listdir(plan_dir))])

    # Create gifs
    gif_imgs_inits = [img for img_list in gif_imgs_inits for img in img_list]
    gif_imgs_plans = [img for img_list in gif_imgs_plans for img in img_list]

    gif_imgs_inits = [imageio.imread(img) for img in gif_imgs_inits]
    gif_imgs_plans = [imageio.imread(img) for img in gif_imgs_plans]

    if len(gif_imgs_inits) > 0:
        imageio.mimsave(f'{fpath}/init.gif', gif_imgs_inits, loop=0)
    if len(gif_imgs_plans) > 0:
        imageio.mimsave(f'{fpath}/plan.gif', gif_imgs_plans, loop=0)

