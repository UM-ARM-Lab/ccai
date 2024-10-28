import os
import sys
import yaml
import pathlib

import numpy as np

from tqdm import tqdm
import PIL

from PIL import Image, ImageDraw, ImageFont

import imageio

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{sys.argv[1]}.yaml').read_text())

def add_text_to_imgs(imgs, labels):
    imgs_with_txt = []
    for i, img in enumerate(imgs):
        img = Image.fromarray(img[0:, 175:-175])
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((20, 20), labels[i].replace('_', ' '), (0, 0, 0), font_size=30, align='center')
        # Show image
        imgs_with_txt.append(np.asarray(img))
    return imgs_with_txt

for trial_num in range(0, config['num_trials']):
# for trial_num in [4, 9]:
    fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}./csvgd/trial_{trial_num + 1}')


    isaac_imgs = [fpath / img for img in sorted(os.listdir(fpath)) if img[-3:] == 'png']#[6:]
    print(len(isaac_imgs))
    isaac_imgs = [imageio.imread(img) for img in isaac_imgs]
    if len(isaac_imgs) > 0:
        imageio.mimsave(f'{fpath}/isaac.gif', isaac_imgs, loop=0)
    # Get names of directories in fpath
    for c_plan in range(4):
        fpath_cind = fpath / f'c{c_plan}'
        # Make directories for gifs
        if not os.path.exists(fpath_cind):
            os.makedirs(fpath_cind)
        dirs = [str(d) for d in fpath.iterdir() if d.is_dir() and f'_{c_plan}_' in str(d)]

        c_imgs = sorted([(int(dir_n.split('_')[-1]), dir_n) for dir_n in dirs if dir_n[-2] == '_' or dir_n[-3] == '_'])

        for sample in tqdm(range(16)):
        # for sample in tqdm(range(1)):
            gif_imgs_inits, gif_imgs_plans, gif_imgs_planned_inits = [], [], []
            labels = []
            labels_planned_inits = []
            for iter, (_, dir_n) in enumerate(c_imgs):
                init_dir = fpath / dir_n / 'init' / str(sample) / 'timestep_0' / 'img'
                plan_dir = fpath / dir_n / 'plan' / str(sample) / 'timestep_0' / 'img'
                planned_init_dir = fpath / dir_n / 'planned_init' / str(sample) / 'timestep_0' / 'img'

                # Read in images in init_dir and plan_dir and append to gif_imgs_inits and gif_imgs_plans
                try:
                    gif_imgs_inits.append([init_dir / img for img in sorted(os.listdir(init_dir))])
                    labels += [dir_n.split('/')[-1]] * len(gif_imgs_inits[-1])
                except:
                    pass
                try:
                    gif_imgs_plans.append([plan_dir / img for img in sorted(os.listdir(plan_dir))])
                except:
                    pass
                try:
                    gif_imgs_planned_inits.append([planned_init_dir / img for img in sorted(os.listdir(planned_init_dir))])
                    contact_mode = dir_n.split('/')[-1]
                    contact_mode = contact_mode[:-3] + str(iter)
                    labels_planned_inits += [contact_mode] * len(gif_imgs_planned_inits[-1])
                except:
                    pass

            # Create gifs
            gif_imgs_inits = [img for img_list in gif_imgs_inits for img in img_list]
            gif_imgs_plans = [img for img_list in gif_imgs_plans for img in img_list]
            gif_imgs_planned_inits = [img for img_list in gif_imgs_planned_inits for img in img_list]

            gif_imgs_inits = [imageio.imread(img) for img in gif_imgs_inits]
            gif_imgs_plans = [imageio.imread(img) for img in gif_imgs_plans]
            gif_imgs_planned_inits = [imageio.imread(img) for img in gif_imgs_planned_inits]

            gif_imgs_inits = add_text_to_imgs(gif_imgs_inits, labels)
            gif_imgs_plans = add_text_to_imgs(gif_imgs_plans, labels)
            gif_imgs_planned_inits = add_text_to_imgs(gif_imgs_planned_inits, labels_planned_inits)


            if len(gif_imgs_inits) > 0:
                imageio.mimsave(f'{fpath_cind}/init_{sample}.gif', gif_imgs_inits, loop=0)
            if len(gif_imgs_plans) > 0:
                imageio.mimsave(f'{fpath_cind}/plan_{sample}.gif', gif_imgs_plans, loop=0)
            if len(gif_imgs_planned_inits) > 0:
                imageio.mimsave(f'{fpath_cind}/planned_init_{sample}.gif', gif_imgs_planned_inits, loop=0, fps=12)

