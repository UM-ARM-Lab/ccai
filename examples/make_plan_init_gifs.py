import os
import sys
import yaml
import pathlib

import numpy as np

from tqdm import tqdm
import PIL

from PIL import Image, ImageDraw, ImageFont

from copy import deepcopy

import imageio
import pickle

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{sys.argv[1]}.yaml').read_text())
# config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/valve/allegro_valve_csvto_recovery_model.yaml').read_text())
# config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_mppi_safe_rl_recovery.yaml').read_text())
config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_csvto_recovery_model_alt_2_noised_s0_9000_bto_recovery_diff_traj_pi_2.yaml').read_text())
# config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_csvto_recovery_model_alt_2_noised_s0_9000_bto_recovery_diff_traj_pi_2.yaml').read_text())



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

dirpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/csvgd')

# for trial_num in range(10):

#CPC
# for trial_num in [1, 2, 3, 4, 8]:
# for trial_num in [8]:

#Ind
# for trial_num in [4, 9]:
# allegro_screwdriver_recovery_data_no_cpc
#Full
# for trial_num in [8, 9]:

# Get all possible trial_inds by reading the names of all directories in dirpath. Structure is trial_{trial_ind}
trial_inds = [int(d.split('_')[-1]) for d in os.listdir(dirpath) if d[:5] == 'trial']

# for trial_num in trial_inds:
# for trial_num in [2, 3]:
# for trial_num in [6, 14]:
for trial_num in [10]:
    fpath = dirpath / f'trial_{trial_num}'

    with open(fpath / 'traj_data.p', 'rb') as f:
        data = pickle.load(f)
        fl = data['pre_action_likelihoods']
    offset_ind = 5 if 'valve' in config['experiment_name'] else 8
    isaac_imgs = [fpath / img for img in sorted(os.listdir(fpath)) if img[-3:] == 'png'][offset_ind:]
    isaac_imgs = [imageio.imread(img) for img in isaac_imgs]

    # Read the names of all directories in fpath
    c_mode_dirs = [str(d) for d in fpath.iterdir() if d.is_dir() and 'turn' in str(d).split('/')[-1] or 'thumb_middle' in str(d).split('/')[-1] or 'index' in str(d).split('/')[-1] or 'thumb' in str(d).split('/')[-1] or 'middle' in str(d).split('/')[-1] or 'mppi' in str(d).split('/')[-1]]
    # Each element of c_mode_dirs has structure mode_ind. We want to sort by ind
    c_mode_dirs = sorted([(int(dir_n.split('_')[-1]), dir_n) for dir_n in c_mode_dirs if dir_n[-2] == '_' or dir_n[-3] == '_'])

    # Construct pause_inds. This is a dictionary where the key is the index of the image to pause at and the value is the text to display. Use c_mode_dirs to construct this. turn mode is 24 images long and the others are 12
    pause_inds_name_dict = {
        'turn': 'all',
        'thumb_middle_regrasp': 'index',
        'index_regrasp': 'thumb_middle'
    }
    
    pause_inds = []
    frame_ind = 0
    last_text = ''
    for i, (ind, dir_n) in enumerate(c_mode_dirs):
        # pause_inds.append((ind*24-offset_ind, pause_inds_name_dict[dir_n.split('/')[-1][:-2]]))
        if 'turn' in dir_n.split('/')[-1]:
            num_exec_steps = len(fl[ind-1]) - 1
            
            if num_exec_steps == config['T_orig'] - 1:
                num_exec_steps += 1
            if num_exec_steps <= 0:
                continue
            if last_text != 'Turn':
                pause_inds.append((frame_ind, 'Turn'))
            num_frames = num_exec_steps * 3
            frame_ind += num_frames
            last_text = 'Turn'
        elif 'thumb_middle' in dir_n.split('/')[-1]:
            pause_inds.append((frame_ind, 'Thumb/middle reset'))
            frame_ind += 9
            last_text = 'Thumb/middle reset'
        elif 'index' in dir_n.split('/')[-1]:
            pause_inds.append((frame_ind, 'Index reset'))
            frame_ind += 9
            last_text = 'Index reset'
        elif 'thumb' in dir_n.split('/')[-1]:
            pause_inds.append((frame_ind, 'Thumb reset'))
            frame_ind += 9
            last_text = 'Thumb reset'
        elif 'middle' in dir_n.split('/')[-1]:
            pause_inds.append((frame_ind, 'Middle reset'))
            frame_ind += 9
            last_text = 'Middle reset'
        elif 'mppi' in dir_n.split('/')[-1]:
            if last_text != 'Recovery':
                pause_inds.append((frame_ind, 'Recovery'))
            frame_ind += 3
            last_text = 'Recovery'
        else:
            raise ValueError(f'Unknown directory name: {dir_n}')
    # Add final pause
    pause_inds.append((len(isaac_imgs)-1, ''))

    # if trial_num == 1:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (21-6, 'thumb_middle'), (33-6, 'thumb_middle'), (45-6, 'turn')]
    # elif trial_num == 2:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (9-6, 'index'), (21-6, 'index'), (33-6, 'turn')]
    # elif trial_num == 3:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (36-6, 'index')]
    # elif trial_num == 4:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (12-6, 'thumb_middle'), (24-6, 'index'), (36-6, 'turn')]
    # elif trial_num == 8:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (15-6, 'index'), (27-6, 'index'), (42-6, 'index'), (54-6, 'index')]


    # if trial_num == 4:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (27-6, 'index'), (45-6, 'thumb_middle'), (57-6, 'turn')]
    # elif trial_num == 9:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (30-6, 'index'), (45-6, 'thumb_middle'), (57-6, 'turn')]

    # if trial_num == 8:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (9-6, 'thumb_middle'), (21-6, 'turn')]
    # elif trial_num == 9:
    #     pause_inds = [(0, 'turn'), (len(isaac_imgs)-1, ''), (30-6, 'index'), (42-6, 'index'), (54-6, 'turn')]
    
    pause_inds = dict(pause_inds)
    # pause_inds = {}

    if 'screwdriver' in config['experiment_name']:
        x0 = 675
        y0 = 750
        width = 300
        text_y = y0 - 450  # Adjust text position relative to progress bar
        progress_bar_y = y0 + 20
        font_size=30
    elif 'valve' in config['experiment_name']:
        x0 = 800
        y0 = 750
        width = 500
        text_y = y0 - 675  # Adjust text position relative to progress bar
        progress_bar_y = y0 + 20
        font_size=50
    progress_bar_x = x0 + width
    # Add a progress bar to the gif by editing each image. Overwrite the original images
    imgs_with_progress_bar = []
    start_end_buffer = 10
    new_isaac_imgs = []

    for j in range(len(isaac_imgs)):
        img = isaac_imgs[j]
        num_adds = start_end_buffer if (j in pause_inds) else 1
        for _ in range(num_adds):
            new_isaac_imgs.append((j, img))

    text = 'Turn'
    for i in range(len(new_isaac_imgs)):
        idx, img = new_isaac_imgs[i]
        if idx in pause_inds and pause_inds[idx] != '':
            text = pause_inds[idx]
        # Add elements to full image
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        
        # Draw progress bar
        draw.rectangle([x0, y0, x0 + width * (idx)/(len(isaac_imgs)-1), y0 + 20], 
                      fill='green')
        
        # Draw text
        font = ImageFont.load_default(size=font_size)
        if text == 'Turn':
            rgb = (0, 255, 0)
        else:
            rgb = (255, 0, 0)
        draw.text((x0 + width//2, text_y), text, rgb, 
                 font=font, anchor="mm")
                 
        # Convert to numpy
        img = np.asarray(img).copy()
        
        # Crop after adding elements
        crop_top = text_y - 40  # Above text
        crop_bottom = y0 + 20  # Below progress bar
        img = img[crop_top:crop_bottom, x0:x0+width]
        
        img[0, 0] += np.array([0, 0, 0, 2 * (i % 2) - 1]).astype(np.uint8)
        imgs_with_progress_bar.append(img)

    if len(imgs_with_progress_bar) > 0:
        imageio.mimsave(f'{fpath}/{config["experiment_name"]}_{trial_num}.gif', imgs_with_progress_bar, loop=0)

    writer = imageio.get_writer(f'{fpath}/{config["experiment_name"]}_{trial_num}_hq.mp4', fps=10,
                                    codec='libx264',  # Ensure to use a good codec
                                    quality=10,  # Use a scale of 1-10, where 10 is maximum quality in some encoders
                                    pixelformat='yuv420p',  # Pixel format compatible with most players
                                    ffmpeg_params=[
                                        '-crf', '17',  # Constant Rate Factor for quality (lower is better quality, 17-23 is common)
                                        '-preset', 'slow'  # Adjust encoder speed/efficiency, 'slow' offers better compression and quality
                                    ])
    for img in imgs_with_progress_bar:
        writer.append_data(np.array(img, np.uint8))
    writer.close()
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
