import glob
import imageio
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import torch

# directory = '/home/fanyang/github/ccai/data/experiments/allegro_peg_insertion_env_force_3/csvgd/trial_2'
directory = '/home/fanyang/github/ccai/baselines/data/experiments/allegro_reorientation_horizon_500_samples/mppi/trial_3'

if __name__ == '__main__':

    png_paths = glob.glob(directory + '/frame*.png')
    png_paths.sort()
    img_list = []
    for i, png_path in enumerate(png_paths):
        # if (i)% 3 == 0:
        #     continue
        img = cv2.imread(png_path)
        # contact_distance = frame_info['distance']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
    # imageio.mimsave(directory + '/result.gif', img_list, duration=20)
    imageio.mimsave(directory + '/result.gif', img_list, format='GIF', duration=200, loop=0)
