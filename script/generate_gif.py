import glob
import imageio
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import torch

directory = '/home/fanyang/github/ccai/data/experiments/allegro_valve/csvgd/trial_1'

if __name__ == '__main__':
    with open(directory + '/info.pkl', 'rb') as f:
        result = pkl.load(f)

    png_paths = glob.glob(directory + '/frame*.png')
    png_paths.sort()
    img_list = []
    for i, png_path in enumerate(png_paths):
        img = cv2.imread(png_path)
        # frame_info = result[i]
        # contact_distance = frame_info['distance']
        # dist2goal = frame_info['distance2goal']
        # con_eq_dict = frame_info['equality_eval']
        # con_ineq_dict = frame_info['inequality_eval']
        # equality_eval = torch.stack(list(con_eq_dict.values()), dim=2)
        # inequality_eval = torch.stack(list(con_ineq_dict.values()), dim=2)
        # if inequality_eval[0][0].max() > 0.1:
            # print(con_ineq_dict)

        # img = cv2.putText(img, 
        #             'dist2goal: {:.4f}'.format(dist2goal), 
        #             (20, 40), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # img = cv2.putText(img,
        #             'next dist index: {:.4f}, thumb: {:4f}'.format(contact_distance[0,0,0], contact_distance[0,0,1]),
        #             (20, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # img = cv2.putText(img,
        #             'max dist: {:.4f}, min dist: {:.4f}'.format(contact_distance.max(), contact_distance.min()),
        #             (20, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # img = cv2.putText(img,
        #             'next max equality(abs): {:.4f}, max equality(abs): {:.4f}'.format(equality_eval[0][0].abs().max(), equality_eval[0].abs().max()),
        #             (20, 160),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # img = cv2.putText(img,
        #             'next max inequality: {:.4f}, max inequality: {:.4f}'.format(inequality_eval[0][0].max(), inequality_eval[0].max()),
        #             (20, 200),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        img_list.append(img)
    # imageio.mimsave(directory + '/result.gif', img_list, duration=20)
    imageio.mimsave(directory + '/result.gif', img_list, format='GIF', duration=1000)
