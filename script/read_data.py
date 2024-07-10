import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    with open('/home/fanyang/github/ccai/data/experiments/allegro_peg_insertion_new_pose/csvgd/trial_1/info.pkl', 'rb') as f:
        info_list = pkl.load(f)
    contact_vio_list = []
    kinematics_vio_list = []
    force_vio_list = []
    friction_vio_list = []
    for info in info_list:
        contact_vio_list.append(info['contact'])
        kinematics_vio_list.append(info['kinematics'])
        force_vio_list.append(info['force'])
        friction_vio_list.append(info['friction'])

    # plt.yscale('log')

    plt.plot(contact_vio_list, label='contact')
    plt.plot(kinematics_vio_list, label='kinematics')
    plt.plot(force_vio_list, label='force')
    plt.plot(friction_vio_list, label='friction')
    plt.xlabel('step')
    plt.ylabel(' violation')
    plt.legend()
    plt.show()