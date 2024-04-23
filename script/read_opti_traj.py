import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    data_dir = "/home/fanyang/github/ccai/data/experiments/allegro_screwdriver/csvgd/trial_1_2k_iter/op_traj.pkl"
    with open(data_dir, "rb") as f:
        data = pkl.load(f)
    
    # plt.plot(data['J'], label='cost')
    plt.plot(data['contact_con'], label='contact')
    plt.plot(data['force_con'], label='force')
    plt.plot(data['kinematics_con'], label='kinematics')
    plt.plot(data['friction_con'], label='friction')

    # plt.plot(data['contact_con_mean'], label='contact_mean', linestyle='--')
    # plt.plot(data['force_con_mean'], label='force_mean', linestyle='--')
    # plt.plot(data['kinematics_con_mean'], label='kinematics_mean', linestyle='--')
    # plt.plot(data['friction_con_mean'], label='friction_mean', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.show()
