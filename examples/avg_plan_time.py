import numpy as np

dir_name = './logs/'

log_files = [
    'projected_cnf_only_all_cnstrts_C_10_rk4_.1'
    # 'allegro_screwdriver_csvto_diff_sine_cosine_planned_replanned_contact_eps_.015_2.5_damping_pi_6',
    # 'allegro_screwdriver_csvto_diff_planned_contact_sine_cosine_1.7',
    # 'allegro_screwdriver_csvto_diff_sine_cosine_eps_.015_2.5_damping_pi_6',
    # 'allegro_screwdriver_csvto_diff_sine_cosine_planned_replanned_contact_single_sample_eps_.015_2.5_damping_pi_6',
    # 'allegro_screwdriver_csvto_diff_sine_cosine_planned_replanned_contact_single_particle_16',
    # 'allegro_screwdriver_csvto_only_eps_.015_2.5_damping_pi_6'
]

# log_files = [
#     'allegro_card_csvto_diff_plan',
#     'allegro_card_csvto_diff_plan_no_replan',
#     'allegro_card_csvto_diff',
#     'allegro_card_csvto_diff_plan_no_uncertainty_prop',
#     'allegro_card_csvto_diff_plan_max_likelihood',
#     'allegro_card_csvto_only'
# ]

# log_files = [
#     'screwdriver_diffusion_policy',
# ]

# log_files = [
#     'planned_replanned_hardware',
#     'planned_replanned_hardware_2',
#     'planned_replanned_hardware_3',
#     'planned_replanned_hardware_4',
#     'planned_replanned_hardware_5',
#     'planned_replanned_hardware_6',
# ]

# log_files = [
#     'screwdriver_csvto_diff'
# ]

diff_time = []
traj_opt_time = []
sequence_search_time = []
for log_file in log_files:
    log_file_name = dir_name + log_file + '.log'

    # Read in log file
    with open(log_file_name, 'r') as f:
        log = f.readlines()

    # sequence_search_time = []

    for line in log:
        if 'sequence search time' in line:
            sequence_search_time.append(float(line.split()[-1]))
        if 'Solve time' in line:
            traj_opt_time.append(float(line.split()[-1]))
        if 'Sampling time' in line:
            diff_time.append(float(line.split()[-1]))
    # print(sorted(sequence_search_time))
    print(log_file)
    print(f'Average sequence search time: {np.mean(sequence_search_time)}')
    print()

# a = np.array(sequence_search_time).reshape(10, 7)[:, :4].mean()
# print(a)
# print(f'Average sequence search time: {np.mean(sequence_search_time)}')
print(f'Average diff time: {np.mean(diff_time)}')
print(f'Average traj opt time: {np.mean(traj_opt_time)}')

print(np.mean(traj_opt_time) * 12)