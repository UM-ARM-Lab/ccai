import torch
import matplotlib.pyplot as plt 
from isaac_victor_envs.utils import get_assets_dir
import pytorch_kinematics as pk
    
if __name__ == "__main__":
    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
        'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
        'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
        'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
        'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
    }
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'
    action_open = torch.zeros(16)
    action_close = torch.tensor([
        0.0, 0.8, 0.8, 0.8,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ])
    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in ee_names.keys()]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    fk_dict = chain.forward_kinematics(action_close, frame_indices=frame_indices)


    data_1 = torch.load('robot_state_1.pt')
    data_500 = torch.load('robot_state_500.pt')
    data_1000 = torch.load('robot_state_1000.pt')
    data_2000 = torch.load('robot_state_2000.pt')
    label = ['1', '500', '1000', '2000']
    data_list = [data_1, data_500, data_1000, data_2000]
    target_pos = data_1['target_pos']
    for i, data in enumerate(data_list):
        robot_state = data['robot_state']
        ee_pos = chain.forward_kinematics(robot_state, frame_indices=frame_indices)
        ee_pos = ee_pos['allegro_hand_hitosashi_finger_finger_0_aftc_base_link'].get_matrix()[:, :3, 3]
        ee_error = torch.linalg.norm(ee_pos - target_pos, dim=-1)
        plt.plot(ee_error[1000:], label=label[i])
    plt.xlabel('Step')
    plt.ylabel('Index tip error (M)')
    plt.legend()
    plt.axhline(y=0.0, color='black', linestyle='--')
    plt.show()

