from isaacgym import gymapi
from isaac_victor_envs.utils import get_assets_dir
from typing import Optional
import torch
import pickle
import pathlib
import numpy as np
from torch.utils.data import Dataset
import pytorch_volumetric as pv
from pytorch_kinematics import transforms as tf
import pytorch_kinematics as pk
from ccai.utils.allegro_utils import partial_to_full_state

from tqdm import tqdm


class AllegroTrajectoryTransitionDataset(Dataset):
    """
    Dataset that loads trajectory.pkl files and provides state-action-next_state transitions.
    
    Args:
        folders: List of folders containing trajectory.pkl files
        cosine_sine: Whether to convert yaw to sine/cosine representation
        states_only: Whether to only process state variables
        action_dim: Dimension of action space (default: None, inferred from data)
        state_dim: Dimension of state space (default: 15)
        transform_fn: Optional function to transform states before computing actions
    """
    
    def __init__(self, 
                 folders: list, 
                 cosine_sine: bool = False, 
                 states_only: bool = False,
                 action_dim: Optional[int] = None,
                 state_dim: int = 13,
                 transform_fn = None,
                 num_fingers=3):
        super().__init__()
        self.cosine_sine = cosine_sine
        self.states_only = states_only
        self.state_dim = state_dim
        self.transform_fn = transform_fn
        
        # Storage for transitions
        states = []
        actions = []
        next_states = []
        dones = []
        
        # Load all trajectory.pkl files
        for folder in folders:
            path = pathlib.Path(folder)
            trajectory_files = list(path.rglob('*trajectory.pkl'))
            full_data_files = list(path.rglob('*traj_data.p'))
            
            for traj_file, full_data_file in zip(trajectory_files, full_data_files):
                with open(traj_file, 'rb') as f, open(full_data_file, 'rb') as f2:
                    trajectory = pickle.load(f)
                    
                    # Process trajectory to extract transitions
                    if len(trajectory) > 1:  # Need at least 2 states for a transition
                        traj_states, traj_actions, traj_next_states, traj_dones = self._process_trajectory(trajectory)
                        states.extend(traj_states)
                        actions.extend(traj_actions)
                        next_states.extend(traj_next_states)
                        dones.extend(traj_dones)
                        
                    # data = pickle.load(f2)
                    # for t in range(3, 0, -1)
                # except Exception as e:
                #     print(f"Error loading {traj_file}: {e}")
        
        if not states:
            raise ValueError("No valid transitions found in the provided folders")
        
        # Convert to tensors
        self.states = torch.stack(states)
        self.actions = torch.stack(actions)[:, :num_fingers*4]
        self.next_states = torch.stack(next_states)
        self.dones = torch.tensor(dones).float()
        
        print(self.states.shape, self.actions.shape, self.next_states.shape)
        
        # Infer action dimension if not provided
        if action_dim is None:
            self.action_dim = self.actions.shape[1]
        else:
            self.action_dim = action_dim
            
        self.device='cuda:0'
        device = 'cuda:0'
        self.robot_p = np.array([0.02, -0.35, .376]).astype(np.float32)
        self.robot_r = [0, 0, 0.7071068, 0.7071068]
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.robot_p)
        # NOTE: for isaac gym quat, angle goes last, but for pytorch kinematics, angle goes first 
        pose.r = gymapi.Quat(*self.robot_r)
        world_trans = tf.Transform3d(pos=torch.tensor(self.robot_p, device=self.device),
                                          rot=torch.tensor(
                                              [self.robot_r[3], self.robot_r[0], self.robot_r[1], self.robot_r[2]],
                                              device=self.device), device=self.device)
            
        self.ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }

        self.collision_link_names = {
            'index':
            ['allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
             'allegro_hand_hitosashi_finger_finger_link_3',
             'allegro_hand_hitosashi_finger_finger_link_2',
             'allegro_hand_hitosashi_finger_finger_link_1',
             'allegro_hand_hitosashi_finger_finger_link_0'],
             'middle':
            ['allegro_hand_naka_finger_finger_1_aftc_base_link',
             'allegro_hand_naka_finger_finger_link_7',
             'allegro_hand_naka_finger_finger_link_6',
             'allegro_hand_naka_finger_finger_link_5',
             'allegro_hand_naka_finger_finger_link_4'],
                'ring':
            ['allegro_hand_kusuri_finger_finger_2_aftc_base_link',
                'allegro_hand_kusuri_finger_finger_link_11',
                'allegro_hand_kusuri_finger_finger_link_10',
                'allegro_hand_kusuri_finger_finger_link_9',
                'allegro_hand_kusuri_finger_finger_link_8'],
                'thumb':
            ['allegro_hand_oya_finger_3_aftc_base_link',
            'allegro_hand_oya_finger_link_15',
            'allegro_hand_oya_finger_link_14',
            'allegro_hand_oya_finger_link_13',
            'allegro_hand_oya_finger_link_12'],
        }
        # self.collision_link_names = {
        #     'index':
        #     ['allegro_hand_hitosashi_finger_finger_0_aftc_base_link'],
        #      'middle':
        #     ['allegro_hand_naka_finger_finger_1_aftc_base_link'],
        #         'ring':
        #     ['allegro_hand_kusuri_finger_finger_2_aftc_base_link'],
        #         'thumb':
        #     ['allegro_hand_oya_finger_3_aftc_base_link'],
        # }
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
        ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }
        valve_asset = f'{get_assets_dir()}/valve/valve_cross.urdf'

        chain = pk.build_chain_from_urdf(open(asset).read()).to(device=device)
        self.fingers = ['index', 'middle', 'thumb']
        self.ee_link_idx = {finger: chain.frame_to_idx[ee_name] for finger, ee_name in self.ee_names.items()}
        self.frame_indices = torch.tensor([self.ee_link_idx[finger] for finger in self.fingers])

        object_type = 'valve'
        ##### SDF for robot and environment ######
        if object_type == 'cuboid_valve':
            asset_object = get_assets_dir() + '/valve/valve_cuboid.urdf'
        elif object_type == 'cylinder_valve':
            asset_object = get_assets_dir() + '/valve/valve_cylinder.urdf'
        elif object_type == 'valve':
            asset_object = get_assets_dir() + '/valve/valve_cross.urdf'
        elif object_type == 'screwdriver':
            asset_object = get_assets_dir() + '/screwdriver/screwdriver.urdf'
        elif object_type == "card":
            asset_object = get_assets_dir() + '/card/card.urdf'
        elif object_type == "peg":
            asset_object = get_assets_dir() + '/peg/short_peg.urdf'
        self.object_asset_pos = np.array([0.0, 0.0, 0.40])
        object_asset_pos = np.array([0.0, 0.0, 0.40])

        chain_object = pk.build_chain_from_urdf(open(asset_object).read())
        chain_object = chain_object.to(device=device)
        if 'valve' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/valve',
                                     use_collision_geometry=True)
            object_sdf_for_viz = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/valve',
                                     use_collision_geometry=True) # Use collision geometry for visualization to check fidelity of contact mesh
        elif 'screwdriver' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/screwdriver',
                                     use_collision_geometry=True)
            object_sdf_for_viz = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/screwdriver',
                                     use_collision_geometry=False)
        elif 'card' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/card',
                                     use_collision_geometry=False)
        elif 'peg' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/peg',
                                     use_collision_geometry=True)
            object_sdf_for_viz = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/peg',
                                     use_collision_geometry=False)
        self.object_sdf = object_sdf
        robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models',
                                use_collision_geometry=False)
        
        obj_to_world_trans = pk.Transform3d(device='cuda:0').translate(object_asset_pos[0], object_asset_pos[1], object_asset_pos[2])
        screwdriver_origin_to_world_trans = pk.Transform3d(device='cuda:0').translate(object_asset_pos[0], object_asset_pos[1], object_asset_pos[2] + .1 + 0.412*2.54/100)
        scene_trans = world_trans.inverse().compose(obj_to_world_trans).to(device=device)
        object_to_hand_trans = world_trans.inverse().compose(screwdriver_origin_to_world_trans)
        
        self.obj_dof = 1
        if self.obj_dof == 3:
            object_link_name = 'screwdriver_body'
        elif self.obj_dof == 1:
            object_link_name = 'cross_1'
        elif self.obj_dof == 6:
            object_link_name = 'peg'
        self.obj_link_name = object_link_name
        # Object to robot transform
        # robot_minus_object = world_trans.to(device=device).compose(obj_to_world_trans.to(device=device).inverse())
        # print(robot_minus_object.get_matrix())
        # contact checking
        # collision_check_links = [self.ee_names[finger] for finger in self.fingers]
        collision_check_links = sum([self.collision_link_names[finger] for finger in self.fingers], [])
            
        self.contact_scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=750,
                                            links_per_finger=len(self.collision_link_names['index']),
                                            obj_link_name=self.obj_link_name
                                            )
        self.dropped = self.calc_constraints()
            
        # Apply cosine/sine transformation if needed
        # self.roll = self.next_states[:, -3].clone()
        # self.pitch = self.next_states[:, -2].clone()
        # self.dropped = (self.roll.abs() > 0.35) | (self.pitch.abs() > 0.35)
        # self.dropped = self.dropped.float().reshape(-1)
        # if self.cosine_sine:
        #     self._apply_cosine_sine_transform()
            
        print(f"Loaded {len(self.states)} transitions")
        print(f"State shape: {self.states.shape}, Action shape: {self.actions.shape}")
    
    
    def calc_constraints(self):
        """
            Uses self.contact_scene.scene_get_sdf to get sdf values for the fingers
            Uses a batch size of 128
        """
        batch_size = 1024
        num_batches = int(np.ceil(len(self.states) / batch_size))
        constraints = []
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.states))
            next_states_batch = self.next_states[start_idx:end_idx]
            next_states_q = next_states_batch[:, :12].to(self.device)
            next_states_q = partial_to_full_state(next_states_q, fingers=['index', 'middle','thumb'])
            next_states_theta = next_states_batch[:, 12:13].to(self.device)
            rvals = self.contact_scene.scene_get_sdf(next_states_q, next_states_theta)
            sdf = rvals['sdf']
            sdf = sdf.reshape(-1, 3).max(1)[0]
            # print(sdf)
            constraint = (sdf > .003).float().cpu()
            # print(constraint)
            constraints.append(constraint)
        
        ret =  torch.cat(constraints, dim=0)
        print(ret.mean())
        return ret
        
            
    
    def return_as_numpy(self):
        """
        Return as list of tuples (s_t, a_t, dropped, s_t+1)

        s_t, a_t, s_t+1 are np arrays 
        """
        dataset = [(self.states[i].cpu().numpy(), self.actions[i].cpu().numpy(), self.dropped[i].cpu().item(), self.next_states[i].cpu().numpy(), self.dones[i].cpu().item()) for i in range(len(self))]
        print(f"Loaded {len(dataset)} transitions")
        print(f"State shape: {self.states.shape}, Action shape: {self.actions.shape}")
        return dataset

    def _process_trajectory(self, trajectory):
        """
        Process a trajectory to extract (state, action, next_state) tuples.
        
        Args:
            trajectory: List of states from trajectory.pkl
            
        Returns:
            tuple: (states, actions, next_states)
        """
        states = []
        actions = []
        next_states = []
        dones = []
        traj = np.concatenate((trajectory[:-1]), axis=0)
        end = trajectory[-1].reshape(1, -1)
        end = np.concatenate((end, np.zeros((1, 21))), axis=1)
        trajectory = np.concatenate((traj, end), axis=0)

        # Convert trajectory to tensor if it's not already
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
        
        # Apply any state transformations if needed
        if self.transform_fn is not None:
            trajectory = self.transform_fn(trajectory)
        
        # Extract transitions
        for i in range(len(trajectory) - 1):
            state = trajectory[i][:self.state_dim].clone()
            next_state = trajectory[i+1][:self.state_dim].clone()
            
            # Extract action - this depends on the structure of your data
            # Option 1: Action is the difference between consecutive states
            # action = next_state - state
            
            # Option 2: Action is stored in the trajectory data
            # This assumes the trajectory data includes actions after the state variables
            # if trajectory.shape[1] > self.state_dim:
            action = trajectory[i][self.state_dim:].clone()
            # else:
            #     # Fallback to computing action as the difference between states
            #     action = next_state - state
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            dones.append(i == len(trajectory) - 2)  # Mark the last transition as done
        
        return states, actions, next_states, dones
    
    def _apply_cosine_sine_transform(self):
        """Apply cosine/sine transformation to yaw angle (typically at position 14)"""
        yaw_idx = 14  # Typical position for yaw angle
        
        # Transform states
        yaw = self.states[:, yaw_idx:yaw_idx+1]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # Remove yaw and add cos/sin representation
        states_without_yaw = torch.cat([self.states[:, :yaw_idx], self.states[:, yaw_idx+1:]], dim=1)
        self.states = torch.cat([states_without_yaw, cos_yaw, sin_yaw], dim=1)
        
        # Transform next_states
        yaw = self.next_states[:, yaw_idx:yaw_idx+1]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # Remove yaw and add cos/sin representation
        next_states_without_yaw = torch.cat([self.next_states[:, :yaw_idx], self.next_states[:, yaw_idx+1:]], dim=1)
        self.next_states = torch.cat([next_states_without_yaw, cos_yaw, sin_yaw], dim=1)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        """
        Get a transition tuple.
        
        Args:
            idx: Index of transition to retrieve
            
        Returns:
            tuple: (state, action, next_state)
        """
        return self.states[idx], self.actions[idx], self.next_states[idx]
    
    def get_state_dim(self):
        """Get the dimension of the state space"""
        return self.states.shape[1]
    
    def get_action_dim(self):
        """Get the dimension of the action space"""
        return self.actions.shape[1]
    
    def get_batch(self, batch_size):
        """
        Get a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to include in batch
            
        Returns:
            tuple: (states, actions, next_states)
        """
        indices = torch.randint(0, len(self), (batch_size,))
        return self.states[indices], self.actions[indices], self.next_states[indices]


