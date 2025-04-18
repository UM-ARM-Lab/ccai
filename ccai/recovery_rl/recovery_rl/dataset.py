from typing import Optional
import torch
import pickle
import pathlib
import numpy as np
from torch.utils.data import Dataset


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
                 state_dim: int = 15,
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
            
            for traj_file in trajectory_files:
                with open(traj_file, 'rb') as f:
                    trajectory = pickle.load(f)
                    
                    # Process trajectory to extract transitions
                    if len(trajectory) > 1:  # Need at least 2 states for a transition
                        traj_states, traj_actions, traj_next_states, traj_dones = self._process_trajectory(trajectory)
                        states.extend(traj_states)
                        actions.extend(traj_actions)
                        next_states.extend(traj_next_states)
                        dones.extend(traj_dones)
                # except Exception as e:
                #     print(f"Error loading {traj_file}: {e}")
        
        if not states:
            raise ValueError("No valid transitions found in the provided folders")
        
        # Convert to tensors
        self.states = torch.stack(states)
        self.actions = torch.stack(actions)[:, :num_fingers*4]
        self.next_states = torch.stack(next_states)
        self.dones = torch.tensor(dones).float()
        
        # Infer action dimension if not provided
        if action_dim is None:
            self.action_dim = self.actions.shape[1]
        else:
            self.action_dim = action_dim
            
        # Apply cosine/sine transformation if needed
        self.roll = self.next_states[:, -3].clone()
        self.pitch = self.next_states[:, -2].clone()
        self.dropped = (self.roll.abs() > 0.35) | (self.pitch.abs() > 0.35)
        self.dropped = self.dropped.float().reshape(-1)
        # if self.cosine_sine:
        #     self._apply_cosine_sine_transform()
            
        print(f"Loaded {len(self.states)} transitions")
        print(f"State shape: {self.states.shape}, Action shape: {self.actions.shape}")
    
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


