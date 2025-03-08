from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
from ccai.utils.allegro_utils import state2ee_pos, partial_to_full_state
import pytorch_kinematics as pk

import os
import sys
import pathlib
import yaml
import torch
import numpy as np
import time
from functools import partial
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

sys.path.append('..')

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class AllegroScrewdriverRLEnv:
    """Custom environment for the Allegro Hand screwdriver turning task"""
    
    def __init__(self, config):
        # Environment configuration
        self.config = config
        self.device = config['sim_device']
        self.num_fingers = len(config['fingers'])
        self.control_dim = 4 * self.num_fingers  # 4 joints per finger
        self.obj_dof = 3
        
        # Create the underlying environment
        self.env = AllegroScrewdriverTurningEnv(
            1, control_mode='joint_impedance',
            use_cartesian_controller=False,
            viewer=config['visualize'],
            steps_per_action=60,
            friction_coefficient=config['friction_coefficient'] * 2.5,
            device=self.device,
            video_save_path=img_save_dir if config['visualize'] else None,
            joint_stiffness=config['kp'],
            fingers=config['fingers'],
            gradual_control=False,
            gravity=True,
            randomize_obj_start=config.get('randomize_obj_start', False),
            randomize_rob_start=config.get('randomize_rob_start', False),
        )
        
        # Get simulation handles
        self.sim, self.gym, self.viewer = self.env.get_sim()
        
        # Define action bounds
        self.action_low = -0.1
        self.action_high = 0.1
        self.action_shape = (self.control_dim,)
        
        # Define observation shape
        self.obs_dim = self.control_dim + self.obj_dof  # Removed *2 as we're no longer including angular velocities
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = config.get('max_episode_steps', 100)
        
        # Previous state for calculating deltas - no longer needed
        # self.prev_orientation = None
        
        # Set up kinematic chain for visualization and state extraction
        asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
        ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }
        self.ee_names = ee_names
        
        chain = pk.build_chain_from_urdf(open(asset).read())
        frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]
        frame_indices = torch.tensor(frame_indices)
        self.state2ee_pos = partial(
            state2ee_pos, 
            fingers=config['fingers'], 
            chain=chain, 
            frame_indices=frame_indices,
            world_trans=self.env.world_trans
        )
        self.chain = chain.to(device=self.device)
    
    def _get_obs(self):
        state_dict = self.env.get_state()
        state = state_dict['q'].reshape(-1)
        
        # Get joint positions
        joint_pos = state[:self.control_dim]
        
        # Get object orientation
        orientation = state[self.control_dim:self.control_dim + self.obj_dof]
        
        # Combine into observation (without angular velocity)
        obs = torch.cat([joint_pos, orientation], dim=0)
        return obs.cpu().numpy()
    
    def reset(self):
        self.env.reset()
        self.episode_steps = 0
        print(f"Episode reset - starting new episode")
        return self._get_obs()
    
    def step(self, action):
        # Convert action to target joint positions
        state_dict = self.env.get_state()
        current_pos = state_dict['q'].reshape(-1)[:self.control_dim]
        target_pos = current_pos + torch.tensor(action, device=self.device)
        
        # Step the environment
        self.env.step(target_pos.reshape(1, -1))
        self.episode_steps += 1
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._compute_reward(obs)
        
        # Extract orientation
        orientation = obs[self.control_dim:self.control_dim + self.obj_dof]
        roll, pitch, yaw = orientation
        
        # Check if episode is done due to max steps or dropping the screwdriver
        done = (self.episode_steps >= self.max_episode_steps) or (abs(roll) > 0.2) or (abs(pitch) > 0.2)
        
        termination_reason = ""
        if done:
            if self.episode_steps >= self.max_episode_steps:
                termination_reason = "reached max steps"
            elif abs(roll) > 0.2 or abs(pitch) > 0.2:
                termination_reason = f"dropped screwdriver (roll={roll:.3f}, pitch={pitch:.3f})"
            print(f"Episode terminated: {termination_reason}, final reward: {reward:.3f}, step: {self.episode_steps}")
        
        return obs, reward, done

    def _compute_reward(self, obs):
        # Extract orientation
        orientation = obs[self.control_dim:self.control_dim + self.obj_dof]
        roll, pitch, yaw = orientation
        
        # Reward turning (yaw change)
        turn_reward = abs(yaw - self.config.get('initial_yaw', 0.0)) * 5.0
        
        # Penalize deviation from upright position
        upright_penalty = (roll**2 + pitch**2) * 10.0
        
        # Calculate total reward
        reward = turn_reward - upright_penalty
        
        return reward

    def close(self):
        if hasattr(self, 'env'):
            if hasattr(self.env, 'close'):
                self.env.close()

# Import necessary ODE libraries
from torchdiffeq import odeint

# Neural network for the actor (policy) using ODE flow
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.combined_dim = input_dim + output_dim
        
        # MLP for modeling derivatives (vector field)
        self.vector_field = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.combined_dim)
        )
        
        # Parameters for controlling integration
        self.integration_time = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
    
    def dynamics(self, t, x):
        """ODE function to integrate"""
        return self.vector_field(x)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Generate random normal vectors for action space
        z = torch.randn(batch_size, self.output_dim, device=x.device)
        
        # Concatenate state and random vector
        x_z = torch.cat([x, z], dim=-1)
        
        # Integrate ODE from t=0 to t=1
        t = torch.tensor([0.0, self.integration_time.item()], device=x.device)
        trajectory = odeint(self.dynamics, x_z, t, method='dopri5')
        
        # Extract the terminal point
        x_final = trajectory[-1]
        
        # Split into state and action parts
        state_part = x_final[:, :self.input_dim]
        action_part = x_final[:, self.input_dim:]
        
        # Scale the actions using the log_std parameter
        std = torch.exp(self.log_std).expand_as(action_part)
        
        return action_part, std

# Neural network for the critic (value function)
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class PPO:
    def __init__(
        self,
        input_dim,
        output_dim,
        action_low,
        action_high,
        device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        num_epochs=10
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Scale actions from [-1, 1] to [low, high]
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        
        # Policy and value networks
        self.actor = ActorNetwork(input_dim, output_dim).to(device)
        self.critic = CriticNetwork(input_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, state, evaluate=False):
        """Select an action from the policy given the state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, std = self.actor(state)
            
            if evaluate:
                # During evaluation, just use the mean
                action = mean
            else:
                # During training, sample from the distribution
                normal = Normal(mean, std)
                action = normal.sample()
                
        return action.cpu().numpy()
    
    def compute_gae(self, values, rewards, dones, next_value):
        """Compute generalized advantage estimates."""
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(self.device)
        
        return returns, advantages
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy and value networks."""
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for num_epochs
        for _ in range(self.num_epochs):
            # Get random minibatch
            indices = torch.randperm(states.size(0))
            for start_idx in range(0, states.size(0), self.batch_size):
                end_idx = min(start_idx + self.batch_size, states.size(0))
                minibatch_indices = indices[start_idx:end_idx]
                
                minibatch_states = states[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_old_log_probs = old_log_probs[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                
                # Evaluate actions
                means, stds = self.actor(minibatch_states)
                dist = Normal(means, stds)
                current_log_probs = dist.log_prob(minibatch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                # Compute value loss
                values = self.critic(minibatch_states).squeeze(-1)
                value_loss = nn.MSELoss()(values, minibatch_returns)
                
                # Compute policy loss
                ratios = torch.exp(current_log_probs - minibatch_old_log_probs)
                surr1 = ratios * minibatch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * minibatch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

def make_env(config):
    """
    Creates an environment
    """
    return AllegroScrewdriverRLEnv(config)

def train_ppo(config, total_timesteps=1000000, save_path=None):
    """
    Train a PPO agent on the AllegroScrewdriver environment
    """
    # Create environment
    env = make_env(config)
    
    # Create log directory
    if save_path is None:
        save_path = f"{CCAI_PATH}/models/allegro_screwdriver_ppo"
    os.makedirs(save_path, exist_ok=True)
    
    # Use GPU for neural networks but keep simulation on CPU
    train_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Simulation running on: {config['sim_device']}")
    print(f"PPO training running on: {train_device}")
    
    # Make sure to install torchdiffeq
    try:
        import torchdiffeq
    except ImportError:
        print("torchdiffeq is not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchdiffeq"])
        
    # Initialize PPO agent
    ppo_agent = PPO(
        input_dim=env.obs_dim,
        output_dim=env.control_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=train_device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        batch_size=64,
        num_epochs=10
    )
    
    # Training variables
    num_steps_per_episode = config.get('max_episode_steps', 100)
    update_frequency = config.get('update_frequency', 1024)
    checkpoint_frequency = config.get('checkpoint_frequency', 2048)
    
    # Initialize tracking variables
    total_steps = 0
    best_average_reward = -float('inf')
    episode_rewards = []
    log_interval = config.get('log_interval', 1)
    
    # Training loop
    while total_steps < total_timesteps:
        # Storage for current update
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        steps_this_update = 0
        print(f"\nCollecting {update_frequency} steps of experience...")
        
        while steps_this_update < update_frequency:
            # Reset environment
            state = env.reset()
            episode_reward = 0
            done = False
            episode_step = 0
            episode_start_time = time.time()
            
            # Run a single episode until termination or max steps
            while not done:
                # Select action
                with torch.no_grad():
                    # Move state to the training device (GPU)
                    state_tensor = torch.FloatTensor(state).to(train_device)
                    mean, std = ppo_agent.actor(state_tensor)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value = ppo_agent.critic(state_tensor).squeeze()
                
                # Take action in environment
                action_np = action.cpu().numpy()
                next_state, reward, done = env.step(action_np)
                
                # Store data
                states.append(state)
                actions.append(action_np)
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())
                log_probs.append(log_prob.item())
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                steps_this_update += 1
                total_steps += 1
                episode_step += 1
                
                # Save checkpoint if needed
                if total_steps % checkpoint_frequency == 0:
                    ppo_agent.save(f"{save_path}/model_{total_steps}.pt")
                    print(f"Checkpoint saved at step {total_steps}")
                
                # Break out of episode if we've collected enough steps
                if steps_this_update >= update_frequency:
                    break
            
            episode_duration = time.time() - episode_start_time
            episode_rewards.append(episode_reward)
            print(f"Episode completed: steps={episode_step}, reward={episode_reward:.3f}, duration={episode_duration:.2f}s, total_steps={total_steps}")
        
        print(f"Experience collection complete. Total steps: {total_steps}, Steps this update: {steps_this_update}")
        
        # Compute returns using GAE
        with torch.no_grad():
            if done:
                next_value = 0
            else:
                # Move state to the training device (GPU)
                next_state_tensor = torch.FloatTensor(next_state).to(train_device)
                next_value = ppo_agent.critic(next_state_tensor).item()
        
        returns, advantages = ppo_agent.compute_gae(values, rewards, dones, next_value)
        
        # Update PPO agent
        update_start_time = time.time()
        policy_loss, value_loss, entropy = ppo_agent.update(
            states, actions, log_probs, returns, advantages
        )
        update_duration = time.time() - update_start_time
        
        print(f"Network update - time: {update_duration:.2f}s, collected steps: {steps_this_update}")
        print(f"  Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Log progress
        if len(episode_rewards) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"===== Progress report =====")
            print(f"Step: {total_steps}/{total_timesteps} ({total_steps/total_timesteps*100:.1f}%)")
            print(f"Average reward (last {log_interval} episodes): {avg_reward:.2f}")
            print(f"Best average reward so far: {best_average_reward:.2f}")
            print(f"Episodes completed: {len(episode_rewards)}")
            
            if avg_reward > best_average_reward:
                best_average_reward = avg_reward
                ppo_agent.save(f"{save_path}/best_model.pt")
                print(f"New best model saved! Avg reward: {avg_reward:.2f}")
    
    # Save the final model
    ppo_agent.save(f"{save_path}/final_model.pt")
    print(f"Training completed after {total_steps} steps and {len(episode_rewards)} episodes")
    print(f"Final average reward (last {min(log_interval, len(episode_rewards))} episodes): {np.mean(episode_rewards[-min(log_interval, len(episode_rewards)):]):.2f}")
    print(f"Best average reward achieved: {best_average_reward:.2f}")
    
    return ppo_agent

def evaluate_policy(model, config, num_episodes=10, render=True):
    """
    Evaluate a trained policy
    """
    env = make_env(config)
    
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        episode_start_time = time.time()
        
        while not done:
            action = model.select_action(obs, evaluate=True)
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            if render:
                # Depending on your rendering setup, you might need to adjust this
                if hasattr(env.env, 'write_image'):
                    env.env.write_image()
                time.sleep(0.01)  # Add a small delay for visualization
        
        episode_duration = time.time() - episode_start_time
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, Duration: {episode_duration:.2f}s")
    
    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"===== Evaluation Results =====")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min(total_rewards):.2f}, Max reward: {max(total_rewards):.2f}")
    return avg_reward

if __name__ == "__main__":
    # Load configuration
    config_path = f'{CCAI_PATH}/examples/config/allegro_screwdriver_rl.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = yaml.safe_load(pathlib.Path(config_path).read_text())
    
    # Set simulation device to CPU
    config['sim_device'] = 'cpu'
    
    # Train PPO agent
    model = train_ppo(
        config, 
        total_timesteps=config.get('total_timesteps', 1000000),
        save_path=config.get('save_path', f"{CCAI_PATH}/models/allegro_screwdriver_ppo")
    )
    
    # Evaluate the trained policy
    if config.get('evaluate', True):
        evaluate_policy(
            model, 
            config, 
            num_episodes=config.get('eval_episodes', 10),
            render=config.get('render_eval', True)
        )
