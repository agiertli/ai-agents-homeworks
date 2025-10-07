"""
Deep Q-Network (DQN) Agent Implementation

A neural network-based Q-learning agent that can handle larger state spaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Network Agent
    
    DQN uses a neural network to approximate Q-values for states.
    Key features:
    - Experience replay
    - Target network for stability
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_target_every: int = 100,
        hidden_size: int = 128,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of actions
            learning_rate: Learning rate
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            update_target_every: Steps between target network updates
            hidden_size: Hidden layer size
            device: Device to use (cpu/cuda)
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.device = device
        
        # Set seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Neural networks
        self.q_network = DQN(state_size, hidden_size, action_size).to(device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon_history = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(
        self,
        env,
        n_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        verbose: bool = True,
        render_frequency: int = 0
    ) -> Dict[str, List[float]]:
        """
        Train the DQN agent.
        
        Args:
            env: Gymnasium environment
            n_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            render_frequency: Render every n episodes
            
        Returns:
            Training history
        """
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = self._preprocess_state(state)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self._preprocess_state(next_state)
                done = terminated or truncated
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Train
                if len(self.memory) > self.batch_size:
                    self.replay()
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                self.steps += 1
                
                # Update target network
                if self.steps % self.update_target_every == 0:
                    self.update_target_network()
                
                # Render if requested
                if render_frequency > 0 and episode % render_frequency == 0:
                    env.render()
                
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-1000:]) if self.losses else 0
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon_history': self.epsilon_history
        }
    
    def evaluate(
        self,
        env,
        n_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        render: bool = False
    ) -> Dict[str, float]:
        """Evaluate the agent's performance."""
        rewards = []
        lengths = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = self._preprocess_state(state)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                action = self.get_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self._preprocess_state(next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
                
                if terminated or truncated:
                    break
            
            rewards.append(total_reward)
            lengths.append(steps)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'total_episodes': n_episodes
        }
    
    def _preprocess_state(self, state):
        """Preprocess state for neural network."""
        if isinstance(state, int):
            # One-hot encoding for discrete states
            one_hot = np.zeros(self.state_size)
            one_hot[state] = 1
            return one_hot
        else:
            # Flatten if necessary
            return state.flatten()
    
    def save(self, filepath: str):
        """Save model and training data."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training data."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.6)
        if len(self.episode_rewards) > 100:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(100)/100, mode='valid')
            ax.plot(range(99, len(self.episode_rewards)), 
                   moving_avg, 'r-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True)
        
        # Losses
        ax = axes[0, 1]
        if self.losses:
            ax.plot(self.losses, alpha=0.6)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True)
        
        # Episode lengths
        ax = axes[1, 0]
        ax.plot(self.episode_lengths, alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.grid(True)
        
        # Epsilon
        ax = axes[1, 1]
        ax.plot(self.epsilon_history)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()

