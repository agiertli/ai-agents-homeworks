"""
Q-Learning Agent Implementation

A simple tabular Q-learning agent that learns to navigate grid worlds.
"""

import numpy as np
import pickle
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning Agent using Q-table
    
    Q-learning is a model-free, off-policy TD control algorithm.
    It learns the optimal action-value function Q*(s,a).
    
    Update rule:
    Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of states in the environment
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            seed: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def get_action(self, state: int, training: bool = True) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.q_table[state])
    
    def update(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool
    ) -> None:
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Calculate target
        if done:
            target = reward
        else:
            # Max Q-value for next state
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self) -> None:
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
        Train the agent in the environment.
        
        Args:
            env: Gymnasium environment
            n_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            render_frequency: Render every n episodes (0 = never)
            
        Returns:
            Training history dictionary
        """
        for episode in range(n_episodes):
            state, _ = env.reset()
            if hasattr(state, '__len__'):  # Handle array observations
                state = self._state_to_index(state, env)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                if hasattr(next_state, '__len__'):
                    next_state = self._state_to_index(next_state, env)
                
                # Update Q-table
                self.update(state, action, reward, next_state, terminated or truncated)
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render_frequency > 0 and episode % render_frequency == 0:
                    env.render()
                
                if terminated or truncated:
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
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }
    
    def evaluate(
        self,
        env,
        n_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            env: Gymnasium environment
            n_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        rewards = []
        lengths = []
        successes = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            if hasattr(state, '__len__'):
                state = self._state_to_index(state, env)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Get action (no exploration)
                action = self.get_action(state, training=False)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                if hasattr(next_state, '__len__'):
                    next_state = self._state_to_index(next_state, env)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
                
                if terminated or truncated:
                    # Check if goal was reached (success)
                    successes.append(terminated and reward > 0)
                    break
            
            rewards.append(total_reward)
            lengths.append(steps)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'success_rate': np.mean(successes) if successes else 0.0,
            'total_episodes': n_episodes
        }
    
    def _state_to_index(self, state: np.ndarray, env) -> int:
        """Convert continuous state to discrete index."""
        if hasattr(env, 'grid_size'):
            # For GridWorld with (x, y) coordinates
            return state[0] * env.grid_size + state[1]
        else:
            # For other environments, assume state is already an index
            return int(state)
    
    def save(self, filepath: str) -> None:
        """Save Q-table and agent parameters."""
        save_dict = {
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Q-table and agent parameters."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.q_table = save_dict['q_table']
        self.n_states = save_dict['n_states']
        self.n_actions = save_dict['n_actions']
        self.learning_rate = save_dict['learning_rate']
        self.discount_factor = save_dict['discount_factor']
        self.epsilon = save_dict['epsilon']
        self.episode_rewards = save_dict.get('episode_rewards', [])
        self.episode_lengths = save_dict.get('episode_lengths', [])
        print(f"Agent loaded from {filepath}")
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        # Moving average
        window = min(100, len(self.episode_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.episode_rewards)), 
                   moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True)
        
        # Episode lengths
        ax = axes[0, 1]
        ax.plot(self.episode_lengths, alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.grid(True)
        
        # Epsilon decay
        ax = axes[1, 0]
        ax.plot(self.epsilon_history)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate Decay')
        ax.grid(True)
        
        # Q-table heatmap
        ax = axes[1, 1]
        im = ax.imshow(self.q_table.T, cmap='viridis', aspect='auto')
        ax.set_xlabel('State')
        ax.set_ylabel('Action')
        ax.set_title('Q-Table Heatmap')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.q_table, axis=1)
    
    def visualize_policy(self, env, grid_size: int) -> None:
        """Visualize the learned policy on a grid."""
        policy = self.get_policy()
        
        # Action symbols
        action_symbols = ['↑', '→', '↓', '←']
        
        # Create policy grid
        policy_grid = np.empty((grid_size, grid_size), dtype=object)
        
        for i in range(grid_size):
            for j in range(grid_size):
                state_idx = i * grid_size + j
                if state_idx < len(policy):
                    policy_grid[i, j] = action_symbols[policy[state_idx]]
                else:
                    policy_grid[i, j] = '?'
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Background colors based on environment
        if hasattr(env, 'grid'):
            for i in range(grid_size):
                for j in range(grid_size):
                    if env.grid[i, j] == -1:  # Wall
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                 facecolor='black'))
                    elif env.grid[i, j] == -5:  # Danger
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                 facecolor='red', alpha=0.5))
                    elif env.grid[i, j] == 5:  # Treasure
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                 facecolor='yellow', alpha=0.5))
                    elif (i, j) == env.goal_pos:  # Goal
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                 facecolor='green', alpha=0.5))
        
        # Plot policy arrows
        for i in range(grid_size):
            for j in range(grid_size):
                if policy_grid[i, j] != '?':
                    ax.text(j, i, policy_grid[i, j], ha='center', va='center',
                           fontsize=20, fontweight='bold')
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True)
        ax.invert_yaxis()
        ax.set_title('Learned Policy', fontsize=16)
        
        plt.tight_layout()
        plt.show()

